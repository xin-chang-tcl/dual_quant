import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.ao.quantization.observer import MovingAverageMinMaxObserver
import re


def _is_per_channel(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_channel_symmetric, torch.per_channel_affine, torch.per_channel_affine_float_qparams]


def _is_per_tensor(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]


def _is_float_qparams(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_channel_affine_float_qparams, ]


class PingPongQuantizer(FakeQuantizeBase):
    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, init_scale=False, lowest_scale=1e-5, observer=MovingAverageMinMaxObserver, quant_min=None, quant_max=None, **observer_kwargs):
        super().__init__()
        # Populate quant_min/quant_max to observer_kwargs if valid
        if quant_min is not None and quant_max is not None:
            assert quant_min <= quant_max, \
                'quant_min must be less than or equal to quant_max'
            dtype = observer_kwargs.get("dtype", torch.quint8)
            if hasattr(observer, "p"):
                # In case observer is _PartialWrapper, dtype can be stored in
                # observer.p.keywords["dtype"]
                dtype = getattr(getattr(observer, "p", {}), "keywords", {}).get(
                    "dtype", dtype
                )
            assert torch.iinfo(dtype).min <= quant_min, 'quant_min out of bound'
            assert quant_max <= torch.iinfo(dtype).max, 'quant_max out of bound'
            observer_kwargs.update({"quant_min": quant_min, "quant_max": quant_max})
        self.activation_post_process = observer(**observer_kwargs)
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        if _is_float_qparams(self.activation_post_process.qscheme):
            zero_point_dtype = torch.float
        else:
            zero_point_dtype = torch.int
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=zero_point_dtype))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        assert _is_per_channel(self.qscheme) or \
            _is_per_tensor(self.qscheme), \
            'Only per channel and per tensor quantization are supported in fake quantize' + \
            ' got qscheme: ' + str(self.qscheme)
        self.is_per_channel = _is_per_channel(self.qscheme)
        self.init_scales = init_scale
        self.lowest_scale = lowest_scale
        self.not_used = False

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    @torch.jit.export
    def calculate_saving(self):
        return self.scale, self.zero_point

    @torch.jit.export
    def calculate_running(self):
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
        if (self.observer_enabled[0] == 1):
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(X.device), _zero_point.to(X.device)
            if not self.init_scales:
                if self.scale.shape != _scale.shape:
                    self.scale.resize_(_scale.shape)
                self.scale.copy_(_scale)
            if self.zero_point.shape != _zero_point.shape:
                self.zero_point.resize_(_zero_point.shape)
            self.zero_point.copy_(_zero_point)
        if self.fake_quant_enabled[0] == 1:

            if self.init_scales:
                scale = scaled_sigmoid(self.scale, min_val=self.lowest_scale).detach()
            else:
                scale = self.scale
            if self.is_per_channel:
                X = torch.fake_quantize_per_channel_affine(
                    X, scale, self.zero_point,
                    self.ch_axis, self.activation_post_process.quant_min, self.activation_post_process.quant_max)
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X, scale, self.zero_point,
                    self.activation_post_process.quant_min, self.activation_post_process.quant_max)
        return X

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.fake_quant_enabled, self.observer_enabled,
                   self.activation_post_process.quant_min, self.activation_post_process.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == 'scale':
                    self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def compute_dual_loss(self, X):
        if len(X.shape) == 2:
            flag = True
        else:
            flag = False
        if self.is_per_channel:
            ch_axis = self.activation_post_process.ch_axis
            # compute the scale and zero-point per channel
            scale_shape = [1] * X.dim()
            scale_shape[ch_axis] = -1
            scale = scaled_sigmoid(self.scale, min_val=self.lowest_scale).view(*scale_shape)
            zero_point_shape = [1] * X.dim()
            zero_point_shape[ch_axis] = -1
            zero_point = self.zero_point.view(*zero_point_shape)
            # perform per-channel quantization
            X = X / scale
            # Element-wise addition of zero_point
            X = X + zero_point

        else:
            scale = scaled_sigmoid(self.scale, min_val=self.lowest_scale)
            X = X / scale
            X = X + self.zero_point.float()

        X_clip_left = torch.where(X < self.activation_post_process.quant_min,
                                  (self.quant_min - X),
                        torch.where((X >= self.activation_post_process.quant_min) & (
                                X <= self.activation_post_process.quant_max),
                                    0,
                                    0))


        X_clip_right = torch.where(X < self.activation_post_process.quant_min,
                        0,
                        torch.where((X >= self.activation_post_process.quant_min) & (
                                X <= self.activation_post_process.quant_max),
                                    0,
                                    (X - self.quant_max)))
        X_round = torch.where(X < self.activation_post_process.quant_min,
                        0,
                        torch.where((X >= self.activation_post_process.quant_min) & (
                                X <= self.activation_post_process.quant_max),
                                    (-torch.log(1 - scale)),
                                    0))

        loss_left = torch.sum(X_clip_left)
        loss_right = torch.sum(X_clip_right)
        loss_round = torch.sum(X_round)
        if flag:
            loss_round = 0
        return (loss_right+loss_left)/torch.prod(torch.tensor(X.shape)), loss_round/torch.prod(torch.tensor(X.shape))

    def forward_finetune(self, X):
        if self.fake_quant_enabled[0] == 1:
            if self.is_per_channel:
                X = torch.fake_quantize_per_channel_affine(
                    X, self.scale, self.zero_point,
                    self.ch_axis, self.activation_post_process.quant_min, self.activation_post_process.quant_max)
            else:
                X = torch.fake_quantize_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.activation_post_process.quant_min, self.activation_post_process.quant_max)
        return X

    def init_grad_scaling(self):
        if not self.init_scales:
            self.scale = torch.clamp(self.scale, self.lowest_scale, 1 - 1e-4)
            self.scale = torch.nn.Parameter(logit(self.scale), requires_grad=True)
            self.init_scales = True
        self.calculate_qparams = self.calculate_running

    def modify_forward(self):
        self.forward = self.forward_finetune

    def resume_grad_scaling(self):
        if self.init_scales:
            self.scale.requires_grad = True
            self.calculate_qparams = self.calculate_running

    def disable_grad_scaling(self):
        if self.init_scales:
            self.scale.requires_grad = False
            self.calculate_qparams = self.calculate_saving

    def enable_fake_quant(self):
        self.fake_quant_enabled = torch.tensor([1], dtype=torch.uint8)

    def disable_fake_quant(self):
        self.fake_quant_enabled = torch.tensor([0], dtype=torch.uint8)

    def enable_observer(self):
        self.observer_enabled = torch.tensor([1], dtype=torch.uint8)

    def disable_observer(self):
        self.observer_enabled = torch.tensor([0], dtype=torch.uint8)

    def prepare_for_convert(self):
        scale = self.scale.clone().detach()
        del self.scale
        if self.init_scales:
            self.register_buffer('scale', torch.tensor(scaled_sigmoid(scale, min_val=self.lowest_scale), dtype=torch.float))
        else:
            self.register_buffer('scale', torch.tensor(scale, dtype=torch.float))
        if hasattr(self, 'qvalue'):
            self.qvalue = None
        self.init_scales = False
        self.calculate_qparams = self.calculate_saving
        self.forward = self.forward_finetune


def scaled_sigmoid(x, min_val=1e-4, max_val=1 - 1e-4):
    # Standard sigmoid
    sigmoid = torch.sigmoid(x)
    # Scale sigmoid to the range (min_val, max_val)
    scaled_sigmoid = min_val + (max_val - min_val) * sigmoid
    return scaled_sigmoid


def reverse_scaled_sigmoid(y, min_val=1e-4, max_val=1 - 1e-4):
    # Reverse the scaling
    reversed_scaling = (y - min_val) / (max_val - min_val)
    # Apply the logit (inverse of sigmoid)
    reversed_sigmoid = torch.logit(reversed_scaling)
    return reversed_sigmoid


def init_grad_scaling(mod):
    """
    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if isinstance(mod, PingPongQuantizer):
        if mod.not_used == False:
            mod.init_grad_scaling()


def resume_grad_scaling(mod):
    """
    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if isinstance(mod, PingPongQuantizer):
        if mod.not_used == False:
            mod.resume_grad_scaling()

def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())

def logit(y):
    return torch.log(y) - torch.log(1-y)


def enable_fake_quant(mod):
    """
    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if isinstance(mod, PingPongQuantizer) or _is_fake_quant_script_module(mod):
        if mod.not_used == False:
            mod.enable_fake_quant()


def disable_fake_quant(mod):
    """
    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if isinstance(mod, PingPongQuantizer) or _is_fake_quant_script_module(mod):
        if mod.not_used == False:
            mod.disable_fake_quant()


def enable_observer(mod):
    """
    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if isinstance(mod, PingPongQuantizer) or _is_fake_quant_script_module(mod):
        if mod.not_used == False:
            mod.enable_observer()


def disable_observer(mod):
    """
    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if isinstance(mod, PingPongQuantizer) or _is_fake_quant_script_module(mod):
        if mod.not_used == False:
            mod.disable_observer()


def disable_grad_scaling(mod):
    """
    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if isinstance(mod, PingPongQuantizer) or _is_fake_quant_script_module(mod):
        if mod.not_used == False:
            mod.disable_grad_scaling()


def prepare_for_convert(mod):
    """
    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if isinstance(mod, PingPongQuantizer) or _is_fake_quant_script_module(mod):
        mod.prepare_for_convert()


def _is_fake_quant_script_module(mod):
    ''' Returns true if given mod is an instance of FakeQuantize script module.
    '''
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # qualified name looks like '__torch__.torch.ao.quantization.fake_quantize.___torch_mangle_2.FakeQuantize'
        suffix = mod._c.qualified_name.split('.', 1)[1]
        name = re.sub(r'\.___torch_mangle_\d+', '', suffix)
        return name == 'torch.ao.quantization.fake_quantize.FakeQuantize' or \
            name == 'torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize'
    return False
