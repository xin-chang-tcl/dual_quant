import torch.ao.quantization
import torch.nn as nn
from dual_quant.implementation.core.components.dual_quantizer import DualQuantizer, disable_fake_quant, enable_fake_quant

def compute_scale(weight):
    return torch.amax(torch.abs(weight), dim=(1, 2, 3))

# Define the CustomConv class with bias handling
def modify_convs(dconv, conv):
    # Compute the scaling factors
    conv_channel_max = torch.abs(dconv.weight.data).max()
    conv_scale1 = 1 / conv_channel_max

    # Scale the conv weights and bias (if present)

    dconv.weight.data.mul_(conv_scale1)
    if dconv.bias is not None:
        dconv.bias.data.mul_(conv_scale1)
    conv.weight.data.mul_(conv_channel_max)
    return dconv, conv

class CustomConvNormal(nn.Module):
    def __init__(self, conv):
        super(CustomConvNormal, self).__init__()
        self.conv = conv
        # Get the optimal scaling (assuming this is already defined elsewhere)
        # self.scale = compute_optimal_scaling(
        #     conv.weight.data).detach()  # Should return tensor of shape [64] or [conv.out_channels]
        channel_max = torch.max(torch.abs(conv.weight.data))
        self.affine_scale = channel_max  # Should return tensor of shape [64] or [conv.out_channels]
        # Convert to float and calculate the inverse
        self.scale1 = 1 / self.affine_scale  # This should still be shape [64]
        self.affine_scale.requires_grad = False
        # Scale the weights (expand self.scale1 along spatial dimensions)
        #with torch.no_grad():
        self.conv.weight.data =self.conv.weight.data * (self.scale1.view(-1, 1, 1, 1))  # Expand only along dimensions that are not channels

    def forward(self, x):
        # Apply the convolution first
        x = x * self.affine_scale.view(1, -1, 1, 1)
        x = self.conv(x)

        return x


class CustomConv(nn.Module):
    def __init__(self, conv):
        super(CustomConv, self).__init__()
        self.conv = conv
        # Get the optimal scaling (assuming this is already defined elsewhere)
        # self.scale = compute_optimal_scaling(
        #     conv.weight.data).detach()  # Should return tensor of shape [64] or [conv.out_channels]
        channel_max = torch.amax(torch.abs(conv.weight.data), dim=(1, 2, 3))
        self.affine_scale = channel_max  # Should return tensor of shape [64] or [conv.out_channels]
        # Convert to float and calculate the inverse
        self.scale1 = 1 / (self.affine_scale+1e-16)  # This should still be shape [64]
        self.affine_scale.requires_grad = False
        # Scale the weights (expand self.scale1 along spatial dimensions)
        #with torch.no_grad():
        self.conv.weight.data =self.conv.weight.data * (self.scale1.view(-1, 1, 1, 1))  # Expand only along dimensions that are not channels

    def forward(self, x):
        # Apply the convolution first
        x = x * self.affine_scale.view(1, -1, 1, 1)
        x = self.conv(x)
        return x

def scaled_sigmoid(x, min_val=1e-10, max_val=1 - 1e-4):
    # Standard sigmoid
    sigmoid = torch.sigmoid(x)
    # Scale sigmoid to the range (min_val, max_val)
    scaled_sigmoid = min_val + (max_val - min_val) * sigmoid
    return scaled_sigmoid

def scale_hook_fn_act_switch(module, name, input, idx, every, total_idx, name_list, enhance=False):
    if isinstance(module, DualQuantizer):
        if not module.not_used:
            if module.quant_max + module.quant_min != 0:
                if len(total_idx) >= (idx+1)*every:
                    module.apply(disable_fake_quant)
                    module.observer_enabled[0] = 0
                    total_idx.append(1)
                    module.scale.requires_grad = False
                    module.to_compute = False
                    name_list.append(name)
                elif len(total_idx) < (idx+1)*every and len(total_idx) >= idx*every:
                    module.apply(enable_fake_quant)
                    module.observer_enabled[0] = 0
                    total_idx.append(1)
                    module.scale.requires_grad = False
                    module.to_compute = False
                    module.switch_scale(enhance)
                    name_list.append(name)
                else:
                    module.apply(enable_fake_quant)
                    module.observer_enabled[0] = 0
                    total_idx.append(1)
                    module.scale.requires_grad = False
                    module.to_compute = False
                    name_list.append(name)
            else:
                module.apply(enable_fake_quant)
                module.observer_enabled[0] = 0
                #total_idx.append(1)
                module.scale.requires_grad = False
                module.to_compute = False


def scale_hook_fn_act_save(module, name, input, idx, every, total_idx, name_list, enhance=False):
    if isinstance(module, DualQuantizer):
        if not module.not_used:
            if module.quant_max + module.quant_min != 0:
                if len(total_idx) >= (idx+1)*every:
                    module.apply(disable_fake_quant)
                    module.observer_enabled[0] = 0
                    total_idx.append(1)
                    module.scale.requires_grad = False
                    module.to_compute = False
                    name_list.append(name)
                elif len(total_idx) < (idx+1)*every and len(total_idx) >= idx*every:
                    module.apply(enable_fake_quant)
                    module.observer_enabled[0] = 0
                    total_idx.append(1)
                    module.scale.requires_grad = False
                    module.to_compute = False
                    module.scale_list.append(module.scale.detach().clone())
                    name_list.append(name)
                else:
                    module.apply(enable_fake_quant)
                    module.observer_enabled[0] = 0
                    total_idx.append(1)
                    module.scale.requires_grad = False
                    module.to_compute = False
                    name_list.append(name)
            else:
                module.apply(enable_fake_quant)
                module.observer_enabled[0] = 0
                module.scale.requires_grad = False
                module.to_compute = False


def scale_hook_fn_act(module, name, input, idx, every, total_idx, name_list, enhance=False):
    if not name in name_list:
        if isinstance(module, DualQuantizer):
            if not module.not_used:
                if module.quant_max + module.quant_min != 0:
                    if len(total_idx) >= (idx+1)*every:
                        module.apply(disable_fake_quant)
                        module.observer_enabled[0] = 0
                        total_idx.append(0)
                        module.scale.requires_grad = False
                        module.to_compute = False
                        name_list.append(name)
                    elif len(total_idx) < (idx+1)*every and len(total_idx) >= idx*every:
                        module.apply(enable_fake_quant)
                        module.observer_enabled[0] = 1
                        module.enhance = enhance
                        total_idx.append(1)
                        module.scale.requires_grad = True
                        module.to_compute = True
                        name_list.append(name)
                    else:
                        module.apply(enable_fake_quant)
                        module.observer_enabled[0] = 0
                        total_idx.append(0)
                        module.scale.requires_grad = False
                        module.to_compute = False
                        name_list.append(name)
                else:
                    module.apply(enable_fake_quant)
                    module.observer_enabled[0] = 0
                    module.scale.requires_grad = False
                    module.to_compute = False

def set_trainable_fn(module, input, total_trainable, scale_ids):
    if isinstance(module, DualQuantizer):
        if not module.not_used:
            scale_id = id(module.scale)  # Get the unique ID of the scale
            if scale_id not in scale_ids:  # Check if this ID is already recorded
                total_trainable.append(module.scale)
                scale_ids.append(scale_id)  # Add the scale ID to the set

def set_trainable_fn_act(module, input, total_trainable, scale_ids):
    if isinstance(module, DualQuantizer):
        if not module.not_used and module.quant_max + module.quant_min != 0:
            scale_id = id(module.scale)  # Get the unique ID of the scale
            if scale_id not in scale_ids:  # Check if this ID is already recorded
                total_trainable.append(module.scale)
                scale_ids.append(scale_id)


def scale_hook_fn(module, name, input, idx, every, enhance_list, name_list, enhance=False):
    if isinstance(module, DualQuantizer):
        if not module.not_used and not module.float:
            if len(enhance_list) >= (idx+1)*every:
                module.apply(disable_fake_quant)
                module.observer_enabled[0] = 0
                enhance_list.append(0)
                module.scale.requires_grad = False
                module.to_compute = False
                name_list.append(name)
            elif len(enhance_list) < (idx+1)*every and len(enhance_list) >= idx*every:
                module.apply(enable_fake_quant)
                module.observer_enabled[0] = 1
                enhance_list.append(1)
                module.scale.requires_grad = True
                module.to_compute = True
                module.enhance = enhance
                name_list.append(name)
            else:
                module.apply(enable_fake_quant)
                module.observer_enabled[0] = 0
                enhance_list.append(0)
                module.scale.requires_grad = False
                module.to_compute = False
                name_list.append(name)

def register_trainble_hook(total_trainable, model, hook_fn, total_ids):
    hooks = []
    for layer1 in model.modules():
        if isinstance(layer1, DualQuantizer):
            def scale_hook_fn(module, input, output, hook_fn=hook_fn, total_trainable=total_trainable, total_ids=total_ids):
                hook_fn(module, input, total_trainable, total_ids)

            hooks.append(layer1.register_forward_hook(scale_hook_fn))
        else:
            continue
    return hooks

def register_scale_hook(total_idx, name_list, model, hook_fn, idx, every, enhance):
    hooks = []
    for name, layer1 in model.named_modules():
        if isinstance(layer1, DualQuantizer):
            def scale_hook_fn(module, input, output, hook_fn=hook_fn, idx=idx, every=every, total_idx=total_idx, name=name, name_list=name_list, enhance=enhance):
                hook_fn(module, name, input, idx, every, total_idx, name_list, enhance=enhance)

            hooks.append(layer1.register_forward_hook(scale_hook_fn))
        else:
            continue
    return hooks

def register_scale_hook_act(total_idx, name_list, model, hook_fn, idx, every, enhance):
    hooks = []
    for name, layer1 in model.named_modules():
        if isinstance(layer1, DualQuantizer):
            if layer1.quant_max + layer1.quant_min != 0:
                def scale_hook_fn(module, input, output, hook_fn=hook_fn, idx=idx, every=every, total_idx=total_idx, name=name, name_list=name_list, enhance=enhance):
                    hook_fn(module, name, input, idx, every, total_idx, name_list, enhance=enhance)

                hooks.append(layer1.register_forward_hook(scale_hook_fn))
        else:
            continue
    return hooks

def dual_loss_hook_fn(loss_values, scales, zeropoints, module, input):
    if isinstance(module, DualQuantizer):
        if not module.not_used and not module.float:
            clip_value, round_value = module.compute_reg_loss(input[0])
            if module.quant_max + module.quant_min == 0:
                if not module.is_per_channel:
                    scales.append(scaled_sigmoid(module.scale, module.lowest_scale))
                    zeropoints.append(module.zero_point)
            else:
                scales.append(scaled_sigmoid(module.scale, module.lowest_scale))
                zeropoints.append(module.zero_point)
            loss_values.append([clip_value, round_value])
            # else:
            #     if module.quant_max + module.quant_min == 0:
            #         if not module.is_per_channel:
            #             scales.append(scaled_sigmoid(module.scale, module.lowest_scale))
            #             zeropoints.append(module.zero_point)
            #     else:
            #         scales.append(scaled_sigmoid(module.scale, module.lowest_scale))
            #         zeropoints.append(module.zero_point)
        # else:
        #     if module.record:
        #         if module.quant_max + module.quant_min == 0:
        #             loss_values.append([torch.tensor(0), torch.tensor(0)])
        #             if not module.is_per_channel:
        #                 scales.append(module.scale)
        #                 zeropoints.append(module.zero_point)
        #         else:
        #             loss_values.append([torch.tensor(0), torch.tensor(0)])
        #             scales.append(module.scale)
        #             zeropoints.append(module.zero_point)


def register_hooks_dual_loss(model, hook_fn, loss_values, scales, zeropoints):
    hooks = []
    for layer in model.modules():
        if isinstance(layer, torch.ao.quantization.FakeQuantizeBase):
            def dual_loss_hook_fn(module, input, output, hook_fn=hook_fn,loss_values=loss_values, scales=scales, zeropoints=zeropoints):
                hook_fn(loss_values, scales, zeropoints, module, input)

            hooks.append(layer.register_forward_hook(dual_loss_hook_fn))
        else:
            continue
    return hooks


import torch.nn as nn
import torch_dag as td

def fuse_for_conv_dconv(vertex):
    if isinstance(vertex.module, nn.Conv2d):
        first = vertex
        if isinstance(first.module, nn.Conv2d):
            if vertex.module.groups != 1:
                if len(vertex.successors) == 1:
                    second = vertex.successors[0]
                    if isinstance(second.module, nn.Conv2d):
                        if second.module.groups == 1:
                            conv1 = first.module
                            conv2 = second.module
                            new_conv1, new_conv2 = modify_convs(conv1, conv2)
                            first.module = new_conv1
                            second.module = new_conv2

                    else:
                        if len(second.successors) == 1:
                            third = second.successors[0]
                            if isinstance(third.module, nn.Conv2d):
                                if third.module.groups == 1:
                                    conv1 = first.module
                                    conv2 = third.module
                                    new_conv1, new_conv2 = modify_convs(conv1, conv2)
                                    first.module = new_conv1
                                    third.module = new_conv2

def find_conv(vertex):
    if isinstance(vertex.module, nn.Conv2d):
        if vertex.module.groups == 1:
            first = vertex
            new_conv = CustomConvNormal(first.module)
            first.module = new_conv

def find_dconv(vertex):
    if isinstance(vertex.module, nn.Conv2d):
        if vertex.module.groups != 1:
            first = vertex
            new_conv = CustomConv(first.module)
            first.module = new_conv

def find_conv_bn(vertex):
    if isinstance(vertex.module, nn.Conv2d):
        first = vertex
        second = vertex.successors[0]
        if isinstance(second.module, td.core.dag_module.DagModule):
            for idx, module in enumerate(second.module.inner_modules):
                if isinstance(module, nn.BatchNorm2d):
                    conv = first.module
                    bn = module

                    # Extract parameters
                    conv_weight = conv.weight
                    conv_bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
                    bn_weight = bn.weight
                    bn_bias = bn.bias
                    bn_mean = bn.running_mean
                    bn_var = bn.running_var
                    bn_eps = bn.eps

                    # Calculate new parameters
                    invstd = 1 / torch.sqrt(bn_var + bn_eps)
                    new_weight = conv_weight * (bn_weight * invstd).reshape([-1, 1, 1, 1])
                    new_bias = (conv_bias - bn_mean) * invstd * bn_weight + bn_bias

                    # Create new Conv2d layer
                    new_conv = nn.Conv2d(
                        in_channels=conv.in_channels,
                        out_channels=conv.out_channels,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                        padding=conv.padding,
                        dilation=conv.dilation,
                        groups=conv.groups,
                        bias=True  # New conv layer will have bias
                    )

                    # Assign new parameters
                    new_conv.weight.data.copy_(new_weight)
                    new_conv.bias.data.copy_(new_bias)
                    first.module = new_conv
                    td.remove_vertex(second.module, vertex=second.module.inner_vertices[0])

        elif isinstance(second.module, torch.nn.BatchNorm2d):
            conv = first.module
            bn = second.module

            # Extract parameters
            conv_weight = conv.weight
            conv_bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels)
            bn_weight = bn.weight
            bn_bias = bn.bias
            bn_mean = bn.running_mean
            bn_var = bn.running_var
            bn_eps = bn.eps

            # Calculate new parameters
            invstd = 1 / torch.sqrt(bn_var + bn_eps)
            new_weight = conv_weight * (bn_weight * invstd).reshape([-1, 1, 1, 1])
            new_bias = (conv_bias - bn_mean) * invstd * bn_weight + bn_bias

            # Create new Conv2d layer
            new_conv = nn.Conv2d(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True  # New conv layer will have bias
            )

            # Assign new parameters
            new_conv.weight.data.copy_(new_weight)
            new_conv.bias.data.copy_(new_bias)
            first.module = new_conv
            td.remove_vertex(second.dag_module, vertex=second)


def combine_convolutions(conv1, conv2):
    # Extract weights and biases from both convolutional layers
    w1, b1 = conv1.weight.data, conv1.bias.data
    w2, b2 = conv2.weight.data, conv2.bias.data

    # Ensure the dimensions match for matrix multiplication
    # w1 shape: (out_channels_1, in_channels_1, 1, 1)
    # w2 shape: (out_channels_2, in_channels_1, 1, 1)
    # Check and reshape if needed to perform matrix multiplication
    w1_reshaped = w1.view(w1.size(0), -1)  # Reshaping to (out_channels_1, in_channels_1)
    w2_reshaped = w2.view(w2.size(0), -1)  # Reshaping to (out_channels_2, in_channels_1)

    # Matrix multiplication for combining weights
    # Resultant weight should have shape (out_channels_2, in_channels_1)
    new_weight = torch.mm(w2_reshaped, w1_reshaped)

    # Combine biases, new_bias = b2 + (w2 * b1), summed over the appropriate dimensions
    # Since b1 is expanded and multiplied by every corresponding weight of w2, then summed
    new_bias = b2 + torch.mv(w2_reshaped, b1)

    # Create a new convolutional layer with the combined parameters
    # Input channels from conv1, output channels from conv2
    new_model = nn.Conv2d(conv1.in_channels, conv2.out_channels, kernel_size=1, stride=1)
    new_model.weight.data = new_weight.view(conv2.out_channels, conv1.in_channels, 1, 1)
    new_model.bias.data = new_bias
    return new_model


def delete_empty(vertex):
    if isinstance(vertex.module, td.core.dag_module.DagModule):
        if len(vertex.module.inner_vertices) == 0:
            td.remove_vertex(vertex.dag_module, vertex=vertex)

