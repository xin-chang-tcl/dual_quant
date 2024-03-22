## CONTENT 
This repository contains the PyTorch implementation for

**Dynamic Data-Driven Quantization Range
Pinpointing: A Ping Pong Ball Approach for
Precision Adjustments** 


## Algorithm implementation
The algorithm is implemented in the `class PingPongQuantizer` at `implementation/core/components/pingpong_quantizer.py`

We exploits TinyNN to prepare the quantizer inserted model because it provides better graph operations, and it supports
the convertion to TFlite. 

The class inherits from the base `class FakeQuantizeBase`, which is the base class for PTQ/QAT purpose from pytorch.
The dual regualarization is implemented in the `compute_reg_loss` function:




```python
    def compute_reg_loss(self, X):
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
        return (loss_right+loss_left)/torch.prod(torch.tensor(X.shape)), loss_round/torch.prod(torch.tensor(X.shape))
```

The functions listed below are used to initialize the scale parameters for training,
The `lowest_scale` parameter are used for converting to TFlite purpose, as
special outputs for TFlite conversion might experience some numerical issues. Some
boundaries are preset, however, it doesn't affect the pytorch metrics.
```python
    def init_grad_scaling(self):
        if not self.init_scales:
            self.scale = torch.clamp(self.scale, self.lowest_scale, 1 - 1e-4)
            self.scale = torch.nn.Parameter(logit(self.scale), requires_grad=True)
            self.init_scales = True
        self.calculate_qparams = self.calculate_running
```
`init_grad_scaling` 
Basically makes the scale parameters trainable, in this implementation, we used sigmoid function
to make sure that the scale parameters are in the range of (0,1), it can avoid a lot of numerical issues
that might appear in clipping solution.

```python
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
```
The functions above mostly control the conversion of scales, since during training they are converted, we will need to convert
them back to the original scales for Pytorch or TFlite conversion purpose.

## ImageNet Example
```python
import torch.fx
from torch.quantization.quantize_fx import fuse_fx
from pingpong_quant.implementation.core.utils.training_utils import get_parameters
import torchvision
from pingpong_quant.implementation.core.utils.model_utils import get_insert_fake_quant_model, init_grad_scaling


if __name__ == '__main__':
    #lowest_scale is only used for converting to TFlite purpose, so far we only 
    # experienced numerical issues with mobilenetv3_small, in this case it needs to be set 
    # to 1e-4, for other models, 1e-10 is enough.
    config = {
        'per_channel': True,
        'lowest_scale': 1e-10,
        'training_type': 'ptq'
    }
    #Create your own ImageNet dataloader
    # ........
    # ........
    # Load the model and fuse the batchnorm layers
    model = torchvision.models.efficientnet_b0(pretrained=True)
    model = fuse_fx(model.eval())
    # Insert the quantization layers
    fake_quant_model = get_insert_fake_quant_model(model,(1,3,224,224), config, val_dataloader)
    
    #PTQ calibration
    #initialization of scale and zero_point should be performed at this stage as described in the paper,
    #usually hunderds of samples are enough for the calibration just like normal PTQ procedure.
    #Simple do:
    model.eval()
    for inputs, _ in val_dataloader:
        _ = model(inputs)
    #Initialize scales for training
    #The function below will convert the scale parameters to trainable state
    # based on the initial values from calibration.
    model.apply(init_grad_scaling)
    
    #Since we perform only PTQ training without changing the weights, 
    #get_parameters function will return only the scale parameters,
    #setting the training_type to `qat` will return all parameters of the quantizer inserted model.
    paramters = get_parameters(model, config)
    
    #Set the optimizer, a lr of 0.01 is suitable for most cases, however, for those case that
    #the quantization is easy, meanning that the original MAX solution is almost sufficient, a lower lr,
    #e.g.: 1e-4 is prefered to make gain the better performance.
    optimizer = torch.optim.Adam(paramters, lr=0.01, momentum=0.9)
    
    #Training the model with your original ImageNet pipline.
    #Note that the loss function should be replaced with the dual_loss function taken from hook functions to collect
    #layer-wise loss.
    from pingpong_quant.implementation.core.components.hooks import register_hooks_dual_loss, dual_loss_hook_fn
    from pingpong_quant.implementation.core.components.loss import compute_dual_loss
    dual_loss = []
    hook_loss = register_hooks_dual_loss(model, dual_loss_hook_fn, dual_loss)
    #forward the inputs
    _ = model(inputs)
    #Get the loss from the hook function and do backpropagation.
    dual_loss = compute_dual_loss(dual_loss)
    #...
    #...
    #Empty the data for current batch
    dual_loss = []
    for hook in hook_loss:
        hook.remove()
    #backpropagation
```
Note that during evaluation, the observers needs to be turned off, to not contaminate the zeropoint calculation.
Use:

```python
## Before evaluation
model.apply(torch.quantization.disable_observer)
## After evaluation and before training
model.apply(torch.quantization.enable_observer)
```

## Evaluation of provided checkpoints

Due to the fact that pytorch doesn't support serialization of their own fake-quantized models, the provided checkpoints
are the dictionary of the state_dict of the quantizer inserted model.

To load them, additional weight path should be provided:
```python
import torch.fx
from torch.quantization.quantize_fx import fuse_fx
from pingpong_quant.implementation.core.utils.training_utils import get_parameters
import torchvision
from pingpong_quant.implementation.core.utils.model_utils import get_insert_fake_quant_model, init_grad_scaling

    config = {
        'per_channel': True,
        'lowest_scale': 1e-10,
        'training_type': 'ptq'
        'weight_path': 'path_to_checkpoint'
    }
    #Create your own ImageNet dataloader
    # ........
    # ........
    # Load the model and fuse the batchnorm layers
    model = torchvision.models.efficientnet_b0(pretrained=True)
    model = fuse_fx(model.eval())
    # Insert the quantization layers
    fake_quant_model = get_insert_fake_quant_model(model,(1,3,224,224), config)
```
Then the model is ready for evaluation in the original ImageNet pipeline.

The example model checkpoints are listed in `model_checkpoints` folder. For convience,
we only listed the most important models that can be directly loaded as shown above.
Some other float32 pretrained weights are discarded in the recent pytorch version.
After the accpetance of the paper, we will provide more models with our own tools handing
different models from timm, etc.

The models can also be converted to TFlite using TinyNN features, we experienced exact same performance in TFlite 
evaluation after the conversion.