import torch
import torch.nn as nn
from dual_quant.implementation.core.components.tiny_nn_dual_wrapper import PingPongwrapper
from dual_quant.implementation.core.components.dual_quantizer import DualQuantizer
from dual_quant.implementation.core.components.dual_quantizer import prepare_for_convert, init_grad_scaling, disable_grad_scaling


def replace_inplace(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.SiLU):
            setattr(model, child_name, nn.SiLU(inplace=False))
        elif isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU(inplace=False))
        elif isinstance(child, nn.ReLU6):
            setattr(model, child_name, nn.ReLU(inplace=False))
        elif isinstance(child, nn.Hardswish):
            setattr(model, child_name, nn.Hardswish(inplace=False))
        elif isinstance(child, nn.modules.linear.NonDynamicallyQuantizableLinear):
            setattr(model, child_name, nn.modules.linear.Linear(child.in_features, child.out_features, child.bias is not None))
        else:
            replace_inplace(child)


def get_insert_fake_quant_model(model, dummy_input_shape, config, ignore_layer_names=[]):
    replace_inplace(model)
    path = 'quant_out'
    dummy_input = torch.randn(dummy_input_shape)
    quantizer = PingPongwrapper(
        model,
        dummy_input,
        work_dir=path,
        config={
            'asymmetric': True,
            'backend': 'qnnpack',
            "disable_requantization_for_cat": True,
            'per_tensor': not config['per_channel'],
            'lowest_scale': config['lowest_scale'],
        }
    )

    quant_model = quantizer.quantize()
    quant_model.eval()
    if ignore_layer_names != []:
        for name in ignore_layer_names:
            for module in quant_model._modules[name].modules():
                if isinstance(module, DualQuantizer):
                    if module.quant_min + module.quant_max != 0:
                        module.not_used = True


        #We do not make training of scales for Linear layers
        # if isinstance(layer, torch.ao.nn.qat.modules.linear.Linear):
        #     for module in layer.modules():
        #         if isinstance(module, DualQuantizer):
        #             module.not_used = True
        # if isinstance(layer, DualQuantizer):
        #     if layer.observer_enabled == 0 and layer.fake_quant_enabled == 0:
        #         layer.not_used = True

    return quant_model










