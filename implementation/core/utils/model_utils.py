import torch
import torch.nn as nn
from pingpong_quant.implementation.core.components.tiny_nn_pingpong_wrapper import PingPongwrapper
from pingpong_quant.implementation.core.components.pingpong_quantizer import PingPongQuantizer
from pingpong_quant.implementation.core.components.pingpong_quantizer import prepare_for_convert, init_grad_scaling, disable_grad_scaling


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


def get_insert_fake_quant_model(model, dummy_input_shape, config, val_dataloader=None):
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
    for layer in quant_model.modules():
        if isinstance(layer, torch.ao.nn.qat.modules.linear.Linear):
            for module in layer.modules():
                if isinstance(module, PingPongQuantizer):
                    module.not_used = True
        if isinstance(layer, PingPongQuantizer):
            if layer.observer_enabled == 0 and layer.fake_quant_enabled == 0:
                layer.not_used = True
    if val_dataloader is not None:
        data = next(iter(val_dataloader))['inputs'][0, ::].unsqueeze(0).permute(0, 3, 1, 2)
    else:
        data = dummy_input
    quant_model(data)
    if config['weight_path'] is not None:
        quant_model.apply(init_grad_scaling)
        quant_model.apply(disable_grad_scaling)
        quant_model.load_state_dict(torch.load(config['weight_path'], map_location=torch.device('cpu')))
        quant_model.apply(prepare_for_convert)
    return quant_model










