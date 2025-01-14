import torch
import torch.nn as nn
import tinynn
import timm
from dual_quant.implementation.core.components.tiny_nn_dual_wrapper import Dualwrapper
from dual_quant.implementation.core.components.dual_quantizer import DualQuantizer
from dual_quant.implementation.core.components.dual_quantizer import prepare_for_convert, init_grad_scaling, \
    disable_grad_scaling


def change_clip_range(quant_model):
    name_list = []
    for module in quant_model._modules:
        name_list.append(module)

    for idx, name in enumerate(name_list):
        if isinstance(quant_model._modules[name_list[idx]], torch.nn.Hardswish):
            for m in quant_model._modules[name_list[idx - 1]].modules():
                if isinstance(m, DualQuantizer):
                    if m.quant_min + m.quant_max != 0:
                        m.clip_min = -3
    for idx, name in enumerate(name_list):
        if isinstance(quant_model._modules[name_list[idx]], torch.nn.Hardsigmoid):
            for m in quant_model._modules[name_list[idx - 1]].modules():
                if isinstance(m, DualQuantizer):
                    if m.quant_min + m.quant_max != 0:
                        m.clip_min = -3

    for idx, name in enumerate(name_list):
        if isinstance(quant_model._modules[name_list[idx]], tinynn.graph.quantization.modules.QSiLU):
            current_module = quant_model._modules[name_list[idx]]
            name_1 = name_list[idx - 1]
            previous_module = quant_model._modules[name_1]

            if isinstance(previous_module, torch.nn.Conv2d):
                for m in previous_module.modules():
                    if isinstance(m, DualQuantizer):
                        if m.quant_min + m.quant_max != 0:
                            m.clip_min = -6
    return quant_model


def replace_inplace(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.SiLU):
            setattr(model, child_name, nn.SiLU(inplace=False))
        elif isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU(inplace=False))
        elif isinstance(child, nn.ReLU6):
            setattr(model, child_name, nn.ReLU6(inplace=False))
        elif isinstance(child, nn.Hardswish):
            setattr(model, child_name, nn.Hardswish(inplace=False))
        elif isinstance(child, nn.modules.linear.NonDynamicallyQuantizableLinear):
            setattr(model, child_name,
                    nn.modules.linear.Linear(child.in_features, child.out_features, child.bias is not None))
        else:
            replace_inplace(child)


def get_insert_fake_quant_model(model, dummy_input, config, only_min_max=False, activation_only=False, output_dir='quant_out', enable_act=False, weights_only=False, ignore_layer_names=[], only_quantize_layers=[],
                                not_quantize_layers=[], special_layers=[], only_min_max_layers=[]):
    replace_inplace(model)
    path = output_dir
    quantizer = Dualwrapper(
        model,
        dummy_input,
        work_dir=path,
        config={
            'asymmetric': True,
            'backend': 'qnnpack',
            "disable_requantization_for_cat": False,
            'set_quantizable_op_stats': True,
            'fuse_bn': True,
            'per_tensor': not config['per_channel'],
            'lowest_scale': config['lowest_scale'],
            'threshold': config['threshold'],
            'penalty_factor': config['penalty_factor'],
        }
    )

    quant_model = quantizer.quantize()
    quant_model.eval()
    # for layer in quant_model.modules():
    #     if isinstance(layer, torch.nn.ReLU6):
    #         for module in layer.modules():
    #             if isinstance(module, DualQuantizer):
    #                 module.observer_enabled[0] = 1
    #                 module.fake_quant_enabled[0] = 1
    for layer in quant_model.modules():
        if isinstance(layer, DualQuantizer):
            if layer.observer_enabled == 0 or layer.fake_quant_enabled == 0:
                layer.float = True
                layer.not_used = True
    #################min-max

    if only_min_max:
        for module in quant_model.modules():
            if isinstance(module, DualQuantizer):
                module.use_minmax = True
                module.not_used = True

    if only_min_max_layers != []:
        for name in only_min_max_layers:
            for module in quant_model._modules[name].modules():
                if isinstance(module, DualQuantizer):
                    module.use_minmax = True
                    module.not_used = True
    if ignore_layer_names != []:
        for name in ignore_layer_names:
            for module in quant_model._modules[name].modules():
                if isinstance(module, DualQuantizer):
                    if module.quant_min + module.quant_max != 0:
                        module.use_minmax = True

    # for layer in quant_model.modules():
    #     if isinstance(layer, torch.ao.nn.qat.modules.linear.Linear):
    #         for module in layer.modules():
    #             if isinstance(module, DualQuantizer):
    #                 if module.quant_min + module.quant_max != 0:
    #                     module.not_used = True

    ################################

    ####
    if only_quantize_layers != []:
        for name, module in quant_model.named_modules():
            if isinstance(module, torch.ao.quantization.FakeQuantizeBase):
                if name.split('.')[0] in only_quantize_layers:
                    module.apply(torch.ao.quantization.enable_fake_quant)
                    module.apply(torch.ao.quantization.enable_observer)
                else:
                    module.apply(torch.ao.quantization.disable_fake_quant)
                    module.apply(torch.ao.quantization.disable_observer)
                    if isinstance(module, DualQuantizer):
                        module.not_used = True
                        module.record = False
                        module.to_compute = False

    if not_quantize_layers != []:
        for name in not_quantize_layers:
            for module in quant_model._modules[name].modules():
                if isinstance(module, DualQuantizer):
                    module.apply(torch.ao.quantization.disable_fake_quant)
                    module.apply(torch.ao.quantization.disable_observer)
                    module.not_used = True
                    module.record = False

    ########################################################
    if special_layers != []:
        for idx, name in enumerate(special_layers):
            for module in quant_model._modules[special_layers[idx]].modules():
                if isinstance(module, DualQuantizer):
                    if module.quant_min + module.quant_max != 0:
                        module.enhance = True

    quant_model = change_clip_range(quant_model)
    # ------------------------------------range modification




    if activation_only:
        for module in quant_model.modules():
            if isinstance(module, DualQuantizer):
                if module.quant_min + module.quant_max == 0:
                    module.apply(torch.ao.quantization.disable_fake_quant)
                    module.float = True
                    module.record = False
    if weights_only:
        for module in quant_model.modules():
            if isinstance(module, DualQuantizer):
                if module.quant_min + module.quant_max != 0:
                    module.apply(torch.ao.quantization.disable_fake_quant)
                    module.float = True
                    module.record = False

    return quant_model










