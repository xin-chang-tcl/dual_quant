from dual_quant.implementation.core.components.dual_quantizer import DualQuantizer


def get_parameters(model, config):
    all_parameters = list(model.parameters())
    scale_parameters_id = set()
    scale_parameters = []
    ori_weights = []
    for p in model.modules():
        if isinstance(p, DualQuantizer):
            if p.scale.requires_grad:
                scale_parameters.append(p.scale)
                scale_parameters_id.add(id(p.scale))
    ####
    # Now, iterate over all parameters and separate them
    for param in all_parameters:
        if (id(param) not in scale_parameters_id) and (id(param)):
            ori_weights.append(param)
    group1 = scale_parameters
    group2 = ori_weights
    param_list = {'model_params': group2, 'scale_params': group1}

    return param_list








