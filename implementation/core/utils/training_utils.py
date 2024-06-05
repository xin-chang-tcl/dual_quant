from dual_quant.implementation.core.components.dual_quantizer import DualQuantizer


def get_parameters(model, config):
    all_parameters = list(model.parameters())
    scale_parameters_id = set()
    scale_parameters = []
    ori_weights = []
    for p in model.modules():
        if isinstance(p, PingPongQuantizer):
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
    if config['training_type'] == 'qat':
        param_list = group1+group2
    else:
        ####For PTQ only, we train only the scale parameters, however its possible to train the whole model as shown above.
        param_list = group1
        for p in model.parameters():
            if id(p) in scale_parameters_id:
                p.requires_grad = True
            else:
                p.requires_grad = False
    return param_list








