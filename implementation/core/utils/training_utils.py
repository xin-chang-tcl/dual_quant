from dual_quant.implementation.core.components.dual_quantizer import DualQuantizer
from dual_quant.implementation.core.components.hooks import set_trainable_fn, register_trainble_hook, set_trainable_fn_act
import torch
from dual_quant.implementation.core.components.dual_quantizer import init_grad_scaling, prepare_for_convert
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dual_quant.implementation.core.components.loss import compute_dual_loss
from dual_quant.implementation.core.components.hooks import register_hooks_dual_loss, dual_loss_hook_fn, \
    register_scale_hook, scale_hook_fn

def get_all_layer_names_and_assign_layers_to_quantize(quantize_model, input_shape, sequential_model, til_layer):
    sample = torch.randn(input_shape)
    if torch.cuda.is_available():
        sample = sample.to('cuda')
    paramters = get_trainable_quantization_parameters(quantize_model, sample)
    total_idx_list = []
    total_names = []
    if sequential_model:
        every = 1
    else:
        every = len(paramters)
    scale_hook = register_scale_hook(total_idx_list, total_names, quantize_model, scale_hook_fn, til_layer, every, False)

    quantize_model(sample)
    for item in scale_hook:
        item.remove()
    total_idx_list = []
    params = [p for p in paramters if p.requires_grad == True]
    return params, total_names

def get_trainable_quantization_parameters(model, dummy_input, act_only=False):
    total_trainable_params = []
    total_ids = []
    if act_only:
        hook = register_trainble_hook(total_trainable_params, model, set_trainable_fn_act, total_ids)
    else:
        hook = register_trainble_hook(total_trainable_params, model, set_trainable_fn, total_ids)
    model(dummy_input)
    for h in hook:
        h.remove()
    return total_trainable_params

def get_parameters_online(model):
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

def make_ptq_training(model, dataloader, input_shape, initial_iteration, qsin_num_iteration, sequential_mode=False, lr=0.01):
    # Do a few iterations for standard min-max calibration
    for idx, data in enumerate(dataloader):
        if idx >= initial_iteration:
            break
        if torch.cuda.is_available():
            data = data.to('cuda')
        model(data)


    model.apply(init_grad_scaling)
    if torch.cuda.is_available():
        model = model.cuda()
    if not sequential_mode:
        model.apply(torch.ao.quantization.disable_observer)
        model.train()
        params, total_names = get_all_layer_names_and_assign_layers_to_quantize(model, input_shape, sequential_mode, 0)
        optimizer = optim.Adam(params, lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=qsin_num_iteration, eta_min=0.001)
        for idxx, data in enumerate(dataloader):
            if torch.cuda.is_available():
                data = data.to('cuda')
            if idxx >= qsin_num_iteration:
                break
            optimizer.zero_grad()
            scales = []
            zero_point = []
            qsin_loss = []
            qsin_hooks = register_hooks_dual_loss(model, dual_loss_hook_fn, qsin_loss, scales, zero_point)
            _ = model(data)
            loss = compute_dual_loss(qsin_loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            for hook in qsin_hooks:
                hook.remove()
            scales = []
            zero_point = []
    model.apply(prepare_for_convert)
    torch.save(model.state_dict(), 'qsin_ptq.pth')






