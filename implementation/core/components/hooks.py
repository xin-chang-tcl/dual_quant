from pingpong_quant.implementation.core.components.pingpong_quantizer import PingPongQuantizer


def dual_loss_hook_fn(loss_values, module, input):
    if module.init_scales:
        clip_value, round_value = module.compute_dual_loss(input[0])
        if module.is_per_channel:
            loss_values.append([clip_value, round_value,0])
        else:
            loss_values.append([clip_value, round_value, 1])


def register_hooks_dual_loss(model, hook_fn, loss_values):
    hooks = []
    for layer1 in model.modules():
        if isinstance(layer1, PingPongQuantizer):
            def dual_loss_hook_fn(module, input, output, hook_fn=hook_fn, loss_values=loss_values):
                hook_fn(loss_values, module, input)

            hooks.append(layer1.register_forward_hook(dual_loss_hook_fn))
        else:
            continue
    return hooks
