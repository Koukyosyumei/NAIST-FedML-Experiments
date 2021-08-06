import copy

import torch


def compute_grad_update(old_model, new_model, device=None):
    # maybe later to implement on selected layers/parameters
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [
        (new_param.data - old_param.data)
        for old_param, new_param in zip(old_model.parameters(), new_model.parameters())
    ]


def flatten(grad_update):
    return torch.cat([update.data.view(-1) for update in grad_update])


def mask_grad_update_by_magnitude(grad_update, mask_constant):
    # mask all but the updates with larger magnitude than <mask_constant> to zero
    # print('Masking all gradient updates with magnitude smaller than ', mask_constant)
    grad_update = copy.deepcopy(grad_update)
    for i, update in enumerate(grad_update):
        grad_update[i].data[update.data.abs() < mask_constant] = 0
    return grad_update
