import copy
import math

import numpy as np
import torch


def transform_grad_to_list(grad_tensor):
    return [g.numpy().tolist() for g in grad_tensor]


def transform_list_to_grad(grad_list):
    return [torch.from_numpy(np.asarray(g)).float() for g in grad_list]


def compute_grad_update(old_model, new_model, device=None):
    # maybe later to implement on selected layers/parameters
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [
        new_param.data - old_param.data
        for old_param, new_param in zip(old_model.parameters(), new_model.parameters())
    ]


def flatten(grad_update):
    return torch.cat([update.data.view(-1) for update in grad_update])


def unflatten(flattened, normal_shape):
    grad_update = []
    for param in normal_shape:
        n_params = len(param.view(-1))
        grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size()))
        flattened = flattened[n_params:]

    return grad_update


def mask_grad_update_by_magnitude(grad_update, mask_constant):
    # mask all but the updates with larger magnitude than <mask_constant> to zero
    # print('Masking all gradient updates with magnitude smaller than ', mask_constant)
    grad_update = copy.deepcopy(grad_update)
    for i, update in enumerate(grad_update):
        grad_update[i].data[update.data.abs() < mask_constant] = 0
    return grad_update


def mask_grad_update_by_order(
    grad_update, mask_order=None, mask_percentile=None, mode="all"
):

    if mode == "all":
        # mask all but the largest <mask_order> updates (by magnitude) to zero
        all_update_mod = torch.cat(
            [update.data.view(-1).abs() for update in grad_update]
        )
        if not mask_order and mask_percentile is not None:
            mask_order = int(len(all_update_mod) * mask_percentile)

        if mask_order == 0:
            return mask_grad_update_by_magnitude(grad_update, float("inf"))
        else:
            topk, indices = torch.topk(all_update_mod, mask_order)
            return mask_grad_update_by_magnitude(grad_update, topk[-1])

    elif mode == "layer":  # layer wise largest-values criterion
        grad_update = copy.deepcopy(grad_update)

        mask_percentile = max(0, mask_percentile)
        for i, layer in enumerate(grad_update):
            layer_mod = layer.data.view(-1).abs()
            if mask_percentile is not None:
                mask_order = math.ceil(len(layer_mod) * mask_percentile)

            if mask_order == 0:
                grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
            else:
                topk, indices = torch.topk(
                    layer_mod, min(mask_order, len(layer_mod) - 1)
                )
                grad_update[i].data[layer.data.abs() < topk[-1]] = 0
        return grad_update
