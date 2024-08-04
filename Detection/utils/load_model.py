import torch
from collections import OrderedDict


def load_model(model, state_dict, load_to_cpu=False):
    new_state_dict = OrderedDict()
    if not isinstance(state_dict, OrderedDict):
        if load_to_cpu:
            state_dict = torch.load(state_dict, weights_only=True, map_location='cpu')
        else:
            state_dict = torch.load(state_dict, weights_only=True)
        
    for k in state_dict.keys():
        if k.startswith('module.'):
            new_k = k[7:]
            new_state_dict[new_k] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]
    
    model.load_state_dict(new_state_dict)
    return model