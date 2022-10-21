import torch


def get_device(need_cpu=False):
    if need_cpu:
        return torch.device('cpu')

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')