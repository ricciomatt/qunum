import torch
import numpy as np
from constants.metric import G, c_exact as c
def schwarts(g, x, metric_args, grad = False):
    if('r_s' not in metric_args):
        r_s = (2*metric_args['M']*G)/c**2
    else:
        r_s = metric_args['r_s']
    if(grad):
        g[:, 0, 0] = 1 - (r_s / x[:, 1])
        g[:, 1, 1] = 1 / (1 - (r_s / x[:, 1]))
        g[:, 2, 2] = x[:, 1] ** 2
        g[:, 3, 3] = (x[:, 1] ** 2) * torch.sin(x[:, 2]) ** 2
    else:
        g[:, 0, 0] = 1 - (r_s / x[:, 1])
        g[:, 1, 1] = 1 / (1 - (r_s / x[:, 1]))
        g[:, 2, 2] = x[:, 1] ** 2
        g[:, 3, 3] = (x[:, 1] ** 2) * np.sin(x[:, 2]) ** 2
    return g
