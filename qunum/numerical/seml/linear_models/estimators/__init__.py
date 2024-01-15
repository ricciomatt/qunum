from .gls import reg as gls_reg
from .ridge import ridge_reg
from .quantile import QuantileLoss
from .lasso import LassoLoss

mapping = {'Lasso':LassoLoss,'Quantile':QuantileLoss, 'GLS':gls_reg, 'Ridge':ridge_reg}