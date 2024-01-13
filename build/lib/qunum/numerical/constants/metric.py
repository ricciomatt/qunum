import pandas as pd
from numpy import pi
from . import unit_conversion as uc

ub = 9.2740100783e-24
G = 6.67430e-11
c = 299_792_458
c_exact = 299_792_458
h = 6.62607015e-34
h_bar = h/(2*pi)
epsilon_0 = 8.8541878128e-12
mu_0 = 1.25663706212e-6
qe = 1.602176634e-19
me = 9.1093837015e-31
alpha = 1/137
kB = 1.380649e-23
R_inf = ((alpha**2)*me*c)/(2*h_bar)
R_K = h/qe**2
KJ = (2*qe)/h
Solar_Mass = 2e30

'''Mass Parameters Standard'''
def get_sm():
    return pd.read_csv(f'./constants/standard_model_quantum_numbers_and_properties.csv').set_index('Particle')

'''Mass Special Composite Paritcles'''
me_ev = 0.510998950e6
mp_ev = 938.272088e6
mn_ev = 939.565346e6

'''Mass Parameters'''
mp = mp_ev*uc.ev_to_joule/c_exact**2
mn = mn_ev*uc.ev_to_joule/c_exact**2
me = me_ev*uc.ev_to_joule/c_exact**2


