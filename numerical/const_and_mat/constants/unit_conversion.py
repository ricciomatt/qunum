import numpy as np

joule_to_ev = 6.242e+18
ev_to_joule = 1/joule_to_ev


def to_kelvin(input, unit = 'C'):
    if(unit == 'C'):
        return input+273.15
    else:
        return (input -32)*(5/9) + 273.15

