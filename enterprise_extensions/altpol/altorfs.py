import numpy as np
import scipy.special as ss
euler_const = 0.5772156649
kpc_to_meter = 3.086e19
light_speed = 299792458


def HD_ORF (angle):
    return 3/2*( (1/3 + ((1-np.cos(angle))/2) * (np.log((1-np.cos(angle))/2) - 1/6)))

def ST_ORF (angle):
    return 1/8 * (3 + np.cos(angle))

def autovl(f, l):
    return (6*np.log(4*np.pi*f*l) - 14 + 6 * euler_const)

def critical_loc(ga,gb, tol = .95):
    gdiff = ga - gb
    if np.all(gdiff == 0.0) :
        c = 2 * (-7 + 3*(np.log(2*ga) + 3*euler_const)) * tol
    else:
        c = (-7 + 3*(np.log(2 * ga * gb/(gdiff)) + euler_const + ss.sici(2 * gdiff)[1])) * tol
    return np.arccos(np.real(1/4*(4 + 3 * ss.lambertw(-8/3 * np.exp(-7/3 - np.float64(c)/3), k = 0))))

def VL_ORF(ang_with_info, f_norm):

    xi = ang_with_info[:,2].astype(dtype = float)
    cos = np.cos(xi)
    orf = 3*np.log(2/(1-np.cos(xi))) -4*np.cos(xi) - 3

    g1 = 2*np.pi * ang_with_info[:,0].astype(dtype = float) * f_norm
    g2 = 2*np.pi * ang_with_info[:,1].astype(dtype = float) * f_norm
    c_loc = critical_loc(g1,g2)
    mask = np.logical_not(xi > c_loc)
    g1 = g1[mask]
    g2 = g2[mask]
    gdiff = g1-g2
    cos = cos[mask]

    if np.all(gdiff == 0):
        orf[mask] = 2 * (-4*cos - 3 + 3*(np.log(2 * g1) + euler_const + ss.sici(2 * g1)[1])*cos)
    else:
        orf[mask] = (-4*cos - 3 + 3*(np.log(2 * g1 * g2/(gdiff)) + euler_const + ss.sici(2 * gdiff)[1])*cos)

    return orf
