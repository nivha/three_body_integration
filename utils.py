# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:13:41 2017

@author: Niv Haim (Weizmann Institute of Science)
"""

import numpy as np
from numpy.linalg import norm
from numpy import sin, cos, sqrt, cross, arctan2
import numba as nb

# Physical constants(cgs)
G = 6.67259e-8
Msun = 1.989e33
c = 2.99792458e10
au = 14959787070000
year = 3600 * 24 * 365

# Simulation constants
N = 2e11
t_unit = np.sqrt(au**3 / G / Msun)
r_dissip_cm = 4e9  # in cm
r_dissip = r_dissip_cm / au  # in scaled units

# Simulation finish reasons
REASON_NONE, FINISHED_ITERATIONS, BAD, TNAN, MAX_PERIODS, SYSTEM_BROKEN = range(6)

# Simulation regions
NEAR_PERI, NEAR_APO = range(2)

# Algebraic operations
abs2_each = lambda x: np.sum(np.square(x), axis=0)  # squared norm for each column
cnorm = lambda x: np.sqrt(abs2_each(x))  # norm of each column
normc = lambda x: x / cnorm(x)  # normalize each column by its norm


@nb.jit(nopython=True)
def cross_jit(vec1, vec2):
    """ Calculate the cross product of two 3d vectors. """
    result = np.zeros(3)
    return cross_jit_(vec1, vec2, result)


@nb.jit(nopython=True)
def cross_jit_(vec1, vec2, result):
    """ Calculate the cross product of two 3d vectors. """
    a1, a2, a3 = np.double(vec1[0]), np.double(vec1[1]), np.double(vec1[2])
    b1, b2, b3 = np.double(vec2[0]), np.double(vec2[1]), np.double(vec2[2])
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result


def get_je(e, i, Omega, omega):

    j = sqrt(1 - e**2)
    jx = j * sin(i) * sin(Omega)
    jy = -j * sin(i) * cos(Omega)
    jz = j * cos(i)
    ex = e * cos(omega) * cos(Omega) - e * sin(omega) * cos(i) * sin(Omega)
    ey = e * cos(omega) * sin(Omega) + e * sin(omega) * cos(i) * cos(Omega)
    ez = e * sin(omega) * sin(i)

    jv = np.array([jx, jy, jz])
    ev = np.array([ex, ey, ez])

    return jv, ev


def xv_flat(e, J, a, f):
    """ Returns the x, v from j, a, e in the simple case where
     j is in the direction of z - axis, e is in the direction of x - axis """
    r_f = a * (1 - e**2) / (1 + e * cos(f))
    f_dot = J / r_f**2
    r_dot = a * (1 - e**2) * f_dot * e * sin(f) / (1 + e * cos(f))**2
    x = np.array([r_f*cos(f), r_f*sin(f), 0])
    v = np.array([r_dot*cos(f)-r_f*sin(f)*f_dot,  r_dot*sin(f)+r_f*cos(f)*f_dot, 0])

    return x, v


def get_xv(ev, Jv, a, f):
    """ Gets x and v for a given Kepler orbit with ev, Jv and a """
    J = norm(Jv)
    e = norm(ev)

    en = ev / e
    Jn = Jv / J
    bn = cross(Jn, en)
    U = np.c_[en, bn, Jn]

    x_, v_ = xv_flat(e, J, a, f)
    x = U.dot(x_)
    v = U.dot(v_)

    return x, v


def get_energy(x1, v1, m1, x2, v2, m2, x3, v3, m3, G):
    r21 = x1 - x2
    r13 = x3 - x1
    r23 = x3 - x2
    K = 0.5 * (m1 * v1 @ v1 + m2 * v2 @ v2 + m3 * v3 @ v3)
    U = - G * (m1 * m2 / norm(r21) + m1 * m3 / norm(r13) + m2 * m3 / norm(r23))
    return U, K


@nb.jit(nopython=True)
def E_from_M(M, e, tol=1e-6):
    """ Get eccentric anomaly from mean anomaly
        Solve Kepler equation:
                M = E - e * sin(E)
        iteratively using fixed point method """
    Estart = M
    E = M + e * sin(Estart)
    i = 0
    while np.abs(E-Estart) > tol:
        Estart = E
        E = M + e * sin(E)
        i += 1
    print('Kepler equation solver num iterations:', i)
    return E


def f_from_M(M, e, tol=1e-6):
    """ Get true anomaly from mean anomaly """
    E = E_from_M(M, e, tol)
    f = 2 * arctan2(sqrt(1+e)*sin(E/2), sqrt(1-e)*cos(E/2))
    return f


def get_dt0(G, m1, m2, a_in0, samples_per_Pcirc, dt00):
    Min = m1 + m2
    Pcirc = 2 * np.pi * np.sqrt(a_in0 ** 3 / G / Min)
    if not np.isnan(dt00):
        dt0 = dt00 * np.sqrt(a_in0**3 / G / Min)
    else:
        dt0 = Pcirc / samples_per_Pcirc
    return dt0


def get_jz_eff(G, m1, m2, m3, a, e, inclination, rper_over_a, eper):
    Min = m1 + m2
    Mout = m1 + m2 + m3
    mu_in = (m1 * m2) / Min
    mu_out = (Min * m3) / Mout
    a_out = a * rper_over_a / (1 - eper)
    Jcirc = mu_in * np.sqrt(G * Min * a)
    Jout = mu_out * np.sqrt(G * Mout * a_out * (1 - eper**2))
    j = np.sqrt(1 - e**2)
    jz_eff = j * np.cos(inclination) + j**2 * Jcirc / 2 / Jout

    return jz_eff


class OrbitalParameters:
    """Container for orbital parameters
        that are necessary for initialization and post processing
        but not necessarily needed for integration time
    """
    def __init__(self, **kwargs):
        self.set_orbital_parameters(**kwargs)

    def set_orbital_parameters(s, G, m1, m2, m3, e, a, M0_in, rper_over_a, eper, f_out, inclination, Omega, omega):
        # basic params
        s.aper = rper_over_a / (1 - eper)
        s.Min = m1 + m2
        s.Mout = m1 + m2 + m3
        s.muin = m1 * m2 / s.Min
        s.muout = s.Min * m3 / s.Mout
        s.f_in = f_from_M(M0_in, e)
        print('f_in:', s.f_in)

        # initialize state for outer orbit
        s.jv_out, s.ev_out = get_je(eper, 0, 0, 0)
        s.Jv_out = s.jv_out * sqrt(G * s.Mout * s.aper)
        s.x_out0, s.v_out0 = get_xv(s.ev_out, s.Jv_out, s.aper, f_out)
        s.x0_cms12 = -(m3 / s.Mout) * s.x_out0
        s.v0_cms12 = -(m3 / s.Mout) * s.v_out0
        s.x03 = (s.Min / s.Mout) * s.x_out0
        s.v03 = (s.Min / s.Mout) * s.v_out0

        # initialize state for inner orbit
        s.jv0, s.ev0 = get_je(e, inclination, Omega, omega)
        s.Jv = s.jv0 * sqrt(G * s.Min * a)
        s.x0_in, s.v0_in = get_xv(s.ev0, s.Jv, a, s.f_in)
        s.x01 = -(m2 / s.Min) * s.x0_in + s.x0_cms12
        s.v01 = -(m2 / s.Min) * s.v0_in + s.v0_cms12
        s.x02 = (m1 / s.Min) * s.x0_in + s.x0_cms12
        s.v02 = (m1 / s.Min) * s.v0_in + s.v0_cms12

        # make state vectors
        s.x0 = np.concatenate((s.x01, s.x02, s.x03))
        s.v0 = np.concatenate((s.v01, s.v02, s.v03))

        # Inner orbit parameters
        s.U_in0 = - G * s.Min * s.muin / norm(s.x0_in)
        s.K_in0 = 0.5 * s.muin * norm(s.v0_in)**2
        s.E_in0 = s.K_in0 + s.U_in0
        s.a_in0 = - G * s.Min * s.muin / 2 / s.E_in0
        s.P_in = 2 * np.pi * sqrt(s.a_in0**3 / G / s.Min)

        # Outer orbit parameters
        s.E_out0 = 0.5 * s.muout * norm(s.v_out0)**2 - G * s.Mout * s.muout / norm(s.x_out0)
        s.a_out0 = - G * s.muout * s.Mout / 2 / s.E_out0
        s.P_out = 2 * np.pi * sqrt(s.a_out0**3 / G / s.Mout)

        # Energy
        s.U0, s.K0 = get_energy(s.x01, s.v01, m1, s.x02, s.v02, m2, s.x03, s.v03, m3, G)
        s.E0 = s.U0 + s.K0

        # jz_eff (not per unit mass)
        s.jz_eff = get_jz_eff(G, m1, m2, m3, a, e, inclination, rper_over_a, eper)

        # Kozai time
        s.tau = 2 * (sqrt(G * s.Min) / (G * m3)) * (s.aper ** 3 / a**(3 / 2)) * (1 - eper**2)**(3 / 2)


def inspect_config_file(path):
    with open(path, 'rb') as f:
        d = np.load(f).item()
    params_phys, params_sim = d['params_phys'], d['params_sim']

    locals().update(params_phys)
    locals().update(params_sim)


