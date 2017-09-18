# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:13:41 2017

@author: Niv Haim (Weizmann Institute of Science)
"""

import numpy as np
from numpy.linalg import norm
from sim.utils import abs2_each, cnorm, normc
from sim.utils import OrbitalParameters
from sim.sim_state import get_state_dict


def rescale(op, scaling_params):
    # rescale variables
    op.Msun = scaling_params['Msun']
    op.au = scaling_params['au']
    op.t_unit = scaling_params['t_unit']
    op.G = scaling_params['G']

    op.X = op.X * op.au
    op.V = op.V * op.au / op.t_unit
    op.DT = op.DT * op.t_unit
    op.dt0 = op.dt0 * op.t_unit
    op.m1 = op.m1 * op.Msun
    op.m2 = op.m2 * op.Msun
    op.m3 = op.m3 * op.Msun
    op.muin = op.muin * op.Msun
    op.Min = op.Min * op.Msun
    op.aper = op.aper * op.au
    # s.eper = s.eper
    # op.ev0 = op.ev0
    # op.jv0 = op.jv0
    op.Xca = op.Xca * op.au
    op.Vca = op.Vca * op.au / op.t_unit


def _post_process(s, scaling_params):
    op = OrbitalParameters(G=s.G, m1=s.m1, m2=s.m2, m3=s.m3, e=s.e, a=s.a, M0_in=s.M0_in,
                           rper_over_a=s.rper_over_a, eper=s.eper, M0_out=s.M0_out,
                           inclination=s.inclination, Omega=s.Omega, omega=s.omega)

    state_d = get_state_dict(s)
    for k, v in state_d.items():
        setattr(op, k, v)

    rescale(op, scaling_params)

    # total energy constant (brute force)
    op.v12 = abs2_each(op.V[0:3, :])
    op.v22 = abs2_each(op.V[3:6, :])
    op.v32 = abs2_each(op.V[6:9, :])
    op.x1 = op.X[0:3, :]
    op.x2 = op.X[3:6, :]
    op.x3 = op.X[6:9, :]
    op.r21 = op.x2 - op.x1
    op.r13 = op.x3 - op.x1
    op.r23 = op.x3 - op.x2
    op.K = 0.5 * (op.m1 * op.v12 + s.m2 * op.v22 + op.m3 * op.v32)
    op.U = -op.G * (op.m1 * s.m2 / cnorm(op.r21) + op.m1 * op.m3 / cnorm(op.r13) + op.m2 * op.m3 / cnorm(op.r23))
    op.E = op.K + op.U

    # inner orbit
    op.r_in = op.r21
    op.v_in = op.V[3:6, :] - op.V[0:3, :]

    op.r2 = abs2_each(op.r_in)
    op.v2 = abs2_each(op.v_in)
    op.U_in = - op.G * op.muin * op.Min / np.sqrt(op.r2)
    op.K_in = 0.5 * op.muin * op.v2
    op.E_in = op.K_in + op.U_in
    op.a_in = - op.G * op.muin * op.Min / 2 / op.E_in
    op.Jin = np.cross(op.r_in.T, op.v_in.T).T
    op.Jin2 = abs2_each(op.Jin)
    op.eccv = (1 / op.G / op.Min) * np.cross(op.v_in.T, op.Jin.T).T - normc(op.r_in)
    op.ecc2 = abs2_each(op.eccv)
    op.ecc = np.sqrt(op.ecc2)
    op.j = op.Jin / np.sqrt(op.G * op.Min * op.a_in)
    op.j2 = abs2_each(op.j)

    # define a,e,j for theoretical computations
    op.aa = op.a_in
    op.ev = op.eccv[0, :]
    op.jv = op.j[0, :]

    op.ev2 = op.ev @ op.ev
    op.evn = op.ev / norm(op.ev)
    op.jv2 = op.jv.dot(op.jv)
    op.jvn = op.jv / norm(op.jv)

    # # da average potential
    # op.phi_da_c = (3 / 4) * (op.G * op.m3 * op.aa ** 2) / (op.aper ** 3 * (1 - op.eper ** 2) ** (3 / 2))
    # op.phi_da = op.phi_da_c * (1 / 6. - op.ecc2 + (5 / 2) * op.eccv[2, :] ** 2 - (1 / 2) * op.jv[2] ** 2)
    #
    # # closest approach parameters
    # op.r_in_closest = op.Xca[3:6, :] - op.Xca[0:3, :]
    # op.v_in_closest = op.Vca[3:6, :] - op.Vca[0:3, :]
    # op.Jin_closest = np.cross(op.r_in_closest.T, op.v_in_closest.T).T
    # op.ecc_closest = norm((1 / op.G / op.Min) * np.cross(op.v_in_closest.T, op.Jin_closest.T).T - normc(op.r_in_closest))

    # # assert that final computations agree on the initial values
    # print("Post process assertions")
    # print('ev', np.c_[op.ev0, op.eccv[0:3, 0]])
    # print('jv', np.c_[op.jv0, op.j[0:3, 0]])

    return op
