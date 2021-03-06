# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:13:41 2017

@author: Niv Haim (Weizmann Institute of Science)
"""

import numpy as np
import numba as nb
from numpy.linalg import norm

from sim.utils import REASON_NONE, OrbitalParameters, get_dt0


spec = [
    # main arrays
    ('X', nb.double[:, :]),
    ('V', nb.double[:, :]),
    ('DT', nb.double[:]),
    ('T', nb.double[:]),
    # last steps arrays
    ('Xlast', nb.double[:, :]),
    ('Vlast', nb.double[:, :]),
    ('DTlast', nb.double[:]),
    ('Tlast', nb.double[:]),
    # closest approaches arrays
    ('Xca', nb.double[:, :]),
    ('Vca', nb.double[:, :]),
    ('Tca', nb.double[:]),
    ('Ica', nb.int64[:]),
    ('Jzeffca', nb.double[:]),
    # specific points arrays
    ('dE_max_x', nb.double[:]),
    ('dE_max_v', nb.double[:]),
    ('jz_eff_min_x', nb.double[:]),
    ('jz_eff_min_v', nb.double[:]),
    # indexes / flags
    ('i', nb.int64),
    ('idx', nb.int64),
    ('caidx', nb.int64),
    ('dE_max_i', nb.int64),
    ('nP', nb.int64),
    ('save_every_P_i', nb.int64),
    # parameters discovered during run
    ('steps_per_P', nb.int64),
    ('fin_reason', nb.int64),
    ('closest_approach_r', nb.double),
    ('dE_max', nb.double),
    ('jz_eff', nb.double),
    ('jz_eff_crossings', nb.int64),
    ('jz_eff_n', nb.int64),
    ('jz_eff_min', nb.double),
    ('jz_eff_mean', nb.double),
    ('jz_eff_M2', nb.double),

    # configuration variables
    # physical
    ('G', nb.double),
    ('m1', nb.double),
    ('m2', nb.double),
    ('m3', nb.double),
    ('a', nb.double),
    ('e', nb.double),
    ('M0_in', nb.double),
    ('M0_out', nb.double),
    ('inclination', nb.double),
    ('Omega', nb.double),
    ('omega', nb.double),
    ('rper_over_a', nb.double),
    ('eper', nb.double),
    # simulation
    ('dt00', nb.double),
    ('max_periods', nb.int64),
    ('save_every', nb.int64),
    ('dump_every', nb.int64),
    ('save_every_P', nb.int64),
    ('samples_per_Pcirc', nb.int64),
    ('save_last', nb.int64),
    ('rmax', nb.double),
    ('ca_saveall', nb.double),

    # computed from configuration variables
    ('f_in', nb.double),
    ('f_out', nb.double),
    ('E0', nb.double),
    ('U_init', nb.double),
    ('P_in', nb.double),
    ('P_out', nb.double),
    ('jz_eff0', nb.double),
    ('dt0', nb.double),
]


@nb.jitclass(spec)
class SimState(object):
    def __init__(self, vsize, save_last):
        # define all arrays
        self.X = np.empty((9, vsize), dtype=np.double)
        self.V = np.empty((9, vsize), dtype=np.double)
        self.DT = np.empty(vsize, dtype=np.double)
        self.T = np.empty(vsize, dtype=np.double)
        self.Xlast = np.empty((9, save_last), dtype=np.double)
        self.Vlast = np.empty((9, save_last), dtype=np.double)
        self.DTlast = np.empty(save_last, dtype=np.double)
        self.Tlast = np.empty(save_last, dtype=np.double)
        self.Xca = np.empty((9, 100000), dtype=np.double)
        self.Vca = np.empty((9, 100000), dtype=np.double)
        self.Tca = np.empty(100000, dtype=np.double)
        self.Ica = np.empty(100000, dtype=np.int64)
        self.Jzeffca = np.empty(100000, dtype=np.double)
        self.dE_max_x = np.zeros(9, dtype=np.double)
        self.dE_max_v = np.zeros(9, dtype=np.double)
        self.jz_eff_min_x = np.zeros(9, dtype=np.double)
        self.jz_eff_min_v = np.zeros(9, dtype=np.double)


def inject_config_params(s, G, m1, m2, m3, a, e, M0_in,
                         M0_out, inclination, Omega, omega, rper_over_a, eper,
                         dt00, max_periods, dump_every, save_every, save_every_P,
                         samples_per_Pcirc, save_last, rmax, ca_saveall):
    # dump args to properties
    s.G = G
    s.m1 = m1
    s.m2 = m2
    s.m3 = m3
    s.a = a
    s.e = e
    s.M0_in = M0_in
    s.M0_out = M0_out
    s.inclination = inclination
    s.Omega = Omega
    s.omega = omega
    s.rper_over_a = rper_over_a
    s.eper = eper
    s.dt00 = dt00
    s.max_periods = max_periods
    s.dump_every = dump_every
    s.save_every = save_every
    s.save_every_P = save_every_P
    s.samples_per_Pcirc = samples_per_Pcirc
    s.save_last = save_last
    s.rmax = rmax
    s.ca_saveall = ca_saveall


def initialize_state(s):
    """should be called only the first time SimState is created
        s: SimState object
    """
    # compute orbital parameters from config params
    op = OrbitalParameters(G=s.G, m1=s.m1, m2=s.m2, m3=s.m3, e=s.e, a=s.a, M0_in=s.M0_in,
                           rper_over_a=s.rper_over_a, eper=s.eper, M0_out=s.M0_out,
                           inclination=s.inclination, Omega=s.Omega, omega=s.omega)
    # set orbital params
    s.f_in = op.f_in
    s.f_out = op.f_out
    s.E0 = op.E0
    s.jz_eff0 = op.jz_eff
    s.P_in = op.P_in
    s.P_out = op.P_out

    # set dt0 and U_init
    s.dt0 = get_dt0(s.G, s.m1, s.m2, op.a_in0, s.samples_per_Pcirc, s.dt00)
    s.U_init = - s.G * s.m1 * s.m2 / norm(op.a_in0)

    # initialize integration flags
    s.fin_reason = REASON_NONE
    s.closest_approach_r = np.infty
    s.save_every_P_i = 1
    s.i = s.idx = s.dE_max_i = 0
    s.nP = s.steps_per_P = s.dE_max = 0

    # jz_eff stuff
    s.jz_eff = op.jz_eff
    s.jz_eff_crossings = 0
    s.jz_eff_min = np.infty
    s.jz_eff_mean = op.jz_eff
    s.jz_eff_n = 1
    s.jz_eff_M2 = 0

    # set initial simulation params
    s.Xlast[:, 0] = op.x0
    s.Vlast[:, 0] = op.v0
    s.DTlast[0] = 0
    s.Tlast[0] = 0
    # set first ca values
    s.Xca[:, 0] = op.x0
    s.Vca[:, 0] = op.v0
    s.Tca[0] = 0
    s.Ica[0] = 0
    s.Jzeffca[0] = op.jz_eff
    s.caidx = 1


def make_copy(s):
    """ Returns a copy of sim_state s (a new SimState instance with all attributes copied) """
    vsize = max(s.idx, s.caidx)
    s_copy = SimState(vsize, s.save_last)
    for var_name, var_type in spec:
        # if var is an array make a copy
        if isinstance(var_type, nb.types.npytypes.Array):
            setattr(s_copy, var_name, getattr(s, var_name).copy())
        else:
            setattr(s_copy, var_name, getattr(s, var_name))
    return s_copy


def chop_arrays(s):
    """ Chops the irrelevant end of the storage arrays using the relevant end-index """
    s.X = s.X[:, :s.idx]
    s.V = s.V[:, :s.idx]
    s.DT = s.DT[:s.idx]
    s.T = s.T[:s.idx]
    s.Ica = s.Ica[:s.caidx]
    s.Xca = s.Xca[:, :s.caidx]
    s.Vca = s.Vca[:, :s.caidx]
    s.Tca = s.Tca[:s.caidx]
    s.Jzeffca = s.Jzeffca[:s.caidx]


def get_state_dict(s):
    """Dumps all sim state variables into a file"""

    chop_arrays(s)

    # # fix (shift) lasts arrays
    # s.Xlast = np.roll(s.Xlast, - s.i % s.save_last, axis=1)
    # s.Vlast = np.roll(s.Vlast, - s.i % s.save_last, axis=1)
    # s.Tlast = np.roll(s.Tlast, - s.i % s.save_last)

    # dump state variables into dictionary
    state_d = {}
    for name, _ in spec:
        state_d[name] = getattr(s, name)

    return state_d
