# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:13:41 2017

@author: Niv Haim (Weizmann Institute of Science)

@details: An example for a simple run of the simulation code

Physical Parameters:

        G: gravity constant
        m1: mass of body no. 1
        m2: mass of body no. 2
        m3: mass of body no. 3
        a: inner semi major axis
        e: inner eccentricity
        M0_in: inner mean anomaly
        M0_out: outer mean anomaly
        inclination: mutual inclination
        Omega: longitude of ascending node
        omega: argument of periapsis
        rper_over_a: hierarchy (outer pericenter over inner semi major axis)
        eper: outer eccentricity

Simulation Parameters:

        samples_per_Pcirc: number of sampling points per circular inner orbit
        max_periods: maximal periods to stop (-1 for don't care)
        dump_every: how many iterations between two state dump to file (heavy operation, better be a big number)
        save_every: how many iterations between two state saves (if too small, result file might be very big)
        save_every_P: how many periods between two state saves (instead of save_every; 0 for don't care)
        rmax: the simulation stops if one of the bodies is unbound to the other two, and its distance is larger
              than rmax times the distance between the other two.
        save_last: how many last iterations to save
        ca_saveall: boolean. save all close approaches or only the smallest one until each time.
        dt00: different method to initialize the time-step (np.nan for don't care)

"""


# append code directory to sys_path, so that imports would work
import os
import sys
CODE_DIR = os.path.join(os.sep, 'home', 'path', 'to', 'code', 'dir')
sys.path.append(CODE_DIR)


import numpy as np
import time
from sim.run_simulation import run_simulation

params_phys = {
    'G': 1.0,               # Gravity constant
    'm1': 0.5,              # mass of body no. 1
    'm2': 0.5,              # mass of body no. 2
    'm3': 0.5,              # mass of body no. 3
    'a': np.double(1),      # inner semi major axis
    'e': 0.1,               # inner eccentricity
    'M0_in': 2.649,         # inner mean anomaly
    'M0_out': 3.89,         # outer mean anomaly
    'inclination': 1.489,   # mutual inclination
    'Omega': 1.34,          # longitude of ascending node
    'omega': 0.932,         # argument of periapsis
    'rper_over_a': 5.0,     # hierarchy (outer pericenter over inner semi major axis)
    'eper': 0.425,          # outer eccentricity
}

params_sim = {
    'samples_per_Pcirc': np.int64(500),
    'max_periods': np.int64(1000),       # -1 for don't care
    'dump_every': np.int64(10000000),
    'save_every': np.int64(20),
    'save_every_P': np.int64(0),         # 0 for don't care
    'rmax': 50 * params_phys['a'],
    'save_last': np.int64(5),
    'ca_saveall': np.int64(0),
    'dt00': np.nan,                      # np.nan for don't care
}

# save results to this path (in matlab file format)
save_to = os.path.join('.', 'sim.mat')

# run simulation
start = time.time()
run_simulation(N=np.int64(1e8), params_phys=params_phys, params_sim=params_sim,
               save_to=save_to, dump_to=save_to, save_as='mat',
               post_process=False)
print('time:', time.time() - start)
