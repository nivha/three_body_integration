
import os
import numpy as np
import time
from sim.utils import r_dissip, year, t_unit
from sim.run_simulation import run_simulation

params_phys = {
    'G': 1,
    'm1': 0.6,
    'm2': 0.6,
    'm3': 0.7,
    'a': 1,
    'e': 0.1,
    'M0_in': np.deg2rad(0),
    'f_out': np.deg2rad(0),
    'inclination': np.deg2rad(85),
    'Omega': np.deg2rad(275),
    'omega': np.deg2rad(90),
    'rper_over_a': 8,
    'eper': 0.01,
}

params_sim = {
    'dt00': np.nan,  # np.nan for don't care
    'samples_per_Pcirc': np.int64(2000),
    'max_periods': np.int64(3000),  # -1 for don't care
    'save_every': np.int64(10),
    'save_every_P': np.int64(10),
    'rmin': r_dissip,
    'tmax': 5e9 * year / t_unit,
    'rmax': 50 * params_phys['a'],
    'save_last': np.int64(10),
    'ca_saveall': np.int64(1),
}

RESULT_PATH_PC_LOCAL = os.path.join('c:', os.sep, 'tmp', 'sim1.mat')

start = time.time()
run_simulation(np.int64(1e7), params_phys, params_sim,
               RESULT_PATH_PC_LOCAL, save_as='mat', post_process=True)
print('time:', time.time() - start)

print("done")