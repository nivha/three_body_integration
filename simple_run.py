
import os
import sys
PROJECT_DIR = os.path.join(os.sep, 'home', 'labs', 'kushnir', 'nivh')
CODE_DIR = os.path.join(PROJECT_DIR, 'three_body_integration')
sys.path.append(CODE_DIR)

import numpy as np
import time
from sim.run_simulation import run_simulation

RESULT_DIR_PC_LOCAL = os.path.join('c:', os.sep, 'three_body_files')
RESULT_DIR_REMOTE = os.path.join(PROJECT_DIR)

DIR_LOCAL = os.path.join('Z:', os.sep)
DIR_REMOTE = PROJECT_DIR


# remote or local
BASE_DIRS = {
    'local': DIR_LOCAL,
    'remote': DIR_REMOTE,
}
SAVE_TO_DIR = {
    'local': RESULT_DIR_PC_LOCAL,
    'remote': RESULT_DIR_REMOTE,
}


def simple():
    params_phys = {
        'G': 1.0,
        'm1': 0.5,
        'm2': 0.5,
        'm3': 0.5,
        'a': np.double(1),
        'e': 0.1,
        'M0_in': 2.649,
        'M0_out': np.deg2rad(220),
        'inclination': 1.489,
        'Omega': 1.34,
        'omega': 0.932,
        'rper_over_a': 5.0,
        'eper': 0.425,
    }

    params_sim = {
        'dt00': np.nan,  # np.nan for don't care
        'samples_per_Pcirc': np.int64(500),
        'max_periods': np.int64(1000),  # -1 for don't care
        'dump_every': np.int64(10000),  # this is a heavy operation, better be a big number
        'save_every': np.int64(20),
        'save_every_P': np.int64(0),  # 0 for don't care
        'rmax': 50 * params_phys['a'],
        'save_last': np.int64(5),
        'ca_saveall': np.int64(0),
    }
    return params_phys, params_sim


def boaz_paper():
    params_phys = {
        'G': 1.0,
        'm1': 0.5,
        'm2': 0.5,
        'm3': 0.5,
        'a': np.double(1),
        'e': 0.1,
        'M0_in': np.deg2rad(0),
        'M0_out': np.deg2rad(0),
        'inclination': np.deg2rad(98),
        'Omega': np.deg2rad(0),
        'omega': np.deg2rad(0),
        'rper_over_a': 10,
        'eper': 0.5,
    }

    params_sim = {
        'dt00': np.nan,  # np.nan for don't care
        'samples_per_Pcirc': np.int64(1000),
        'max_periods': np.int64(5000),  # -1 for don't care
        'save_every': np.int64(20),
        'save_every_P': np.int64(0),  # 0 for don't care
        'rmax': 50 * params_phys['a'],
        'save_last': np.int64(5),
        'ca_saveall': np.int64(0),
    }
    return params_phys, params_sim


def boaz_wh():
    params_phys = {
        'G': 1.0,
        'm1': 0.6,
        'm2': 0.6,
        'm3': 0.7,
        'a': np.double(1),
        'e': 0.1,
        'M0_in': np.deg2rad(0),
        'M0_out': np.deg2rad(0),
        'inclination': np.deg2rad(85),
        'Omega': np.deg2rad(275),
        'omega': np.deg2rad(90),
        'rper_over_a': 8,
        'eper': 0.01,
    }

    params_sim = {
        'dt00': np.nan,  # np.nan for don't care
        'samples_per_Pcirc': np.int64(1000),
        'max_periods': np.int64(3000),  # -1 for don't care
        'dump_every': np.int64(10000),
        'save_every': np.int64(10),
        'save_every_P': np.int64(0),  # 0 for don't care
        'rmax': 50 * params_phys['a'],
        'save_last': np.int64(5),
        'ca_saveall': np.int64(0),
    }
    return params_phys, params_sim


def job():
    proj = 'm106m206m304'
    rovera = 5
    job_number = 431
    BASE_DIR = BASE_DIRS[HOST]

    config_file = os.path.join(BASE_DIR, 'ConfigFiles', 'Jobs', proj, str(rovera), '{}.config'.format(job_number))
    with open(config_file, 'rb') as f:
        d = np.load(f).item()
    print(d)
    params_phys, params_sim = d['params_phys'], d['params_sim']
    params_sim['samples_per_Pcirc'] = np.int64(250)
    params_sim['max_periods'] = np.int64(68000)
    params_sim['save_every'] = np.int64(20)
    params_sim['save_every_P'] = np.int64(0)
    params_sim['dump_every'] = np.int64(1000000)  # this is a heavy operation, better be a big number
    print('params loaded from', config_file)
    return params_phys, params_sim


HOST = 'remote'  # <--- change this to 'remote' or 'local'
params_phys, params_sim = job()
save_to = os.path.join(SAVE_TO_DIR[HOST], 'sim5.mat')

start = time.time()
run_simulation(np.int64(26000000), params_phys, params_sim, save_to=save_to, dump_to=save_to, save_as='mat', post_process=False)
print('time:', time.time() - start)
