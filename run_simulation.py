# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:13:41 2017

@author: Niv Haim (Weizmann Institute of Science)
"""

import numpy as np
from sim.utils import year, t_unit, r_dissip
import os
from scipy.io import savemat
from sim.sim_state import SimState, get_state_dict, inject_config_params, initialize_state
from sim.integration import advance_state
from sim.post_processing import _post_process


def process_state(N, sim_state, post_process=False):

    # don't integrate if
    if sim_state.E0 > 0:
        print('system configuration is unbounded, E0:', sim_state.E0)
        return get_state_dict(sim_state)

    # integrate
    print('advancing state', flush=True)
    advance_state(sim_state, N)

    # dump state (with or without post processing)
    if post_process:
        print('post processing')
        # scaling_params = {'Msun': Msun, 'G': G, 'au': au, 't_unit': t_unit}
        scaling_params = {'Msun': 1.0, 'G': 1.0, 'au': 1.0, 't_unit': 1.0}  # don't scale anything
        op = _post_process(sim_state, scaling_params)
        return op.__dict__

    return get_state_dict(sim_state)


def run_simulation(N, params_phys, params_sim, path_dst_dump, save_as='mat', d_dump=None, post_process=False):

    print('run simulation')

    # create state instance
    vsize = max(np.int64(N / params_sim['save_every']), np.int64(5000))
    sim_state = SimState(vsize, params_sim['save_last'])

    # create parameters dict
    d = {}
    d.update(params_phys)
    d.update(params_sim)

    # initialize state from configuration parameters
    inject_config_params(sim_state, **d)
    initialize_state(sim_state)

    # print some stats
    print("dt0:", sim_state.dt0)
    print("jz_eff0:", sim_state.jz_eff0)
    print("jz_eff_crossed_zero:", sim_state.jz_eff_crossed_zero)

    # process simulation (advance and post_process)
    result_d = process_state(N, sim_state, post_process)

    # dump d_dump into result
    if d_dump is not None:
        result_d.update(d_dump)

    # dump
    if save_as != 'mat':
        raise Exception('save_as must be \'mat\'')
    savemat(path_dst_dump, mdict=result_d, oned_as='column')

    print("dumped state to {}".format(path_dst_dump))


# def load_state_from_file(path_src_dump):
#     d_prev = np.load(path_src_dump).item()
#     print('loaded state from file', path_src_dump)
#
#     # create state instance
#     vsize = np.int64(N / d_prev['save_every'])
#     sim_state = SimState(vsize, d_prev['save_last'])
#     print('BEFORE:', sim_state.X.shape)
#     # load data from file into state
#     print(d_prev.keys())
#     for k, v in d_prev.items():
#         if isinstance(d_prev[k], np.ndarray):
#             arr = getattr(sim_state, k)
#             if v.ndim == 1:
#                 arr[:v.shape[0]] = v
#             elif v.ndim == 2:
#                 arr[:, :v.shape[1]] = v
#             else:
#                 raise Exception('something is wrong with dimensions of array from loaded file ({} with ndim {})'.format(k, v.ndim))
#         else:
#             setattr(sim_state, k, v)
#     print('copied file to state')
#     print('AFTER:', sim_state.X.shape)
#     return sim_state
#
#
# def continue_simulation(N, path_src_dump, path_dst_dump, post_process=False):
#
#     print('continue simulation')
#     sim_state = load_state_from_file(path_src_dump)
#
#     # process simulation (advance and post_process)
#     result_d = process_state(N, sim_state, post_process)
#
#     savemat(path_dst_dump, mdict=result_d, oned_as='column')
#     print("dumped state to {}".format(path_dst_dump))


if __name__ == "__main__":
    RESULT_PATH_PC_LOCAL = os.path.join('c:', os.sep, 'tmp', 'sim1.mat')

    # job_number = 965
    # config_file = os.path.join('Z:', os.sep, 'ConfigFiles', 'Jobs', '{}.config'.format(job_number))
    # with open(config_file, 'rb') as f:
    #     d = np.load(f).item()
    # params_phys, params_sim = d['params_phys'], d['params_sim']
    # params_sim['max_periods'] = np.int64(1000)
    # params_sim['N'] = np.int64(1e6)
    # params_sim['save_every'] = np.int64(1)
    # print('params loaded from', config_file)

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
        'dt00': np.nan,
        'max_periods': np.int64(3000),  # -1 for don't care
        'save_every': np.int64(10),
        'samples_per_Pcirc': np.int64(2000),
        'rmax': 50 * params_phys['a'],
        'save_last': np.int64(10),
        'ca_saveall': np.int64(0),
    }

    run_simulation(np.int64(1e7), params_phys, params_sim, RESULT_PATH_PC_LOCAL, save_as='mat', post_process=True)
    # continue_simulation(1e8, RESULT_MID_PATH_PC_LOCAL, RESULT_PATH_PC_LOCAL, post_process=True)

    print("done")



