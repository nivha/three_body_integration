# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:13:41 2017

@author: Niv Haim (Weizmann Institute of Science)
"""

import numpy as np
from scipy.io import savemat
from sim.sim_state import SimState, get_state_dict, inject_config_params, initialize_state, chop_arrays, make_copy
from sim.integration import advance_state, FINISHED_ITERATIONS
from sim.post_processing import _post_process


def process_state(N, sim_state, dump_to, post_process=False):

    # don't integrate if
    if sim_state.E0 > 0:
        print('system configuration is unbounded, E0:', sim_state.E0)
        return get_state_dict(sim_state)

    # integrate
    print('advancing state', flush=True)
    while sim_state.i < N:
        advance_state(sim_state, sim_state.i + sim_state.dump_every)
        if sim_state.fin_reason != FINISHED_ITERATIONS: break
        print('i/N_total:', (sim_state.i / N))

        # dump a copy of the state
        sim_state_copy = make_copy(sim_state)
        chop_arrays(sim_state_copy)
        d = get_state_dict(sim_state_copy)
        savemat(dump_to, mdict=d, oned_as='column')
        print("dumped state to {}".format(dump_to))

    chop_arrays(sim_state)

    # dump state (with or without post processing)
    if post_process:
        print('post processing')
        # scaling_params = {'Msun': Msun, 'G': G, 'au': au, 't_unit': t_unit}
        scaling_params = {'Msun': 1.0, 'G': 1.0, 'au': 1.0, 't_unit': 1.0}  # don't scale anything
        op = _post_process(sim_state, scaling_params)
        return op.__dict__

    return get_state_dict(sim_state)


def run_simulation(N, params_phys, params_sim, save_to, dump_to=None, save_as='mat', d_dump=None, post_process=False):

    print('run simulation')

    # create state instance
    vsize = max(np.int64(1 + N / params_sim['save_every']), params_sim['max_periods'])
    print('vsize:', vsize)
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
    print("jz_eff_crossings:", sim_state.jz_eff_crossings)

    # process simulation (advance and post_process)
    result_d = process_state(N, sim_state, dump_to, post_process)

    # dump d_dump into result
    if d_dump is not None:
        result_d.update(d_dump)

    # dump
    if save_as != 'mat':
        raise Exception('save_as must be \'mat\'')
    savemat(save_to, mdict=result_d, oned_as='column')

    print("dumped state to {}".format(save_to))


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





