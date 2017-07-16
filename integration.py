# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:13:41 2017

@author: Niv Haim (Weizmann Institute of Science)
"""

import numpy as np
from numpy.linalg import norm
from numba import jit
from sim.utils import REASON_NONE, TNAN, MAX_PERIODS, SYSTEM_BROKEN, BAD, FINISHED_ITERATIONS


@jit(nopython=True)
def get_R(x):
    x1 = x[0:3]
    x2 = x[3:6]
    x3 = x[6:9]
    r12 = x2 - x1
    r13 = x3 - x1
    r23 = x3 - x2
    R = np.concatenate((r12, r13, r23))
    return R


@jit(nopython=True)
def getA(G, m1, m2, m3, R):
    r12 = R[0:3]
    r13 = R[3:6]
    r23 = R[6:9]

    a12_ = (G/norm(r12)**3) * r12
    a13_ = (G/norm(r13)**3) * r13
    a23_ = (G/norm(r23)**3) * r23
    A = np.concatenate(((a12_*m2 + a13_*m3), (-a12_*m1 + a23_*m3), (-a13_*m1 - a23_*m2)))
    return A


@jit(nopython=True)
def fU(U, U0):
    return (U / U0)**(-3/2)


@jit(nopython=True)
def get_U(G, m1, m2, m3, R):
    U = - G * (m1 * m2 / norm(R[0:3]) + m1 * m3 / norm(R[3:6]) + m2 * m3 / norm(R[6:9]))
    return U


@jit(nopython=True)
def get_K(v, m1, m2, m3):
    v1 = v[0:3]
    v2 = v[3:6]
    v3 = v[6:9]
    K = 0.5 * (m1 * v1 @ v1 + m2 * v2 @ v2 + m3 * v3 @ v3)
    return K


@jit(nopython=True)
def get_E(G, mi, xi, vi, mj, xj, vj, xp, vp, mp):
    """ Returns the energy of mp and (mi,mj) combined """
    m_in = mi + mj
    cms_in = (mi * xi + mj * xj) / m_in
    v_in = (mi * vi + mj * vj) / m_in
    E = 0.5 * (m_in * norm(v_in)**2 + mp * norm(vp)**2) - G * m_in * mp / norm(xp - cms_in)
    return E


@jit(nopython=True)
def is_system_broken(G, R, x, v, m1, m2, m3, rmax):

    r12 = norm(R[0:3])
    r13 = norm(R[3:6])
    r23 = norm(R[6:9])

    # find inner and perturber
    x1 = x[0:3];x2 = x[3:6];x3 = x[6:9]
    v1 = v[0:3];v2 = v[3:6];v3 = v[6:9]
    if r13 / r12 > rmax:
        E = get_E(G, m1, x1, v1, m2, x2, v2, x3, v3, m3)
    elif r12 / r13 > rmax:
        E = get_E(G, m1, x1, v1, m3, x3, v3, x2, v2, m2)
    elif r12 / r23 > rmax:
        E = get_E(G, m2, x2, v2, m3, x3, v3, x1, v1, m1)
    else:
        return False

    if E > 0:
        return True
    else:
        return False


@jit(nopython=True)
def check_stopping_conditions(s, x, v, t, N, R):
    fin_reason = REASON_NONE

    if np.isnan(t):
        print('[BREAK] t is nan')
        fin_reason = TNAN

    # break if max_periods reached
    if s.max_periods > 0 and s.nP >= s.max_periods:
        print('[BREAK] max periods reached:', s.nP)
        fin_reason = MAX_PERIODS

    # update nP, update steps_per_P, break if system is broken
    if t / s.P_in - s.nP > 1:
        s.nP += 1
        if s.nP == 1:
            s.steps_per_P = s.i
            print("steps per P:", s.i)
            if s.steps_per_P > 3e4:  # TODO: verify this is a good condition
                print('[BREAK] assuming bad system')
                fin_reason = BAD

        if s.nP % 1000 == 0:
            print('nP:', s.nP, 'i', s.i, '/', N, '(', (s.i / N), ')')
            if is_system_broken(s.G, R, x, v, s.m1, s.m2, s.m3, s.rmax):
                print('[BREAK] system broken')
                fin_reason = SYSTEM_BROKEN

    return fin_reason


@jit(nopython=True)
def update_dE_max(s, x, v, dt):
    # compute dE
    R = get_R(x - v * dt/2)
    U = get_U(s.G, s.m1, s.m2, s.m3, R)
    K = get_K(v, s.m1, s.m2, s.m3)
    E = U + K
    dE = np.abs(E/s.E0 - 1)
    # update if bigger than previous
    if dE > s.dE_max:
        s.dE_max = dE
        s.dE_max_i = s.i


@jit(nopython=True)
def save_all_params(s, x, v, dt, t):
    s.X[:, s.idx] = x - v * dt / 2
    s.V[:, s.idx] = v
    s.DT[s.idx] = dt
    s.T[s.idx] = t
    s.idx += 1


@jit(nopython=True)
def save_state_params(s, x, v, dt, t):

    # maintain "last" arrays
    s.Xlast[:, s.i % s.save_last] = x - v * dt / 2
    s.Vlast[:, s.i % s.save_last] = v
    s.Tlast[s.i % s.save_last] = t

    # maintain "all" arrays
    if not s.save_every_P and s.i % s.save_every == 0:
        save_all_params(s, x, v, dt, t)

    # only consider i > 2
    if s.i < 2: return

    # compute 2 previous states
    x_cur = s.Xlast[:, s.i % s.save_last]
    r12_cur = norm(x_cur[0:3] - x_cur[3:6])
    x_prev = s.Xlast[:, (s.i - 1) % s.save_last]
    r12_prev = norm(x_prev[0:3] - x_prev[3:6])
    x_prev2 = s.Xlast[:, (s.i - 2) % s.save_last]
    r12_prev2 = norm(x_prev2[0:3] - x_prev2[3:6])

    # apocenter
    if r12_prev > r12_cur and r12_prev > r12_prev2:
        update_dE_max(s, x, v, dt)
        if s.save_every_P and s.nP % s.save_every_P == 0:
            save_all_params(s, x, v, dt, t)

    # pericenter
    if s.ca_saveall or r12_prev < s.closest_approach_r:
        if r12_prev < r12_cur and r12_prev < r12_prev2:
            s.closest_approach_r = r12_prev
            s.Ica[s.caidx] = s.i - 1
            s.Xca[:, s.caidx] = s.Xlast[:, (s.i - 1) % s.save_last]
            s.Vca[:, s.caidx] = s.Vlast[:, (s.i - 1) % s.save_last]
            s.Tca[s.caidx] = s.Tlast[(s.i - 1) % s.save_last]
            s.caidx += 1


@jit(nopython=True)
def kick(v, s, R):
    a = getA(s.G, s.m1, s.m2, s.m3, R)
    U = get_U(s.G, s.m1, s.m2, s.m3, R)
    dt = s.dt0 * fU(U, s.U_init)
    v += a * dt
    return v


@jit(nopython=True)
def drift(x, s, v):
    K = get_K(v, s.m1, s.m2, s.m3)
    dt = s.dt0 * fU(s.E0 - K, s.U_init)
    x += v * dt
    return x, dt


@jit(nopython=True)
def advance_state(s, N):
    """ s: simulation state, numba.jitclass containing all relevant variables
        N: iterate simulation up to N iterations (from s.i)
    """

    # initiate loop variables
    x = s.Xlast[:, s.i % s.save_last].copy()
    v = s.Vlast[:, s.i % s.save_last].copy()
    t = s.Tlast[s.i % s.save_last]

    # calc x1/2 for leapfrog
    K = get_K(v, s.m1, s.m2, s.m3)
    dt = s.dt0 * fU(s.E0 - K, s.U_init)
    x += v * dt / 2
    print('dt at t=0:', dt)

    while s.i < N:

        # save params
        save_state_params(s, x, v, dt, t)

        # get distances between masses
        R = get_R(x)

        # check stopping conditions
        s.fin_reason = check_stopping_conditions(s, x, v, t, N, R)
        if s.fin_reason != REASON_NONE: break

        # Kick (update v)
        v = kick(v, s, R)

        # Drift (update x and get dt)
        x, dt = drift(x, s, v)

        # Time (update t)
        t += dt

        # update iterations index
        s.i += 1

    if s.i == N:
        s.fin_reason = FINISHED_ITERATIONS

    print('#fin_reason:', s.fin_reason)
    print('#iterations:', s.i)
    print('#periods:', s.nP)
    print('caidx:', s.caidx)
    print('idx:', s.idx)
    print('closest approach:', s.closest_approach_r)
