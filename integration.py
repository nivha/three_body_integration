# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:13:41 2017

@author: Niv Haim (Weizmann Institute of Science)
"""

import numpy as np
from numpy.linalg import norm
from numba import jit
from sim.utils import REASON_NONE, TNAN, MAX_PERIODS, TMAX, RMIN, SYSTEM_BROKEN, BAD, FINISHED_ITERATIONS


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
    a12_ = (G/norm(R[0:3])**3) * R[0:3]
    a13_ = (G/norm(R[3:6])**3) * R[3:6]
    a23_ = (G/norm(R[6:9])**3) * R[6:9]
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
def update_closest_approach(i, caidx, closest_approach_r, save_last, ca_saveall, Xlast, Vlast, Tlast, Ica, Xca, Vca, Tca):
    # update closest approach variables
    x_prev = Xlast[:, (i - 1) % save_last]
    r12_prev = norm(x_prev[0:3] - x_prev[3:6])

    if i < 2:
        return caidx, closest_approach_r

    if ca_saveall or r12_prev < closest_approach_r:
        # only update if this is really a minimum
        x_prev2 = Xlast[:, (i - 2) % save_last]
        r12_prev2 = norm(x_prev2[0:3] - x_prev2[3:6])

        x_cur = Xlast[:, i % save_last]
        r12_cur = norm(x_cur[0:3] - x_cur[3:6])

        if r12_cur > r12_prev and r12_prev2 > r12_prev:
            # print('>>> ca:', caidx)
            closest_approach_r = r12_prev
            Ica[caidx] = i - 1
            Xca[:, caidx] = Xlast[:, (i - 1) % save_last]
            Vca[:, caidx] = Vlast[:, (i - 1) % save_last]
            Tca[caidx] = Tlast[(i - 1) % save_last]
            caidx += 1
    return caidx, closest_approach_r


@jit(nopython=True)
def check_stopping_conditions(x, v, t, nP, steps_per_P, i, N, P_in, G, R, m1, m2, m3, rmax, max_periods, tmax):
    fin_reason = REASON_NONE

    if np.isnan(t):
        print('[BREAK] t is nan')
        fin_reason = TNAN

    # break if max_periods reached
    if max_periods > 0 and nP >= max_periods:
        print('[BREAK] max periods reached:', nP)
        fin_reason = MAX_PERIODS

    # break if max time reached
    if t > tmax:
        print('[BREAK] tmax reached:', t)
        fin_reason = TMAX

    # # break if rmin reached
    # if r12 < rmin:
    #     print('[BREAK] rmin reached:', r12)
    #     fin_reason = RMIN
    #     break

    # update nP and check stuff once in a while
    if t / P_in - nP > 1:
        nP += 1
        if nP == 1:
            steps_per_P = i
            print("steps per P:", i)
            if steps_per_P > 3e4:  # TODO: verify this is a good condition
                print('[BREAK] assuming bad system')
                fin_reason = BAD

        if nP % 1000 == 0:
            print('nP:', nP, 'i', i, '/', N, '(', (i / N), ')')
            if is_system_broken(G, R, x, v, m1, m2, m3, rmax):
                print('[BREAK] system broken')
                fin_reason = SYSTEM_BROKEN

    return nP, steps_per_P, fin_reason


@jit(nopython=True)
def save_state_params(s, x, v, dt, t):
    if s.i % s.save_every == 0:
        s.X[:, s.idx] = x - v * dt / 2
        s.V[:, s.idx] = v
        s.DT[s.idx] = dt
        s.T[s.idx] = t
        s.idx += 1
    # save last save_last
    s.Xlast[:, s.i % s.save_last] = x - v * dt / 2
    s.Vlast[:, s.i % s.save_last] = v
    s.Tlast[s.i % s.save_last] = t


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
    return x


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
    x += v * dt/2

    while s.i < N:

        # save params once in a while
        save_state_params(s, x, v, dt, t)

        # get distances between masses
        R = get_R(x)

        # closest approach handling
        s.caidx, s.closest_approach_r = update_closest_approach(s.i, s.caidx, s.closest_approach_r, s.save_last, s.ca_saveall,
                                                                s.Xlast, s.Vlast, s.Tlast, s.Ica, s.Xca, s.Vca, s.Tca)

        # Stopping Conditions:
        s.nP, s.steps_per_P, s.fin_reason = check_stopping_conditions(x, v, t,
                                                                      s.nP, s.steps_per_P,
                                                                      s.i, N,
                                                                      s.P_in,
                                                                      s.G, R, s.m1, s.m2, s.m3,
                                                                      s.rmax, s.max_periods, s.tmax)
        if s.fin_reason != REASON_NONE:
            break

        # Kick (update v)
        v = kick(v, s, R)

        # Drift (update x)
        x = drift(x, s, v)

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
    print('closest approach:', s.closest_approach_r)
