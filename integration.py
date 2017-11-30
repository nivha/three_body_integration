# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:13:41 2017

@author: Niv Haim (Weizmann Institute of Science)
"""

import numpy as np
from numpy.linalg import norm
from numba import jit
from sim.utils import cross_jit
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
def get_jz_eff(G, m1, m2, m3, a, x, v):
    M = m1 + m2
    mu_in = m1 * m2 / M
    mu_out = m3 * M / (M + m3)

    # compute Jv_in
    r_in = x[3:6] - x[0:3]
    v_in = v[3:6] - v[0:3]
    Jv_in = mu_in * cross_jit(r_in, v_in)

    # compute Jv_out
    rcms_in = (m1*x[3:6] + m2*x[0:3]) / M
    vcms_in = (m1*v[3:6] + m2*v[0:3]) / M
    r_out = x[6:9] - rcms_in
    v_out = v[6:9] - vcms_in
    Jv_out = mu_out * cross_jit(r_out, v_out)

    Jout = norm(Jv_out)
    Jcirc = mu_in * np.sqrt(G*M*a)

    jz_eff = Jv_in @ Jv_out / Jout / Jcirc + norm(Jv_in)**2 / Jout / Jcirc / 2

    return jz_eff


@jit(nopython=True)
def get_r0_rm_rp(s, i_delta):
    """ compute 3 points r0, r_minus and r_plus to determine apsis
        compute these at s.i-i_delta and  s.i-2*i_delta
    """
    xp = s.Xlast[:, s.i % s.save_last]
    x0 = s.Xlast[:, (s.i - i_delta) % s.save_last]
    xm = s.Xlast[:, (s.i - 2 * i_delta) % s.save_last]

    rp = norm(xp[0:3] - xp[3:6])
    r0 = norm(x0[0:3] - x0[3:6])
    rm = norm(xm[0:3] - xm[3:6])

    return r0, rm, rp


@jit(nopython=True)
def is_peri(r0, rm, rp):
    return r0 < rp and r0 < rm


@jit(nopython=True)
def is_apo(r0, rm, rp):
    return r0 > rp and r0 > rm


@jit(nopython=True)
def is_system_broken(G, m1, m2, m3, x, v, rmax):

    R = get_R(x)
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
def check_stopping_conditions(s, x, v, t, N):
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

        if s.nP % 1000 == 0:
            print('nP:', s.nP, 'i:', s.i, 'N:', N)
            if is_system_broken(s.G, s.m1, s.m2, s.m3, x, v, s.rmax):
                print('[BREAK] system broken')
                fin_reason = SYSTEM_BROKEN

    return fin_reason


@jit(nopython=True)
def update_dE_max(s, x, v):
    # compute dE
    R = get_R(x)
    U = get_U(s.G, s.m1, s.m2, s.m3, R)
    K = get_K(v, s.m1, s.m2, s.m3)
    E = U + K
    dE = np.abs(E/s.E0 - 1)
    # update if bigger than previous
    if dE > s.dE_max:
        s.dE_max = dE
        s.dE_max_i = s.i
        s.dE_max_x = x
        s.dE_max_v = v


@jit(nopython=True)
def save_all_params(s, i):
    s.X[:, s.idx] = s.Xlast[:, i % s.save_last]
    s.V[:, s.idx] = s.Vlast[:, i % s.save_last]
    s.DT[s.idx] = s.DTlast[i % s.save_last]
    s.T[s.idx] = s.Tlast[i % s.save_last]
    s.idx += 1


@jit(nopython=True)
def handle_jz_eff(s, x_apo, v_apo):
    jz_eff = get_jz_eff(s.G, s.m1, s.m2, s.m3, s.a, x_apo, v_apo)

    # maintain crossings index
    if jz_eff * s.jz_eff < 0:
        s.jz_eff_crossings += 1
    if abs(jz_eff) < s.jz_eff_min:
        s.jz_eff_min = jz_eff
        s.jz_eff_min_x = x_apo
        s.jz_eff_min_v = v_apo
    s.jz_eff = jz_eff

    # maintain stats (compute mean and M2 with Welford algorithm)
    s.jz_eff_n += 1
    delta = s.jz_eff - s.jz_eff_mean
    s.jz_eff_mean += delta / s.jz_eff_n
    delta2 = s.jz_eff - s.jz_eff_mean
    s.jz_eff_M2 += delta * delta2


@jit(nopython=True)
def treat_apocenter(s, i_apo, t):
    x_apo = s.Xlast[:, i_apo % s.save_last]
    v_apo = s.Vlast[:, i_apo % s.save_last]

    # save params every save_every_P period
    if s.save_every_P > 0 and t / s.P_in - s.save_every_P * s.save_every_P_i >= 0:
        save_all_params(s, i_apo)
        s.save_every_P_i += 1

    # update dE_max
    update_dE_max(s, x_apo, v_apo)

    # handle jz_eff stuff
    handle_jz_eff(s, x_apo, v_apo)


@jit(nopython=True)
def treat_pericenter(s, r0, i_peri):
    if s.ca_saveall or r0 < s.closest_approach_r:
        x_peri = s.Xlast[:, i_peri % s.save_last]
        v_peri = s.Vlast[:, i_peri % s.save_last]
        t_peri = s.Tlast[i_peri % s.save_last]

        s.closest_approach_r = r0
        s.Ica[s.caidx] = i_peri
        s.Xca[:, s.caidx] = x_peri
        s.Vca[:, s.caidx] = v_peri
        s.Tca[s.caidx] = t_peri
        s.Jzeffca[s.caidx] = get_jz_eff(s.G, s.m1, s.m2, s.m3, s.a, x_peri, v_peri)
        s.caidx += 1


@jit(nopython=True)
def save_state_params(s, x, v, dt, t):
    # maintain "last" arrays
    s.Xlast[:, s.i % s.save_last] = x
    s.Vlast[:, s.i % s.save_last] = v
    s.DTlast[s.i % s.save_last] = dt
    s.Tlast[s.i % s.save_last] = t

    # save all state every once in a while
    if not s.save_every_P and s.i % s.save_every == 0:
        save_all_params(s, s.i)

    # handle apsis stuff
    i_delta = 1
    if s.i < 2 * i_delta: return
    r0, rm, rp = get_r0_rm_rp(s, i_delta)
    if r0 < s.a and is_peri(r0, rm, rp):
        treat_pericenter(s, r0, i_peri=s.i - i_delta)
    elif s.a < r0 < 2.5 * s.a and is_apo(r0, rm, rp):
        treat_apocenter(s, i_apo=s.i - i_delta, t=t)


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

    # save first state
    if s.i == 0: save_all_params(s, s.i)

    while s.i < N:

        # save params and check stopping conditions
        xfixed = x - v * dt / 2
        save_state_params(s, xfixed, v, dt, t)
        s.fin_reason = check_stopping_conditions(s, xfixed, v, t, N)
        if s.fin_reason != REASON_NONE: break

        # Kick (update v)
        R = get_R(x)
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
