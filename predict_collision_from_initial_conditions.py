# -*- coding: utf-8 -*-
"""
@author     Niv Haim <niv.haim@weizmann.ac.il>

@details    This file implements the criterion for close approaches
in triple systems with an equal mass binary. The crietrion is
computed from the initial conditions of a given such system.
For an example please see main section below.

@section    The MIT License (MIT)
Copyright (c) 2018 Niv Haim, Liantong Luo, Weizmann Institute of Science

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import numba as nb
from numpy.linalg import norm
from numpy import cos, sin, sqrt


@nb.jit(nopython=True)
def get_je(e, i, Omega, omega):
    """Compute the angular momentum and eccentricity vectors of 
       two body system in 3D.

    Args:
        e: eccentricity.
        i: inclination of the orbit relative to z-axis
        Omega: Longitude of the ascending node
        omega: Argument of periapsis

    Returns:
        jv: the angular momentum vector
        ev: the eccentricity vector
    """
    j = sqrt(1 - e**2)
    jx = j * sin(i) * sin(Omega)
    jy = -j * sin(i) * cos(Omega)
    jz = j * cos(i)
    ex = e * cos(omega) * cos(Omega) - e * sin(omega) * cos(i) * sin(Omega)
    ey = e * cos(omega) * sin(Omega) + e * sin(omega) * cos(i) * cos(Omega)
    ez = e * sin(omega) * sin(i)

    jv = np.array([jx, jy, jz])
    ev = np.array([ex, ey, ez])

    return jv, ev


@nb.jit(nopython=True)
def fjv_fev(m1, m2, m3, jv, ev, e_out, f_out):
    """Computes the expansion terms from Luo et al. (2016)
        See there: eq.(31) and Appendix B1
    """

    # Expressions from Appendix B1 in Luo et al. (2016)
    Jf0 = (3/4) * np.array([                                                   
        ( -5*ev[1]*ev[2] + jv[1]*jv[2]),
        (  5*ev[0]*ev[2] - jv[0]*jv[2]),
           0,
    ])
    Jsf2 = (3/4) * np.array([
        ( -5*ev[0]*ev[2] + jv[0]*jv[2]),
        (  5*ev[1]*ev[2] - jv[1]*jv[2]),
        (  5*ev[0]**2 - 5*ev[1]**2 - jv[0]**2 + jv[1]**2),
    ])
    Jcf2 = (3/4) * np.array([
        (   5*ev[1]*ev[2] -   jv[1]*jv[2]),
        (   5*ev[0]*ev[2] -   jv[0]*jv[2]),
        ( -10*ev[0]*ev[1] + 2*jv[0]*jv[1]),
    ])
    Jcf1 = e_out * (Jf0 + Jcf2 / 2)
    Jcf3 = e_out *  Jcf2 / 2
    Jsf1 = e_out *  Jsf2 / 2
    Jsf3 = e_out *  Jsf2 / 2

    Ef0 = (3/4) * np.array([
        ( -3*ev[2]*jv[1] -   ev[1]*jv[2] ),
        (  3*ev[2]*jv[0] +   ev[0]*jv[2] ),
        (  2*ev[1]*jv[0] - 2*ev[0]*jv[1] ),
    ])
    Esf2 = (3/4) * np.array([
        (    ev[2]*jv[0] - 5*ev[0]*jv[2] ),
        (   -ev[2]*jv[1] + 5*ev[1]*jv[2] ),
        (  4*ev[0]*jv[0] - 4*ev[1]*jv[1] ),
    ])
    Ecf2 = (3/4) * np.array([
        (   -ev[2]*jv[1] + 5*ev[1]*jv[2] ),
        (   -ev[2]*jv[0] + 5*ev[0]*jv[2] ),
        ( -4*ev[1]*jv[0] - 4*ev[0]*jv[1] ),
    ])
    Ecf1 = e_out * (Ef0 + Ecf2 / 2)
    Ecf3 = e_out *  Ecf2 / 2
    Esf1 = e_out *  Esf2 / 2
    Esf3 = e_out *  Esf2 / 2

    # Expression from equation (31) in Luo et al. (2016)
    fjv = (
            cos(f_out)*Jsf1 + cos(2*f_out)*Jsf2/2 + cos(3*f_out)*Jsf3/3
          - sin(f_out)*Jcf1 - sin(2*f_out)*Jcf2/2 - sin(3*f_out)*Jcf3/3
    )
    fev = (
            cos(f_out)*Esf1 + cos(2*f_out)*Esf2/2 + cos(3*f_out)*Esf3/3
          - sin(f_out)*Ecf1 - sin(2*f_out)*Ecf2/2 - sin(3*f_out)*Ecf3/3
    )
    
    return fjv, fev


@nb.jit(nopython=True)
def get_je_mean(m1, m2, m3, a, a_out, e_out, e, jv, ev, f_out):
    """Compute j_mean and e_mean from the initial conditions
       from Luo et al. (2016), see there: eq. (20), (31) and Appendix B1
    """

    fjv, fev = fjv_fev(m1, m2, m3, jv, ev, e_out, f_out)
    esa = (a/a_out)**(3/2) * (1-e_out**2)**(-3/2) * m3 / sqrt((m1 + m2) * (m1 + m2 + m3))
    
    jv_mean = jv + esa * fjv
    ev_mean = ev + esa * fev
    
    return jv_mean, ev_mean


class OrbitalParameters(object):
    """Computes orbital parameters for systems of three bodies.

    Retrieves initial conditions for a three body system and compute
    other parameters.

    Attributes:
        G: Gravitational Constant
        m1, m2, m3:  Masses of the the three bodies
        a, a_out: Inner and outer semi major axes
        e, e_out: Inner and outer eccentricities
        inclination: Mutual inclination between inner and outer orbits
        Omega: Longitude of the ascending node
        omega: Argument of periapsis
        f_out: True anomaly of the outer orbit
    """    
    def __init__(self, G, m1, m2, m3, e, a, a_out, 
                 e_out, inclination, Omega, omega, f_out):
        self.G = G
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.e = e
        self.a = a
        self.a_out = a_out
        self.e_out = e_out
        self.inclination = inclination
        self.Omega = Omega
        self.omega = omega
        self.f_out = f_out

        self.set_orbital_parameters(G, m1, m2, m3, e, a, a_out, 
                                    e_out, inclination, Omega, omega, f_out)

    def set_orbital_parameters(s, G, m1, m2, m3, e, a, a_out,
                               e_out, inclination, Omega, omega, f_out):
        # basic params
        s.Min = m1 + m2
        s.Mout = m1 + m2 + m3
        s.muin = m1 * m2 / s.Min
        s.muout = s.Min * m3 / s.Mout
        s.Jcirc = s.muin * sqrt(G * s.Min * a)

        # outer orbit
        s.jv_out, s.ev_out = get_je(e_out, 0, 0, 0)
        s.Jv_out = s.jv_out * s.muout * sqrt(G * s.Mout * a_out)
        s.Jout = norm(s.Jv_out)

        # inner orbit
        s.jv_in, s.ev_in = get_je(e, inclination, Omega, omega)
        s.Jv_in = s.jv_in * s.Jcirc 
        s.j = norm(s.jv_in)

        # general
        s.Jv_tot = s.Jv_in + s.Jv_out
        s.jz_eff = s.jv_in @ s.Jv_out / s.Jout + s.j**2 * s.Jcirc / 2 / s.Jout
        s.jv_mean, s.ev_mean = get_je_mean(m1, m2, m3, a, a_out, 
                                           e_out, e, s.jv_in, s.ev_in, s.f_out)
 
        # jz_eff_mean
        s.Jv_out_mean = s.Jv_tot - s.jv_mean * s.Jcirc
        s.Jout_mean = norm(s.Jv_out_mean)
        s.jz_eff_mean = s.jv_mean @ s.Jv_out_mean / s.Jout_mean + norm(s.jv_mean)**2 * s.Jcirc / 2 / s.Jout_mean
        s.esa = (a/a_out)**(3/2) * (1-e_out**2)**(-3/2) * m3 / sqrt((m1 + m2) * (m1 + m2 + m3))
    
        # maximal delta in jz_eff_mean        
        C = -e**2 + (5/2)*s.ev_in[2]**2 - (1/2)*s.jv_in[2]**2
        s.jz_delta = s.esa*(15/8)*(3/5 - (2/5)*C - -(1/5)*s.jz_eff_mean**2)*(1 + (2*e_out*sqrt(2)/3))
        s.jzeff_delta_factor = 1 + s.jz_eff_mean * s.Jcirc / s.Jout_mean
        s.jz_eff_delta_max = s.jz_delta * s.jzeff_delta_factor
        
        
if __name__=="__main__":
    """ An example on how to compute the criterion for close approaches from
        the initial conditions of a three body system
    """
    
    # initial conditions for some three body system
    d = {
            'G': 1.0,
            'm1': 0.6,
            'm2': 0.6,
            'm3': 0.5,
            'a': 1.0,
            'a_out': 22,
            'e': 0.89,
            'e_out': 0.72,
            'inclination': 1.98,
            'Omega': 5.57,
            'omega': 5.45,
            'f_out': 1.28,
    }
    
    # compute all relevant parameters
    op = OrbitalParameters(**d)
    
    # check criterion for close approaches
    criterion = op.jz_eff_delta_max > op.jz_eff_mean
    
    # report result
    print('Is this system likely to experience a close approach?')
    if criterion:
        print('Yes!')
    else:
        print('Probably not :(')
    
    