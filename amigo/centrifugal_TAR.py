#!/usr/bin/env python3
#
# File: centrifugal_TAR.py
# Authors: Vincent Prat <vincent.prat@cea.fr>
#          Timothy Van Reeth 
# License: GPL-3+
# Description: Module that computes Hough functions

import numpy as np
import matplotlib.pyplot as plt

def _A(epsilon):
    return np.asarray(1 + 2*epsilon)

def _B(x, epsl2, A_arr):
    Bs = A_arr - 3*epsl2*(1-x**2)
    return np.asarray(Bs)

def _C(A_arr, B_arr):
    return np.asarray(B_arr/A_arr)

def _D(x, A_arr, C_arr, spin):
    Ds = A_arr * (1 - (spin**2)*(x**2)*(C_arr**2))
    return np.asarray(Ds)

def _E(epsilon):
    return np.asarray(6*epsilon)

def _dA_dx(x, epsl2):
    return np.asarray(6*x*epsl2)

def _dB_dx(x, epsl2):
    dB = 12*x*epsl2
    return np.asarray(dB)

def _dC_dx(A_arr, B_arr, dA, dB):
    dC = (dB*A_arr - dA*B_arr)/(A_arr**2)
    return np.asarray(dC)

def _dD_dx(x, spin, A_arr, C_arr, dA, dC):
    x2 = x**2
    s2 = spin**2
    dD = dA*(1-s2*x2*(C_arr**2)) - 2*A_arr*C_arr*x*s2*(C_arr + x*dC)
    return np.asarray(dD)

def _dE_dx(x, eps2):
    return 18*x*eps2


def hough(nu, l, m, npts=400, lmbd_est=None, extra=False, only_lambda=False, 
                                                            epsl0=0., epsl2=0.):
    """ Compute Hough functions
    Parameters:
    - nu:       spin factor
    - l:        degree
    - m:        azimuthal order
    - npts:     number of points
    - lmbd_est: estimate of the eigenvalue
    - extra:    if True, returns extra quantities
    Return value:
    - hr, ht, hp
    - if extra, also their derivatives with respect to mu = cos theta
    """
    
    # enforce an even number of points
    m_size = npts // 2
    npts = m_size * 2 # (axi)symmetric

    # define the parity
    parity = (l-m) % 2

    # Calculate the interior/root points (mu_i = cos(((2i-1)Pi)/2N), where 
    # i = 1,...,N , N = total number of collocation points; Wang et al. 2016)
    n = np.arange(m_size)
    mu = np.cos(np.pi/npts * (n+0.5))
    
    s = np.sqrt(1-mu**2)
    
    
    # define the coefficients of the differential equation for the radial Hough 
    # function (see Section 2.1.2, Henneco et al. 2021)
    if((epsl0 == 0.) & (epsl2 == 0.)):
        denom = 1 - nu**2 * mu**2
        coeffs2 = s**2 / denom
        coeffs1 = -2 * mu * (1 - nu**2) / denom**2
        coeffs0 = nu * m * (1 + nu**2*mu**2) / denom**2 - m**2 / (s**2*denom)
        
    else:
        legl2 = 0.5*(3.*mu**2. - 1.)
        eps = epsl0 + epsl2*legl2
    
        Aa = _A(eps)
        Ba = _B(mu, epsl2,Aa)
        Ca = _C(Aa, Ba)
        Da = _D(mu, Aa, Ca, nu)
        Ea = _E(eps)
    
        dAdx = _dA_dx(mu, epsl2)
        dBdx = _dB_dx(mu, epsl2)
        dCdx = _dC_dx(Aa, Ba, dAdx, dBdx)
        dDdx = _dD_dx(mu, nu, Aa, Ca, dAdx, dCdx)
        dEdx = _dE_dx(mu, epsl2)
        
        denom = Da
        coeffs2 = (s**2)/Da
        coeffs1 = ( -(s**2)*dDdx - 2*mu*Da )/(Da**2) + ( (s**2)*dEdx )/Da
        coeffs0 = -( (m**2) / (s**2 * Da)) + ( m * nu * mu * Ca * dEdx / Da ) \
                  + ( m * nu * (mu*Da*dCdx - mu*Ca*dDdx + Ca*Da) / (Da**2) ) 
    
    # define the parity factor
    pf = m % 2

    # Calculate the Chebyshev matrix (see Wang et al. 2016; Boyd 2001)
    full = np.zeros((m_size, m_size), dtype=float)
    d0 = np.zeros_like(full)    # d0 = chebyshev polynomial T_n (/times pf)
    d1 = np.zeros_like(full)    # d1 = first derivative with respect to mu
    d2 = np.zeros_like(full)    # d2 = second derivative with respect to mu
    
    if extra:
        # calculate for other parity
        d1_other = np.zeros_like(full)
        d0_other = np.zeros_like(full)
    
    for i in range(m_size):
        
        for j in range(m_size):
            
            j_index = 2 * j + parity # cfr. 'n' in T_n
            cij = np.cos(np.pi*j_index/npts * (npts+i+0.5))
            sij = np.sin(np.pi*j_index/npts * (npts+i+0.5))
            
            if pf:
                d0[i,j] = cij * s[i]
                d1[i,j] = j_index * sij - cij * mu[i] / s[i]
                d2[i,j] = (-j_index**2 * cij * s[i]**2 \
                           - j_index*sij*mu[i]*s[i] - cij) / s[i]**3
            
            else:
                d0[i,j] = cij
                d1[i,j] = j_index * sij / s[i]
                d2[i,j] = j_index * (mu[i]*sij - j_index*cij*s[i]) / s[i]**3
            
            if extra:
                j_index = 2 * j + 1 - parity
                cij = np.cos(np.pi*j_index/npts * (npts+i+0.5))
                sij = np.sin(np.pi*j_index/npts * (npts+i+0.5))
                
                if pf:
                    d0_other[i,j] = cij
                    d1_other[i,j] = j_index * sij / s[i]
                
                else:
                    d0_other[i,j] = cij * s[i]
                    d1_other[i,j] = j_index * sij - cij * mu[i] / s[i]
    
    # multiply derivatives with inverse of T_n (d0) 
    # --> obtain unity matrix coeffs0 accompanying term 
    # (see Full matrix projection below)
    
    d0 = np.linalg.inv(d0)
    d1 = np.dot(d1,d0)
    d2 = np.dot(d2,d0)
    
    if extra:
        d0_other = np.linalg.inv(d0_other)
        d1_other = np.dot(d1_other,d0_other)

    # Full matrix projection into Chebyshev space of differential equation for 
    # the radial Hough function (see Appendix A, Prat et al. 2019)
    full = np.dot(np.diag(coeffs2), d2) + np.dot(np.diag(coeffs1), d1) \
           + np.diag(coeffs0)

    # computation of eigenvalues/-functions
    (vals, vecs) = np.linalg.eig(full)

    # remove complex eigenvalues
    ind = vals.imag == 0
    vals = vals[ind].real
    vecs = vecs[:,ind].real


    # remove positive eigenvalues
    ind = vals < 0
    vals = -vals[ind]
    vecs = vecs[:,ind]

    # sort eigenvalues and locate fundamental
    ind = np.argsort(vals)
    vals = vals[ind]
    vecs = vecs[:,ind]
    
    # if estimate for eigenvalue not provided, generate automatic estimate
    if lmbd_est is None:
        lmbd_est = l*(l+1) # eigenvalue for nu = 0 provided as estimate
    
    # first occurence of smallest distance to calculated eigenvalues
    ind_pos = np.argmin(np.abs(lmbd_est-vals)) 
    
    # eigenvalue for the radial Hough function differential equation
    val = vals[ind_pos]
        
    if(only_lambda):
        return val
        
    # eigenfunction for the radial Hough function differential equation
    hr = vecs[:, ind_pos]
    
    # normalisation of the radial Hough function (hr)
    if hr[m_size-1] < 0: # if last point is negative, switch sign
        hr *= -1
    hr /= np.abs(hr).max()
    
    # compute ht (latitudinal Hough function) from hr
    # see Appendix A (Prat et al. 2019), where Hr' denotes a derivative with
    # respect to theta. This has to be converted to a derivative with 
    # respect to mu (=cos(theta)) in order to obtain coefficients below.
    coeffs1_ht = - s**2 / denom
    coeffs0_ht = - m * nu * mu / denom
    full_ht = np.dot(np.diag(coeffs1_ht), d1) + np.diag(coeffs0_ht)
    
    ht = np.dot(full_ht, hr) / s
    
    # compute hp (azimuthal Hough function) from hr
    # again see Appendix A (Prat et al. 2019).
    coeffs1_hp = nu * mu * s**2 / denom
    coeffs0_hp = m / denom
    full_hp = np.dot(np.diag(coeffs1_hp), d1) + np.diag(coeffs0_hp)
    
    hp = np.dot(full_hp, hr) / s
    
    # calculate the extra terms requested
    if extra:
        
        hrp = np.dot(d1, hr)
            
        # htp
        c2_htp = -s**4 / denom
        c1_htp = ( s**2 / denom**2 ) \
                 * (-m*nu*mu*denom + mu + mu**3 * nu**2 - 2*mu*nu**2)
        c0_htp = m * nu * (2*mu**4 * nu**2 - mu**2 * nu**2 - 1) / denom**2
        full_htp = np.dot(np.diag(c2_htp), d2) + np.dot(np.diag(c1_htp), d1) \
                   + np.diag(c0_htp)
        htp = np.dot(full_htp, hr) / s**3
            
        # hpp
        c2_hpp = nu * mu * s**4 / denom
        c1_hpp = ( s**2 / denom**2 ) \
                 * (m*denom + nu + nu**3 * mu**2 - 2*nu*mu**2)
        c0_hpp = ( m * mu / denom**2 ) \
                 * (1 - nu**2 * mu**2 + 2 * nu**2 - 2 * nu**2 * mu**2)
        full_hpp = np.dot(np.diag(c2_hpp), d2) \
                   + np.dot(np.diag(c1_hpp), d1) + np.diag(c0_hpp)
        hpp = np.dot(full_hpp, hr) / s**3

    # append the symmetric terms
    mu = np.append(mu, -mu[::-1])
    s = np.append(s, s[::-1])
        
    if parity:
        hr = np.append(hr, -hr[::-1],)
        ht = np.append(ht, ht[::-1])
        hp = np.append(hp, -hp[::-1])
        
    else:
        hr = np.append(hr, hr[::-1])
        ht = np.append(ht, -ht[::-1])
        hp = np.append(hp, hp[::-1])

    # calculate the extra terms requested and generate output
    if extra:
        
        if parity:
            hrp = np.append(hrp, hrp[::-1])
            htp = np.append(htp, -htp[::-1])
            hpp = np.append(hpp, hpp[::-1])
        
        else:
            hrp = np.append(hrp, -hrp[::-1])
            htp = np.append(htp, htp[::-1])
            hpp = np.append(hpp, -hpp[::-1])

        return val, mu, hr, ht, hp, hrp, htp, hpp
        
    else:
        return val, mu, hr, ht, hp



def get_radial_Hough_function(nu,l,m):
    
    (lmbd, mu, hr, _, _) = hough_functions(nu,l,m)
    
    return lmbd, mu, hr



def get_theta_Hough_function(nu,l,m):

    (lmbd, mu, _, ht, _) = hough_functions(nu,l,m)

    return lmbd, mu, ht



def get_phi_Hough_function(nu,l,m):

    (lmbd, mu, _, _, hp) = hough_functions(nu,l,m)

    return lmbd, mu, hp




