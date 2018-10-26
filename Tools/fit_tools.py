"""

Module containing functions for fitting TGA data

"""

import numpy as np
from scipy.integrate import odeint
from scipy.signal import savgol_filter

from matplotlib import pyplot as plt

# Constants
R   = 8.314     # J/mol-K
e   = np.exp(1)

# Computes RHS for 'odeint' assuming series mechanism for 
def series_mechanism(m, T, K, beta):

    # parse parameters
    N_r     = K.shape[0]
    A       = K[0,:]
    E       = K[1,:]
    nu      = K[2,:]

    # compute rate constants
    r       = A*np.exp(-E/(R*T))/beta

    # compute RHS
    dm_dT   = np.zeros(N_r + 1)

    dm_dT[0] = -r[0] * m[0]
    
    for i in range(1, N_r):
        dm_dT[i] = -r[i]*m[i] + nu[i-1]*r[i-1]*m[i-1]
    dm_dT[-1] = nu[-1]*r[-1]*m[-2]

    return dm_dT

# Computes RHS for 'odeint' assuming parallel mechanism for 
def parallel_mechanism(m, T, K, beta):

    # parse parameters
    N_r     = K.shape[1]
    A       = K[0,:]
    E       = K[1,:]

    # compute rate constants
    r       = A*np.exp(-E/(R*T))/beta

    # compute RHS
    dm_dT       = np.zeros(N_r + 1)
    dm_dT[0:-1] = -r*m[0:-1]
    dm_dT[-1]   = -np.sum( r*m[0:-1] )
    
    return dm_dT

# Model prediction
def predict_tga( mech, K, beta, m_0, T ):
    
    if mech == 'series':
        m_ar    = odeint( series_mechanism, m_0, T, args=(K, beta) )
        m       = np.sum( m_ar, 1 ) 
    
    if mech == 'parallel':
        m_ar    = odeint( parallel_mechanism, m_0, T, args=(K, beta) )
        m       = m_ar[:,-1]

    return m 

# Generate manufactured, noisy TGA data for analysis
def generate_tga( mech, K, beta, T_range, smp_rate, noise ):
     
    # compute number of data points and temperature values
    N_pts   = int( (T_range[1] - T_range[0])*smp_rate/beta )
    T       = np.linspace( T_range[0], T_range[1], N_pts )

    N_r     = K.shape[1] 
    
    if mech == 'series':
        m_0     = np.zeros( N_r + 1 )
        m_0[0]  = 1.

        # integrate with known parameters
        m_ar    = odeint( series_mechanism, m_0, T, args=(K, beta) )
        m       = np.sum( m_ar, 1 )
    
    if mech == 'parallel':
        dm_i    = K[2,:] 
        m_0     = np.append( dm_i, 1. )

        # integrate with known parameters
        m_ar    = odeint( parallel_mechanism, m_0, T, args=(K, beta) )
        m       = m_ar[:,-1]

    # add noise
    m       = m + np.random.normal( 0., noise, N_pts )
    
    # clip small/negative values
    m[np.where( m < 1e-9 )] = 0.

    return T, m

# Compute smoothed mass loss rate
def smooth_mlr( T, m, dT_c ):
   
    temp_range  = dT_c
    pt_den      = len(T)/(T[-1] - T[0])
    win_len     = int(np.ceil(temp_range*pt_den)//2*2 + 1)
    mDot_sm     = -savgol_filter(m, window_length=win_len, polyorder=2, 
                                 deriv=1, delta=T[1]-T[0])
    
    return mDot_sm

# get kinetic parameters from shape parameters
def shape2arrhenius( T_p, delT, beta ):
    
    E   = R*T_p*T_p/delT
    A   = (beta/delT)*np.exp( T_p/delT )

    return A, E

# compute RMS error
def error_2( m_d, mDot_d, m_m, mDot_m ):

    # clip data to active region
    I_c = np.where( (m_d < 0.98*m_d[0])*(m_d > 1.02*m_d[-1]) )

    e_2_m = np.sqrt(np.sum( (m_d[I_c] - m_m[I_c])**2 )/len(I_c[0]))
    e_2_mDot = np.sqrt(np.sum( (mDot_d[I_c] - mDot_m[I_c])**2 )/len(I_c[0]))

    return e_2_m, e_2_mDot

def process_tga( T, m, dT_c, eps_c ):

    # get data parameters
    N       = len(T)            # number of data points
    h       = T[1] - T[0]       # temperature step size
    dI_c    = int(dT_c/h)       # number of data points corresponding to crtical temp. width
    
    # compute smoothed mass loss rate
    mDot    = smooth_mlr( T, m, dT_c )
    
    # compute averaged initial and final masses over first and last dT_c
    m_0     = np.mean( m[0:dI_c] )
    m_f     = np.mean( m[-dI_c:] )
    
    # find peaks
    i_max   = np.argmax(mDot)   # identify maximum mass loss rate
    I_p     = [i_max]           # create array of peak indices
    for i in range(dI_c, N-dI_c):
        if ( all( mDot_j <= mDot[i] for mDot_j in mDot[i-dI_c:i+dI_c] )
             and mDot[i] > eps_c*mDot[i_max] 
             and i != i_max ):
            I_p.append( i )
   
    # order and count peaks
    I_p.sort()
    N_r     = len(I_p)
    
    # get peak values
    m_p     = m[I_p]
    T_p     = T[I_p]
    mDot_p  = mDot[I_p]
    
    # compute trough masses
    m_t     = np.zeros( N_r + 1 )
    m_t[0]  = m_0
    m_t[-1] = m_f
   
    # solve for intermediate trough masses, A*m_t = b
    if N_r > 1:
        
        # build trough mass matrices
        d_low       = np.ones( N_r )/(e - 1.)
        d_mid       = 2.*np.ones( N_r + 1 )
        d_upp       = (e - 1.)*np.ones( N_r )
        A           = np.diag( d_low, -1 ) + np.diag( d_mid, 0 ) + np.diag( d_upp, 1 )
        A[0,0]      = 1.
        A[0,1]      = 0.
        A[-1,-1]    = 1.
        A[-1,-2]    = 0.
       
        # build right-hand side
        b           = np.zeros( N_r + 1 )
        b[0]        = m_0
        b[-1]       = m_f
        b[1:-1]     = (e/(e - 1.))*m_p[0:-1] + e*m_p[1:]
    
        # force values for deep troughs
        for i in range(0, N_r-1):
            
            i_dt    = np.argmin( mDot[I_p[i]:I_p[i+1]] ) + I_p[i]
            
            if mDot[i_dt] < eps_c*mDot[i_max]:
                A[i+1,:]    = np.zeros( N_r + 1)
                A[i+1,i+1]  = 1.
                b[i+1]      = m[i_dt]
    
        # solve system for trough masses
        m_t = np.linalg.solve( A, b )
        
    # compute reaction mass losses 
    dm  = m_t[0:-1] - m_t[1:]
    
    return I_p, T_p, m_p, mDot_p, m_t, dm

# point estimate of peak widths, Method 1
def point_estimate( m_p, mDot_p, m_t ):

    return (m_p - m_t[1:])/mDot_p

# approximate point estimate of peak widths, Method 2
def approx_point_estimate( mDot_p, m_t, dm ):

    return dm/(e*mDot_p)

# linear fit method for peak widths and temperatures, Method 3
def linear_fit( I_p, m_t, dm, dT_e, T, m ):

    N_r     = len( I_p )            # number of reactions
    h       = T[1] - T[0]           # temperature step size
    
    # initiate fit peak temperatures and peak widths
    T_p_f    = np.zeros( N_r )
    dT_f    = np.zeros( N_r )

    for i in range(0, N_r):
        
        # compute fit data
        dI_p    = int( 0.5*dT_e[i]/h )
        j_ar    = np.arange(I_p[i] - dI_p, I_p[i] + dI_p + 1)
        y       = T[j_ar]
        x       = np.log( -np.log( (m[j_ar] - m_t[i+1])/dm[i] ) )
        
        # fit parameters
        dT_f[i], T_p_f[i]  = np.polyfit( x, y, 1 )
    
    return T_p_f, dT_f

