#!/usr/bin/env python
"""

Main script for fitting TGA data 

"""

import sys
import numpy as np
from matplotlib import pyplot as plt

import fit_tools as ft

# fit parameters
dT_c    = 10.                   # critical temperature width, K
eps_c   = 0.05                  # critical fraction of maximum mass loss rate

# get input data
file_in = str(sys.argv[1])
file_label = file_in[:-4]
dat_raw = np.loadtxt(file_in, skiprows=1, delimiter=',')
beta    = float(sys.argv[2])    # heating rate, K/min

print("===========================================================")
print(" ")
print(" Fitting TGA data from file: ", file_in)
print("   at heating rate of ", beta, " K/min.")
print(" ")
print("-----------------------------------------------------------")

# parse input data
T_d     = dat_raw[:,0] + 273.15     # data temperatures, K
m_d     = dat_raw[:,1]/dat_raw[0,1] # normalized mass data
beta    = beta/60.                  # heating rate, K/s

# data parameters
h_d     = T_d[1] - T_d[0]       # temperature step size, K
N_d     = len( T_d )

# compute smoothed mass loss rate
mDot_d  = ft.smooth_mlr( T_d, m_d, dT_c )

# process TGA data to find quantities of interest
I_p, T_p, m_p, mDot_p, m_t, dm_m = ft.process_tga( T_d, m_d, dT_c, eps_c )

# Method 1: point estimate
dT_m1       = ft.point_estimate( m_p, mDot_p, m_t )
A_m1, E_m1  = ft.shape2arrhenius( T_p, dT_m1, beta )
K_m1        = np.vstack((A_m1, E_m1, dm_m))
m_m1        = ft.predict_tga( 'parallel', K_m1, beta, np.append(dm_m, 1.), T_d )
mDot_m1     = ft.smooth_mlr( T_d, m_m1, dT_c )

# Method 2: point estimate
dT_m2       = ft.approx_point_estimate( mDot_p, m_t, dm_m )
A_m2, E_m2  = ft.shape2arrhenius( T_p, dT_m2, beta )
K_m2        = np.vstack((A_m2, E_m2, dm_m))
m_m2        = ft.predict_tga( 'parallel', K_m2, beta, np.append(dm_m, 1.), T_d )
mDot_m2     = ft.smooth_mlr( T_d, m_m2, dT_c )

# compute L_2 errors for masses
e_m_m1, e_mDot_m1 = ft.error_2( m_d, mDot_d, m_m1, mDot_m1 )
e_m_m2, e_mDot_m2 = ft.error_2( m_d, mDot_d, m_m2, mDot_m2 )

# print results to terminal
print(" ")
print("Number of reactions = ", len(I_p))
print(" ")
print("Method 1 Results: ")
print("  L_2 Mass Error           = ", e_m_m1)
print("  L_2 Mass Loss Rate Error = ", e_mDot_m1)
print("  A (1/s)                  = ", K_m1[0,:])
print("  E (kJ/mol)               = ", K_m1[1,:])
print("  dm (-)                   = ", K_m1[2,:])
print(" ")
print("Method 2 Results: ")
print("  L_2 Mass Error           = ", e_m_m2)
print("  L_2 Mass Loss Rate Error = ", e_mDot_m2)
print("  A (1/s)                  = ", K_m2[0,:])
print("  E (kJ/mol)               = ", K_m2[1,:])
print("  dm (-)                   = ", K_m2[2,:])
print(" ")
print("===========================================================")

# convert units for plotting
mDot_d  = mDot_d*beta
mDot_m1 = mDot_m1*beta
mDot_m2 = mDot_m2*beta
T_d     = T_d - 273.15

# plotting parameters plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('lines', linewidth=1.0)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

plt.figure(1)
plt.plot(T_d, mDot_d, 'k-', label='Expt.')
plt.plot(T_d, mDot_m1, 'r--', label='Method 1')
plt.plot(T_d, mDot_m2, 'g--', label='Method 2')
plt.xlabel(r"Temperature ($^{\circ}$ C)", fontsize=22)
plt.ylabel(r"Residual Mass Loss Rate (1/s)", fontsize=22)
plt.legend(loc=1, numpoints=1, prop={'size':16})
plt.tight_layout()
plt.savefig(file_label + "_mlr_fit.pdf")
 
plt.figure(2)
plt.plot(T_d, m_d, 'k-', label='Expt.')
plt.plot(T_d, m_m1, 'r--', label='Method 1')
plt.plot(T_d, m_m2, 'g--', label='Method 2')
plt.xlabel(r"Temperature ($^{\circ}$ C)", fontsize=22)
plt.ylabel(r"Residual Mass Fraction", fontsize=22)
plt.legend(loc=1, numpoints=1, prop={'size':16})
plt.tight_layout()
plt.savefig(file_label + "_m_fit.pdf")

