#!/usr/bin/env python3
"""

verification_data.py

Script to generate simulated TGA data for verification of 'tga_fit.py'

"""

import numpy as np
import fit_tools as ft
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

smp_rate= 1.            # samples per second
T_range = (300., 1200.) # maximum temperature range, K

plt.ion()

# function to generate data file from case parameters
def write_ver_case( name, A, E, dm, beta, noise ):
    
    # ...generate data
    K       = np.vstack((A, E, dm))
    T, m    = ft.generate_tga( 'parallel', K, beta, T_range, smp_rate, noise )
   
    # ...save simulated data to file
    out     = np.vstack((T-273.15, m)).T
    np.savetxt( '../Materials/Simulated_Data/'+name+'.csv', 
                out, fmt='%.4e', delimiter=',', newline='\n', 
                header='Temperature (C), Mass Fraction')

# Case 1: Single, non-charring reaction, no noise, 10 K/min

# ...parameters
name    = 'OneRxn_NoNoise_10K'
A       = 1e14          # pre-exponential, 1/s
E       = 250e3         # activation energy, kJ/mol
dm      = 1.            # change in mass
beta    = 10./60.       # heating rate, K/s
noise   = 0.            # signal noise

write_ver_case( name, A, E, dm, beta, noise )

# Case 2: Two, parallel reactions, no noise, 10 K/min
name    = 'TwoRxns_NoNoise_10K'
A       = [1e14, 1e14]      # pre-exponential, 1/s
E       = [250e3, 300e3]    # activation energy, kJ/mol
dm      = [0.5, 0.5]        # change in mass
beta    = 10./60.           # heating rate, K/s
noise   = 0.                # signal noise

write_ver_case( name, A, E, dm, beta, noise )

# Case 3: Two, parallel reactions, moderate noise, 10 K/min
name    = 'TwoRxns_ModNoise_10K'
A       = [1e14, 1e14]      # pre-exponential, 1/s
E       = [250e3, 300e3]    # activation energy, kJ/mol
dm      = [0.5, 0.5]        # change in mass
beta    = 10./60.           # heating rate, K/s
noise   = 1e-4              # signal noise

write_ver_case( name, A, E, dm, beta, noise )

# Case 4: Two, parallel reactions, heavy noise, 10 K/min
name    = 'TwoRxns_HeavyNoise_10K'
A       = [1e14, 1e14]      # pre-exponential, 1/s
E       = [250e3, 300e3]    # activation energy, kJ/mol
dm      = [0.5, 0.5]        # change in mass
beta    = 10./60.           # heating rate, K/s
noise   = 5e-4              # signal noise

write_ver_case( name, A, E, dm, beta, noise )

# Case 5: Three, parallel reactions, heavy noise, 10 K/min
name    = 'ThreeRxns_HeavyNoise_10K'
A       = [1e13, 1e12,1e12]     # pre-exponential, 1/s
E       = [200e3, 225e3, 275e3] # activation energy, kJ/mol
dm      = [0.2, 0.4, 0.1]       # change in mass
beta    = 10./60.               # heating rate, K/s
noise   = 5e-4                  # signal noise

write_ver_case( name, A, E, dm, beta, noise )

