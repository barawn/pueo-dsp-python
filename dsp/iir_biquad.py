#!/usr/bin/env python3

### N.B.: NOT TESTED YET
import numpy as np
from scipy.special import eval_chebyu
import scipy.signal as signal

# Feedback portion of a IIR pipelined at samp_per_clock speed.
# The FIR portion can just be run as a FIR.
# This version only works with filters with conjugate poles, so coefficients
# are passed as magnitude/angle!
# - ins : ndarray of input samples (typ of base shape, i.e. arr.shape = (len(arr),))
# - samp_per_clock : Number of samples per clock
# - mag : Magnitude of the pole ( mag = sqrt(a2) , has to be positive )
# - angle : Angle of the pole ( arccos(a1/(-2*mag)) = angle ) in RADIANS
# - ics : Initial conditions if provided. These are the "samp_per_clock" previous clock inputs +
#         "samp_per_clock" 2x previous clock outputs (e.g. 3 samp_per_clock arrays where
#         ics[0] = x[-1]
#         ics[1] = y[-2]
#         ics[2] = y[-1]
def iir_biquad( ins , samp_per_clock, mag, angle, ics = None ):
    if ics is None:
        # Debugging
        print("No initial conditions!")
        ics = np.zeros(samp_per_clock*3)
        ics = ics.reshape(-1,4)

    # Generate the FIR coefficients for the "f" term (constant for sample 0)
    # These go from i=0 to i=samp_per_clock-2
    f_fir = np.zeros(samp_per_clock - 2)
    for i in range(0, samp_per_clock-2):
        f_fir[i] = pow(mag, i)*eval_chebyu(i,np.cos(angle) )
    # And the same for the "g" term (constant for sample 1).
    # This is written a bit differently in the document b/c of the block representation.
    # But remember ANY FIR costs only num_tap DSPs period for each sample.
    # This is just the same as the other FIR but with 1 add'l tap
    g_fir = np.zeros(samp_per_clock - 1)
    for i in range(0, samp_per_clock-1):
        g_fir[i] = pow(mag, i)*eval_chebyu(i,np.cos(angle) )
    # Note f_fir[0]/g_fir[0] are going to both be 1

    # Debugging.
    print("Magnitude:", mag, "Angle:", angle)
    print("Coefficients of f FIR:", f_fir)
    print("Coefficients of g FIR:", g_fir)

    # Expand the inputs with the initial conditions
    newins = np.concatenate( (ics[0],ics[1],ins) )
    
    # Run the FIRs
    f = signal.lfilter( f_fir, [1], newins )
    g = signal.lfilter( g_fir, [1], newins )
    # Now we decimate f/g by 8, because we only want the 0th and 1st samples out of 8 for each.
    f = f.reshape(-1, samp_per_clock).transpose()[0]
    g = g.reshape(-1, samp_per_clock).transpose()[1]
    # n.b. f[0]/g[0] are initial conditions

    # f[2] (real sample 8) is calculated from 8, 7, 6, 5, 4, 3, 2.
    # g[2] (real sample 9) is calculated from 9, 8, 7, 6, 5, 4, 3, 2.

    # Now we need to compute the F and G functions, which *again* are just FIRs,
    # however they're cross-linked, so it's a little trickier.
    # We split it into
    # F = (fir of f) + G_coeff*g(previous clock)
    # G = (fir of g) + F_coeff*f(previous clock)
    F_fir = [ 1.0, -1*pow(mag, samp_per_clock)*eval_chebyu(samp_per_clock-2, np.cos(angle)) ]
    G_fir = [ 1.0, pow(mag, samp_per_clock)*eval_chebyu(samp_per_clock, np.cos(angle)) ]

    # Debugging
    print("Coefficients of F FIR:", F_fir)
    print("Coefficients of G FIR:", G_fir)
    
    # Filter them
    F = signal.lfilter( F_fir, [1], f )
    G = signal.lfilter( G_fir, [1], g )
    # Create a temporary of F because we cross-link them
    Ftmp = F.copy()
    # F[1:0]/G[1:0] are going to be dropped anyway.
    F[2:] += pow(mag, samp_per_clock-1)*eval_chebyu(samp_per_clock-1, np.cos(angle))*G[1:-1]
    G[2:] += pow(mag, samp_per_clock+1)*eval_chebyu(samp_per_clock-1, np.cos(angle))*Ftmp[1:-1]
    # drop the initial conditions
    F = F[2:]
    G = G[2:]
    
    # Now reshape our outputs.
    arr = ins.reshape(-1, samp_per_clock).transpose()
    # arr[0] is now every 0th sample (e.g. for samp_per_clock = 8, it's 0, 8, 16, 24, etc.)
    # arr[1] is now every 1st sample (e.g. for samp_per_clock = 8, it's 1, 9, 17, 25, etc.)
    # IIR parameters
    C = np.zeros(4)
    C[0] = pow(mag, 2*samp_per_clock)*(pow(eval_chebyu(samp_per_clock-2, np.cos(angle)), 2) -
                                       pow(eval_chebyu(samp_per_clock-1, np.cos(angle)), 2))

    C[1] = pow(mag, 2*samp_per_clock-1)*((eval_chebyu(samp_per_clock-1, np.cos(angle)))*
                                         (eval_chebyu(samp_per_clock,np.cos(angle)) -
                                          eval_chebyu(samp_per_clock-2, np.cos(angle))))

    C[2] = pow(mag, 2*samp_per_clock+1)*((eval_chebyu(samp_per_clock-1, np.cos(angle)))*
                                         (eval_chebyu(samp_per_clock-2, np.cos(angle))-
                                          eval_chebyu(samp_per_clock, np.cos(angle))))

    C[3] = pow(mag, 2*samp_per_clock)*(pow(eval_chebyu(samp_per_clock, np.cos(angle)), 2) -
                                       pow(eval_chebyu(samp_per_clock-1, np.cos(angle)), 2))

    # Debugging
    print("Update step (matrix) coefficients:", C)
    
    # Now compute the IIR.
    # INITIAL CONDITIONS STEP
    arr[0][0] =  C[0]*ics[2][0] + C[1]*ics[2][1] + F[0]
    arr[0][1] =  C[0]*ics[3][0] + C[1]*ics[3][1] + F[1]
    arr[1][0] =  C[2]*ics[2][0] + C[3]*ics[2][1] + G[0]
    arr[1][1] =  C[2]*ics[3][0] + C[3]*ics[3][1] + G[1]
    for i in range(len(arr[0][2:])):
        # THIS IS THE ONLY RECURSIVE STEP
        arr[0][i] = C[0]*arr[0][i-2] + C[1]*arr[1][i-2] + F[i]
        arr[1][i] = C[2]*arr[0][i-2] + C[3]*arr[1][i-2] + G[i]
        # THIS IS NOT RECURSIVE B/C WE DO NOT TOUCH THE SECOND INDEX
        for j in range(2, samp_per_clock):
            arr[j][i] += 2*mag*np.cos(angle)*arr[j-1][i] - pow(mag, 2)*arr[j-2][i]
    # and this flattens everything
    return arr.reshape(-1)
    
        
    
        
        
