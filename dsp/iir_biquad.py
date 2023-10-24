#!/usr/bin/env python3

### N.B.: NOT TESTED YET
import numpy as np
from scipy.special import eval_chebyu
import scipy.signal as signal

def test():
    orig = np.zeros(1024)
    orig[512] = 1000
    b, a = signal.iirnotch(350, 5, 3000)
    pole = signal.tf2zpk(b,a)[1][0]
    mag=np.abs(pole)
    angle=np.angle(pole)
    return iir_biquad(orig, 8, mag, angle)

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
#         If your steady-state signal is just noise, these are just constants you can calculate.
#         (it... might be zero, not sure).
#         If your steady-state signal has CW, the initial conditions depend on the phase of the CW
#         you put in. (Also, technically only samples 0/1 of y[-2] and y[-1] need to be filled in).
#
# NOTE NOTE NOTE: The resulting output will not be *identical* to an equivalent filter due to
# numerical effects, however its stability *improves* compared to the original filter. In testing
# fractional differences on the order of 10^-12 to 10^-15 were observed. This is most likely due
# to floating-point limitations, since the calculation involves constants raised on the order of
# a power of samp_per_clock*2.
#
def iir_biquad( ins , samp_per_clock, mag, angle, ics = None ):
    if ics is None:
        # Debugging
        print("No initial conditions!")
        ics = np.zeros(samp_per_clock*3)
        ics = ics.reshape(3,-1)

    # Generate the FIR coefficients for the "f" term (constant for sample 0)
    # These go from i=0 to i=samp_per_clock-2
    f_fir = [0]*(samp_per_clock-2)
    for i in range(0, samp_per_clock-2):
        f_fir[i] = pow(mag, i)*eval_chebyu(i,np.cos(angle) )
    # And the same for the "g" term (constant for sample 1).
    # This is written a bit differently in the document b/c of the block representation.
    # But remember ANY FIR costs only num_tap DSPs period for each sample.
    # This is just the same as the other FIR but with 1 add'l tap
    g_fir = [0]*(samp_per_clock-1)
    for i in range(0, samp_per_clock-1):
        g_fir[i] = pow(mag, i)*eval_chebyu(i,np.cos(angle) )
    # Note f_fir[0]/g_fir[0] are going to both be 1

    # Debugging.
    print("Magnitude:", mag, "Angle:", angle)
    print("f/g FIRs are calculated only for sample 0 and sample 1 respectively.")
    print("f FIR:", f_fir)
    print("g FIR:", g_fir)

    # Expand the inputs with the initial conditions
    newins = np.concatenate( (ics[0],ins) )
    
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

    # Crosslink coefficients. 
    Coeff_g_in_F = pow(mag, samp_per_clock-1)*eval_chebyu(samp_per_clock-1, np.cos(angle))
    Coeff_f_in_G = pow(mag, samp_per_clock+1)*eval_chebyu(samp_per_clock-1, np.cos(angle))
    
    # Debugging
    print("F/G FIRs operate on f/g inputs respectively")
    print("F FIR:", F_fir, "+g*", Coeff_g_in_F)
    print("G FIR:", G_fir, "-f*", Coeff_f_in_G)
    print()
    print("As full FIRs calculated only for sample 0 and 1 respectively:")
    print("F = f + (fz^-", samp_per_clock, ")+",Coeff_g_in_F,"*(gz^-", samp_per_clock-1, ")",sep='')
    print("G = g + (gz^-", samp_per_clock, ")-",Coeff_f_in_G,"*(fz^-", samp_per_clock+1, ")",sep='')
        
    # Filter them
    F = signal.lfilter( F_fir, [1], f )
    G = signal.lfilter( G_fir, [1], g )

    # Now we need to feed the f/g paths into the opposite filter
    # F[0]/G[0] are going to be dropped anyway.
    F[1:] += Coeff_g_in_F*g[0:-1]
    G[1:] -= Coeff_f_in_G*f[0:-1]    

    # drop the initial conditions
    F = F[1:]
    G = G[1:]
    
    # Now reshape our outputs.
    arr = ins.reshape(-1, samp_per_clock).transpose()
    # arr[0] is now every 0th sample (e.g. for samp_per_clock = 8, it's 0, 8, 16, 24, etc.)
    # arr[1] is now every 1st sample (e.g. for samp_per_clock = 8, it's 1, 9, 17, 25, etc.)

    # IIR parameters. See the 'update step' in paper.
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
    print("As an IIR:")
    print("y[0] =", C[1], "*z^-", samp_per_clock*2-1," + ", C[0], "*z^-", samp_per_clock*2,
          "+F[0]",sep='')
    print("y[1] =", C[3], "*z^-", samp_per_clock*2," + ", C[2], "*z^-", samp_per_clock*2+1,
          "+G[1]",sep='')
    # Now compute the IIR.
    # INITIAL CONDITIONS STEP
    y0_0 =  C[0]*ics[1][0] + C[1]*ics[1][1] + F[0]
    y1_0 =  C[2]*ics[1][0] + C[3]*ics[1][1] + G[0]
    y0_1 =  C[0]*ics[2][0] + C[1]*ics[2][1] + F[1]
    y1_1 =  C[2]*ics[2][0] + C[3]*ics[2][1] + G[1]
    for i in range(len(arr[0])):
        if i == 0:
            # Compute from initial conditions.
            arr[0][i] = C[0]*y0_0 + C[1]*y1_0 + F[i]
            arr[1][i] = C[2]*y0_0 + C[3]*y1_0 + G[i]
        elif i==1:
            # Compute from initial conditions
            arr[0][i] = C[0]*y0_1 + C[1]*y1_1 + F[i]
            arr[1][i] = C[2]*y0_1 + C[3]*y1_1 + G[i]
        else:
            # THIS IS THE ONLY RECURSIVE STEP
            arr[0][i] = C[0]*arr[0][i-2] + C[1]*arr[1][i-2] + F[i]
            arr[1][i] = C[2]*arr[0][i-2] + C[3]*arr[1][i-2] + G[i]

        # THIS IS NOT RECURSIVE B/C WE DO NOT TOUCH THE SECOND INDEX
        for j in range(2, samp_per_clock):
            arr[j][i] += 2*mag*np.cos(angle)*arr[j-1][i] - pow(mag, 2)*arr[j-2][i]

    # now transpose arr and flatten
    return arr.transpose().reshape(-1)
