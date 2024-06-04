# sigh, try random stuff

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import sys

# generate log2 approx via floor-shift (piecewise linear between powers of 2)
def floorshift_log2(i, nfrac=9):
    if i==0:
        return 0
    int_part = np.int32(np.log2(i))
    remainder = (i - (2**int_part))/(2**int_part)
    frac_part = np.int32( remainder * (2**(nfrac+1)) )
    # convergent round
    frac_part = (((frac_part & 0x1)<<1) | frac_part)>>1
    return int_part + (frac_part/(2**nfrac))
    
# WOO FPGA MATH
def approx_probit(total, nsamples):
    # generate constant to 22 bits fractional
    offset = np.int32(np.round(np.log2(nsamples)*(2**22)))
    # generate approximate log2 and upscale by 2^13. Note that
    # floorshift_log2 gives the value as a float, not integer.
    approx_log2 = np.int32(floorshift_log2(total, nfrac=9)*(2**22))
    # calculate offset log2
    offset_log2 = offset - approx_log2
    # now calculate square root. We can just do this with
    # math because the FPGA square root is exact.
    return np.int32(np.sqrt(offset_log2))
    
    
def gennoise(sc=32, off=0, num=1048576):
    noise_scale = sc
    fs = 3000

    # just generate a lot of noise
    noise = np.random.normal(loc=off, scale=noise_scale, size=num)
    # whatever. random bull$#!+ filter
    hp = signal.butter(N=5, Wn=300, btype='highpass', fs=fs, output='sos')
    lp = signal.butter(N=5, Wn=1200, btype='lowpass', fs=fs, output='sos')
    
    nfilt = signal.sosfilt(hp, noise)
    nfilt = signal.sosfilt(lp, nfilt)
    return nfilt

# kinda same thing except this says how many fractional bits to shift
# we now no longer have any parameters for 'desired'
def quant_run(sc=32, off=0, cw_scale=0, cw_freq=400e6, verbose=False):
    # We apply the 'off' here to the MATH not the RANDOM
    # because we HIGH-PASS FILTER (duh)
    nfilt = gennoise(sc, 0)

    if cw_scale > 0:
        nr_sine = len(nfilt)
        fs = 3E9
        scaled_freq = cw_freq/fs
        # this always starts with the same phase, but I DON'T CARE
        cw = cw_scale*np.sin(2*np.pi*scaled_freq*np.arange(nr_sine))
        nfilt = nfilt + cw
        
    # THIS IS WRONG BUT WE'RE GOING TO SCALE AGAIN LATER TO GAIN FRACTIONAL BITS
    # THE **IDEAL** OFFSET WITH NO REAL DC IS JUST -0.5 LSBS
    nfilt = nfilt + off
    nfilt = np.round(nfilt)
    nfilt = np.int32(nfilt)
    # Generate a pure binary representation. I don't know if this is
    # necessary.
    nbinary = np.bitwise_and(nfilt, 0xFFFFFFFFFFFF)
    # Our 'desired' scale is to have the RMS at 32
    # and then we pick off 5 bits such that the RMS=4.
    # So we effectively downshift by 3 and pick off 5
    
    # so it's kindof like 3.2 format except
    # we offset it.
    # So first we need to make sure we handle over/underflow.
    # RMS at 32 means the mask we're looking at is
    # .... xxx.yy ...
    # Except the TOP bit is a SIGN bit. So to check for
    # underflow/overflow, the TOP bit (bit 7) must match
    # ALL OTHER BITS above it. So valid is
    # val & mask == 0
    # and
    # val & mask == mask
    mask = 0xFFFFFFFFFF80
    topb = 0x800000000000
    max = 0xFF
    # The logic here is the same thing we use in the FPGA.
    # The pattern detector matches (PATTERN=all zeros)
    # (P & MASK) == PATTERN  = PATTERNMATCH
    #   -- indicates a value has all of its top bits zero
    # (P & MASK) == ~PATTERN = PATTERNBMATCH
    #   -- indicates a value has all of its top bits one
    # if (PATTERNMATCH || PATTERNBMATCH) then val <= out[3 +: 5]
    # else if (out[47]) val <= 5'h1F
    # else val <= 5'h0F
        
    nsat = np.select(
        [(nbinary & mask)==0,(nbinary & mask)==mask,(nbinary&topb)==topb],
        [nbinary & 0xFF, (nbinary & 0x7F)-128, -128], 127)
    nred = nsat >> 3
    if verbose:
        fig, (ax1,ax2) = plt.subplots(2, sharex=True)
        fig.suptitle('Trigger AGC and scaling')
        # Find a goddamn index where the saturation actually happens
        idx = 0
        oob = None
        for i in nbinary:
            if (i & mask) != 0 and (i & mask) != mask:
                oob = idx
                break
            idx = idx + 1

        if oob is None:
            print("goddamnit, no OOB points")
            min = 100
            max = 300
        else:
            print(oob)
            if oob > 100:
                min = oob-100
            else:
                min = 0
            if oob < len(nfilt)-100:
                max = oob+100
            else:
                max = len(nfilt)-100
        ax1.plot(nfilt[min:max])
        ax1.plot(nsat[min:max])
        ax2.plot(nred[min:max]/4 + 0.125)
        plt.show()

    resc = nred/4 + 0.125
    if verbose:
        plt.hist(resc,32,density=True)
        plt.show()
    rms = np.sqrt(np.average(resc**2))
    # Now our goal is to look for values so we can count.
    # These look ASYMMETRIC but they are NOT because we need to
    # add an OFFSET to produce a SYMMETRIC REPRESENTATION
    # So we are checking for
    # val_neg = -9,-10,-11,-12,-13,-14,-15,-16
    # val_pos =  8,  9, 10, 11, 12, 13, 14, 15
    # val_pos is just (nred & 0x18 == 0x08)
    # val_neg is just (nred & 0x18 == 0x10)
    val_pos = np.sum((nred & 0x18)==0x08)
    val_neg = np.sum((nred & 0x18)==0x10)
    return (val_pos+val_neg, val_pos-val_neg, rms)

# ok, the BASIC IDEA of this works.
# But I need to QUANTIZE the damn thing
def run(sc=32, off=0, desired=32):
    nfilt = gennoise(sc, off)
    nfilt = np.round(nfilt)
    # we remove ~40% of the power, and since
    # vrms \propto \sqrt(power), this should be
    # \sqrt(0.6)*32 ~ 24.78
    # ends up being a little lower b/c of ripple
    #rms = np.sqrt(np.mean(nfilt**2))
    rms = round(desired)
    # ok, so now we need to think about this
    # suppose we have a 3-bit digitization.
    # as inequalities, this is 
    # 111 x > 4
    # 110 3 > x > 2
    # 101 2 > x > 2
    # 100 1 > x > 0
    # 011 0 > x > -1
    # 010 -1 > x > -2
    # 001 -2 > x > -3
    # 000 -3 > x
    # Technically the transitions here are 50/50.
    # This is generated by literally making
    # sure the distribution's symmetric on both
    # sides.
    
    # ok so what's the distribution look like here?
    gr2 = np.sum(nfilt > 2*rms)
    lt2 = np.sum(nfilt < -2.*rms)

    balance = gr2 - lt2
    total = gr2 + lt2
    return (total, balance)

#### OLD TESTING USING NON-QUANTIZED VERSION
def test_me():
    # ok let's like, attempt this I guess
    in_scale = 24
    cur_factor = 4096
    mask = 0x3FFF8
    i_value = -1/128
    p_value = 1/256
    last_err = None
    cur_err = None
    for i in range(100):
        apply_factor = (cur_factor & mask)/4096
        total, balance = run(in_scale*apply_factor, 0, 32)
        cur_err = total - 47760
        delta_err = 0
        i_term = i_value * cur_err
        if last_err is None:
            p_term = 0
            last_err = cur_err
        else:
            p_term = p_value*(cur_err - last_err)
            last_err = cur_err
            
        delta_factor = round(i_term) + round(p_term)
        new_factor = cur_factor + delta_factor
        print("step", i, "old factor", cur_factor, "total:", total, "new factor", new_factor, "scale", new_factor/4096, "apply", (new_factor & mask)/4096)
        cur_factor = new_factor
            
    in_scale = 28
    for i in range(100):
        apply_factor = (cur_factor & mask)/4096
        total, balance = run(in_scale*apply_factor, 0, 32)
        cur_err = total - 47760
        delta_err = 0
        i_term = i_value * cur_err
        if last_err is None:
            p_term = 0
            last_err = cur_err
        else:
            p_term = p_value*(cur_err - last_err)
            last_err = cur_err
            
        delta_factor = round(i_term) + round(p_term)
        new_factor = cur_factor + delta_factor
        print("step", i, "old factor", cur_factor, "total:", total, "new factor", new_factor, "scale", new_factor/4096, "apply", (new_factor & mask)/4096)
        cur_factor = new_factor

# Run AGC loop for ~100 time ticks.
# This is the correct one.
def loop_quant(in_scale,
               params,
               history,
               cw_scale=0):

    last_err = history['last_err']
    lastlast_err = history['lastlast_err']
    last_dc_err = history['last_dc_err']
    cur_factor = history['cur_factor']
    cur_offset = history['cur_offset']

    for i in range(100):
        
        apply_factor = (cur_factor & params['mask'])/4096
        total, balance, rms = quant_run(in_scale*apply_factor, cur_offset/256, cw_scale=cw_scale*apply_factor)
        # First handle the DC term, it doesn't have modes.
        if last_dc_err is not None:
            cur_dc_err = ((balance - params['dc_target'])>>params['dc_exp_avg']) + (last_dc_err - (last_dc_err >> params['dc_exp_avg']))
        else:
            cur_dc_err = balance - params['dc_target']

        if last_dc_err is None:
            dc_p_term = 0
        else:
            dc_p_term = params['dc_p']*(cur_dc_err - last_dc_err)        
        dc_i_term = params['dc_i'] * cur_dc_err
        # compute offset (this is a velocity PID: compute change in control rather than control)
        delta_offset = round(dc_i_term + dc_p_term)
        new_offset = cur_offset + delta_offset
        # and update
        last_dc_err = cur_dc_err

        # Now handle the gain term. Here we have two modes: raw and probit.
        if params['mode'] == "raw":            
            cur_err = total - params['raw_target']
            i_factor = params['raw_i']
            p_factor = params['raw_p']
            d_factor = params['raw_d']
        elif params['mode'] == "probit":
            cur_err = approx_probit(np.int32(np.round(total/2)),1024*1024) - params['probit_target']
            i_factor = params['probit_i']
            p_factor = params['probit_p']
            d_factor = params['probit_d']
        elif params['mode'] == "combined":
            scale_err = np.round((1./rms)*4096) - 4096
            scale_err = np.int32(np.round((scale_err*3./4)))
            if scale_err > params['scale_max']:
                scale_err = params['scale_max']
                print("scale maxed:", scale_err, "becomes", params['scale_max'])
            
            probit_err = approx_probit(np.int32(np.round(total/2)),1024*1024) - params['probit_target']
            # deadband the probit error
            # and use it to panic if scale error is small
            if (probit_err > 0 and probit_err < params['combined_dead_p']) or (probit_err < 0 and probit_err > params['combined_dead_n']):
                print("probit deadbanded:", probit_err," becomes 0")
                if np.abs(scale_err) < 32:
                    history['cw_warning'].append(probit_err)
                else:
                    # need to do something better here
                    # or maybe just smooth
                    history['cw_warning'].append(0)
                probit_err = 0
                cur_err = 2*scale_err
            else:
                history['cw_warning'].append(0)
                cur_err = scale_err + probit_err

            print("scale  err:", scale_err)
            print("probit err:", probit_err)
            i_factor = params['combined_i']
            p_factor = params['combined_p']
            d_factor = params['combined_d']
            
        if last_err is None:
            p_term = 0
        else:
            cur_err = (cur_err >> params['exp_avg']) + (last_err - (last_err >> params['exp_avg']))            
            p_term = p_factor*(cur_err - last_err)

        i_term = i_factor*cur_err            
        if lastlast_err is None:
            d_term = 0
        else:
            # centroid second derivative
            d_term = d_factor*(cur_err - 2*last_err + lastlast_err)
            
        # compute factor (this is a velocity PID: compute change in control rather than control)
        delta_factor = round(i_term + p_term + d_term)
        new_factor = cur_factor + delta_factor
        # minimum (prevent it from going negative)
        if new_factor < params['min_factor']:
            new_factor = params['min_factor']
        
        # and update
        lastlast_err = last_err
        last_err = cur_err
        
        # If this is our first run, we don't average.
        # Note that using the averaging kinda stinks at the beginning because
        # the response is an exponential, so it takes effectively forever
        # for the initial error to decay away. So in our testing we allow it to turn off.
        # In hardware I might be able to implement this automatically by switching to
        # exponential averaging based on a timer or something.
        history['input'].append(in_scale)
        history['factor'].append(cur_factor/4096)
        history['err'].append(cur_err)
        history['balance'].append(balance)
        history['dc_err'].append(cur_dc_err)
        history['offset'].append(cur_offset/256)
        history['rms'].append(rms)
                    
        print(i, total, cur_err, balance, cur_dc_err, rms, cur_factor, new_factor, cur_offset, new_offset, p_term, i_term, d_term)
        cur_factor = new_factor
        cur_offset = new_offset

    history['last_err'] = last_err
    history['lastlast_err'] = lastlast_err
    history['last_dc_err'] = last_dc_err    
    history['cur_factor'] = cur_factor    
    history['cur_offset'] = cur_offset

    return history
        
def test_quant(scaling_factor):
    print("Running quantized test!")
    # ok let's like, attempt this I guess
    in_scale = 32/scaling_factor
    cur_factor = 4096
    mask = 0x3FFF8
    # The I-value here is fairly aggressive and converges in just a handful of steps.

    # create param hash
    params = {}
    # Raw mode parameters
    params['mask'] = 0x3FFF8
    params['raw_i'] = -1/256
    params['raw_p'] = 0
    params['raw_d'] = 0
    params['raw_target'] = 47815 # (1+erf(-sqrt(2)))*number of samples
    # Probit mode parameters. Here POSITIVE error means POSITIVE gain increase
    # If you do the standard tuning algorithm stuff with this, it gives 0.3375/0.2025/0 for PI mode
    # For PID mode it would be 0.45/0.45/0.1125.
    params['probit_i'] = 1/4
    params['probit_p'] = 1/4
    params['probit_d'] = 0
    params['probit_target'] = 4820

    params['combined_i'] = 1/8
    params['combined_p'] = 1/8
    params['combined_d'] = 0
    params['combined_dead_p'] = 1300
    params['combined_dead_n'] = -300
    params['scale_max'] = 4300
    
    # making it zero is kinda silly
    params['min_factor'] = 8

    params['exp_avg'] = 0 # no exponential averaging
    # DC parameters
    params['dc_i'] = -1/256
    params['dc_p'] = 0
    params['dc_target'] = 0
    params['dc_exp_avg'] = 0 # no exponential averaging to begin with

    params['mode'] = 'combined'
    
    # create history hash
    history = {}
    history['cur_factor'] = 4096
    history['cur_offset'] = 0
    history['last_err'] = None
    history['lastlast_err'] = None
    history['last_dc_err'] = None
    
    time_idx = np.arange(300)*1024*1024/375e6
    history['input'] = []
    history['factor'] = []
    history['offset'] = []
    history['err'] = []
    history['balance'] = []
    history['dc_err'] = []
    history['rms'] = []
    history['cw_warning'] = []
    
    # show the beginning
    quant_run(in_scale, 0, verbose=True)
    in_scale = 32/scaling_factor
    cw_scale = 0
    print("Running loop run with", in_scale, "and CW", cw_scale)
    history = loop_quant(in_scale,
                         params,
                         history)

    # Turn on exponential averaging (alpha=1/4). Also kill the last dc err.    
    params['dc_exp_avg'] = 4
    params['exp_avg'] = 2
    history['last_dc_err'] = 0
    history['last_err'] = 0
    in_scale = 32/scaling_factor
    cw_scale = 32/scaling_factor
    print("Running loop run with", in_scale, "and CW", cw_scale)    
    history = loop_quant(in_scale,
                         params,
                         history,
                         cw_scale = cw_scale)

    in_scale = 24/scaling_factor
    cw_scale = 0
    print("Running loop run at", in_scale, "with CW", cw_scale)
    history = loop_quant(in_scale,
                         params,
                         history,
                         cw_scale = cw_scale)
    # show the end
    quant_run(in_scale*(history['cur_factor'] & params['mask'])/4096, history['cur_offset']/256, verbose=True, cw_scale=cw_scale*(history['cur_factor']&params['mask'])/4096)
        
    fig, axs = plt.subplots(7, sharex=True, layout='constrained')    
    axs[0].plot(time_idx, history['input'])
    axs[0].set_title("Input scale")
    axs[0].set_xlabel("Time (s)")
    axs[1].plot(time_idx, history['err'])
    axs[1].set_title("Measured Error")
    axs[2].plot(time_idx, history['balance'])
    axs[2].plot(time_idx, history['dc_err'])
    axs[2].set_title("DC Balance")
    axs[3].plot(time_idx, history['factor'])
    axs[3].set_title("Scale Factor")
    axs[4].plot(time_idx, history['rms'])
    axs[4].set_title("Output RMS")
    axs[5].plot(time_idx, history['offset'])
    axs[5].set_title("DC Correction")
    axs[6].plot(time_idx, history['cw_warning'])
    axs[6].set_title("CW Warning")
    plt.show()

# figure out scaling factor
scal=0
for i in range(100):
    n = gennoise(1, 0)
    scal += np.sqrt(np.average(n**2))
scal = scal/100.
print("Scaling factor is", scal)

# do ONE RUN showing the processing
total, balance, rms = quant_run(50, -0.5, True)

test_quant(scal)
