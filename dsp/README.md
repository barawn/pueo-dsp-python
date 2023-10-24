# DSP tools

## iir_biquad

Implements a parallelized and pipelined version of a generic biquad
with complementary poles. Parallelization is controllable by samp_per_clock
but it always has to be 3 or more. Gives results with fractional error of
10^-12 to 10^-15 (due to rounding).

Outputs all coefficients needed to implement in an FPGA in debugging outputs.

Note that it only does the denominator of a biquad, the numerator can
be done by lfilter.