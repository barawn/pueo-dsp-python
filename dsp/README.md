# DSP tools

## iir_biquad

Implements a parallelized and pipelined version of a generic biquad
with complementary poles. Parallelization is controllable by samp_per_clock
but it always has to be 3 or more.

Still needs to be debugged a bit. Note that it will **not** produce
an "identical" output to the equivalent filter - it generates a
__transformed__ filter which has the same frequency properties, but
will not be bit-for-bit identical.