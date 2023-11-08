# Processing scripts

These are a bunch of processing scripts used to evaluate how (idealized)
filters will behave.

* compare_analytic.py : This is looking at how a simplistic Hilbert
  transform works to generate an analytic signal to compare "true"
  magnitude with a simple square and sum. Not quite perfect yet
  because the impulse used is just a delta function passed through
  the Shannon-Whitaker LP, which means it retains DC, which doesn't
  pass through the Hilbert transform.
  