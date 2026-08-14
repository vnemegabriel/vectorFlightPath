"""Scalar and velocity fields.

Design invariant: this layer is agnostic to what is being transported. `mean.py`
and `variance.py` take a SourceSpec and a diffusivity and return a scalar field.
Metofluthrin from a controlled-release device is a second SourceSpec at the
device location with a different emission rate -- it needs no new numerics here.
Keep it that way.
"""
