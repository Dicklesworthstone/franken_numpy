narrow-int argmax/argmin along axis 1.4-2.5x -> parity (delegate to numpy SIMD); wide ints already win (int32 ax0 0.06x). 0 parity fails across dtypes/shapes/axes.
