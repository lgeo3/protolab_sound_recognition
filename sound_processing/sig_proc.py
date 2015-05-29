from __future__ import division
__author__ = 'lgeorge'

import numpy as np
#from scikits.samplerate import resample

# apt-get install libsamplerate0-dev
# pip install scickits-samplerate

def downsample_signal(signal, origin_fs=1, final_fs=1):
    raise NotImplementedError
    #factor = origin_fs/float(final_fs)
    #print("factor{}".format(factor))
    #new_signal = resample(signal, factor, 'sinc_best')
    #return new_signal, final_fs