__author__ = 'lgeorge'

import numpy as np
from scikits.audiolab import Sndfile

def load_sound(filename):
    """
    load a sound file and return a numpy array

    INFO: The values are normalized between -1 and 1
    :param filename:
    :return: numpy array with (sound_lenght, channels) shape
    """
    f = Sndfile(filename, 'r')
    data = f.read_frames(f.nframes, dtype=np.float64)
    return data, f.samplerate

