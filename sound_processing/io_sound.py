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

def duration(filename):
    """
    return duration of wav file in second
    :param filename:
    :return:
    """
    f = Sndfile(filename, 'r')
    return f.nframes / float(f.samplerate)

def wav_to_array(wav_fname):
    """
    convert wav fname to numpy 1d array (only first channel is kept)
    """
    data, fs = load_sound(wav_fname)
    if len(data.shape) > 1:
        data = data[:, 0]
    return data, fs

# split one big wav in segmented axis signal
# TODO
