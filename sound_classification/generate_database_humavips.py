__author__ = 'lgeorge'

import glob
import os
import traceback
import pandas as pd

from sound_processing.features_extraction import extract_mfcc_features_one_channel, _flatten_features_dict
from sound_processing.sig_proc import downsample_signal
from sound_processing.io_sound import load_sound

def _generate_humavips_dataset(glob_file_pattern="/mnt/protolab_server/media/sounds/datasets/NAR_dataset/*/*.wav", nfft=1024):
    files = glob.glob(glob_file_pattern)
    assert(files!=[])

    res = []
    for num, f in enumerate(files):
        try:
            data = {}
            data['file_path'], data['file_name'] = os.path.split(f)
            data['expected_class'] = os.path.split(data['file_path'])[-1]
            signal, fs = load_sound(f)
            data['features'] = extract_mfcc_features_one_channel(signal, nfft=nfft)
            res.append(data)
        except Exception as e:
            print("ERROR on %s" % f)
            print(traceback.format_exc())

    return res


def _generate_aldebaran_dataset(files, nfft=1024, expected_fs=48000, window_block=None):
    assert(files!=[])


    res = []
    for num, f in enumerate(files):
        try:
            data = {}
            data['file_path'], data['file_name'] = os.path.split(f)
            data['expected_class'] = data['file_name'].split('-')[0]
            signal, fs = load_sound(f)
            if fs!=expected_fs:
                print("warning file %s, wrong fs %s, using it.. please remove the file if you don't want" % (f, fs))
                #continue

            for frame_feature in extract_mfcc_features_one_channel(signal, nfft=nfft, window_block=window_block):
                data['features'] = frame_feature
                res.append(data)
        except Exception as e:
            print("ERROR on %s" % f)
            print(traceback.format_exc())

    return res

def generate_humavips_dataset(glob_file_pattern="/mnt/protolab_server/media/sounds/datasets/NAR_dataset/*/*.wav"):
    dict_with_features = _generate_humavips_dataset(glob_file_pattern=glob_file_pattern)
    df = pd.DataFrame(dict_with_features)
    df['features'] = df['features'].apply(lambda x : _flatten_features_dict(x))
    return df

def generate_aldebaran_dataset(files, nfft=1024, window_block=None):
    dict_with_features = _generate_aldebaran_dataset(files, window_block=window_block)
    df = pd.DataFrame(dict_with_features)
    df['features'] = df['features'].apply(lambda x : _flatten_features_dict(x))
    return df


def main():
    df = generate_aldebaran_dataset()
    store = pd.HDFStore('test_database_aldebaran_features_1024_48000Hz.h5', 'w')
    store['df'] = df
    store.close()


    df = generate_humavips_dataset()
    store = pd.HDFStore('test_database_humavips_features_1024_48000Hz.h5', 'w')
    store['df'] = df
    store.close()


if __name__ == "__main__":
    main()

