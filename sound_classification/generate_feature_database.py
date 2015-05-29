__author__ = 'lgeorge'
"""
little script to offline generate features database from wav file
"""

import glob
import os
import traceback
import pandas as pd

from sound_processing.features_extraction import extract_mfcc_features_one_channel, _flatten_features_dict
from sound_processing.sig_proc import downsample_signal
from sound_processing.io_sound import load_sound

def _generate_8k_dataset_dict(glob_file_pattern='/mnt/protolab_server_8k/fold*/*.wav', nfft=1024, downsampling_freq=None):
    """

    :param glob_file_pattern:
    :param nfft:
    :param downsampling_freq: if set it's used for downsampling
    :return:
    """
    files = glob.glob(glob_file_pattern)
    assert(files!=[])

    res = []
    for num, f in enumerate(files):
        try:
            data = {}
            data['file_path'], data['file_name'] = os.path.split(f)

            signal, fs = load_sound(f)
            # using channel_1 only
            if downsampling_freq:
                signal, fs = downsample_signal(signal, origin_fs=fs, final_fs=downsampling_freq)

            data['fs'] = fs
            try:
                features = extract_mfcc_features_one_channel(signal, nfft=nfft)
                data['features'] = features
            except Exception as e:
                import IPython
                IPython.embed()

            res.append(data)

        except Exception as e:
            print("ERROR on %s" % f)
            print(traceback.format_exc())
    return res





def generate_8k_dataset(glob_file_pattern='/mnt/protolab_server_8k/fold*/*.wav', nfft=1024, downsampling_freq=None):
    dict_with_features = _generate_8k_dataset_dict(glob_file_pattern=glob_file_pattern, nfft=nfft)
    df = pd.DataFrame(dict_with_features)

    df['fold'] = df['file_path'].apply(lambda x: int(os.path.basename(x)[4:]))  # string = foldXY , so we take string[4:
    df['features'] = df['features'].apply(lambda x : _flatten_features_dict(x))
    df['expected_class'] = df['file_name'].apply(lambda x: _add_class_from_filename_8kdataset(x))
    return df



def _add_class_from_filename_8kdataset(x):
    """

    :param df:  a dataframe with columns file_name, file_path, 'fs',
    :return:
    """

    # adding class name to dataframe
    class_id_to_name = {0:"air_conditioner",
                        1:"car_horn",
                        2:"children_playing",
                        3:"dog_bark",
                        4:"drilling",
                        5:"engine_idling",
                        6:"gun_shot",
                        7:"jackhammer",
                        8:"siren",
                        9:"street_music"}
    return  class_id_to_name[int(x.split('-')[1])]




