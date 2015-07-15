__author__ = 'lgeorge'

import pytest
import os
import glob
import subprocess
from test_common import _get_training_data
from sound_classification import classification_service

@pytest.mark.parametrize("enable_calibration_of_score", [(False), (True)])
def test_multiple_detection(wav_file_url, csv_url, min_true_positive=1, max_false_positive=0, dataset_url=None, enable_calibration_of_score=False):
    """
    Success if the number of true positive detected on the file is above a threshold, and number of false positive under a threshold

    :param wav_filename: url or filename
    :param csv_filename: url or filename
    :param min_true_positive: int
    :param max_false_positive: int
    :return:
    """
    dataset_path = _get_training_data(dataset_url)
    file_regexp = os.path.join(dataset_path, '*.wav')
    files = glob.glob(file_regexp)
    sound_classification_obj = classification_service.SoundClassification(wav_file_list=files, calibrate_score=enable_calibration_of_score)
    sound_classification_obj.learn()

    test_file = "test.wav"
    p = subprocess.Popen(['wget', wav_file_url, '-O', test_file])  # using wget simpler than urllib with droppox changing urlname in http response
    p.wait()
    test_file = os.path.abspath(test_file)

    csv_file = "test.csv"
    p = subprocess.Popen(['wget', csv_url, '-O', csv_file])  # using wget simpler than urllib with droppox changing urlname in http response
    p.wait()
    csv_file = os.path.abspath(csv_file)



    res = sound_classification_obj.processed_wav(test_file)
    ## TODO assert deskbell is better than doorbell
    assert('DeskBell' in set([x.class_predicted for x in res]))
