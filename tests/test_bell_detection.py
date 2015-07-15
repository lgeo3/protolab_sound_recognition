__author__ = 'lgeorge'

import os
import glob
import subprocess

from sound_classification import classification_service
from test_common import _get_training_data
import pytest

def test_classifier_simple():
    """
    just check that the service is correctly installed
    :return:
    """
    sound_classification_obj = classification_service.SoundClassification()
    assert(True)


@pytest.mark.parametrize("enable_calibration_of_score", [(False), (True)])
def test_bell_detection(enable_calibration_of_score):
    dataset_path = _get_training_data()
    file_regexp = os.path.join(dataset_path, '*.wav')
    files = glob.glob(file_regexp)
    sound_classification_obj = classification_service.SoundClassification(wav_file_list=files, calibrate_score=enable_calibration_of_score)
    sound_classification_obj.learn()
    test_file_url = "https://www.dropbox.com/s/8dlr28s9gby46h1/bell_test.wav?dl=0"
    test_file = "test_bell.wav"
    p = subprocess.Popen(['wget', test_file_url, '-O', test_file])  # using wget simpler than urllib with droppox changing urlname in http response
    p.wait()
    test_file = os.path.abspath(test_file)

    res = sound_classification_obj.processed_wav(test_file)
    ## TODO assert deskbell is better than doorbell
    assert('DeskBell' in set([x.class_predicted for x in res]))
