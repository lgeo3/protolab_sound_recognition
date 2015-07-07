__author__ = 'lgeorge'

import os
import subprocess

from sound_classification import classification_service

def test_classifier_simple():
    """
    just check that the service is correctly installed
    :return:
    """
    sound_classification_obj = classification_service.SoundClassification()
    assert(True)

def _get_training_data():
    """
    Download training data from public dropbox
    :return: path of dataset
    """
    dataset_url = "https://www.dropbox.com/s/ekldjq8o1wfhcq1/dataset_aldebaran_6sounds.tar.gz?dl=0"
    dataset_filename = os.path.join(os.path.abspath('.'), 'dataset_dl.tar.gz')
    if not(os.path.isfile(dataset_filename)):
        p = subprocess.Popen(['wget', dataset_url, '-O', dataset_filename])  # using wget simpler than urllib with droppox changing urlname in http response
        p.wait()
    dataset_path = 'dataset_learning'
    p = subprocess.Popen(['mkdir', '-p', dataset_path])
    p.wait()
    command = ['tar', '-xvzf', dataset_filename, '-C', dataset_path, '--strip-components=1']
    proc = subprocess.Popen(command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE)
    proc.wait()
    return os.path.abspath(dataset_path)

def test_bell_detection():
    dataset_path = _get_training_data()
    file_regexp = os.path.join(dataset_path, '*.wav')
    sound_classification_obj = classification_service.SoundClassification(wav_file_list=file_regexp)
    sound_classification_obj.learn()
    test_file_url = "https://www.dropbox.com/s/8dlr28s9gby46h1/bell_test.wav?dl=0"
    test_file = "test_bell.wav"
    p = subprocess.Popen(['wget', test_file_url, '-O', test_file])  # using wget simpler than urllib with droppox changing urlname in http response
    p.wait()
    test_file = os.path.abspath(test_file)

    res = sound_classification_obj.processed_wav(test_file)
    ## TODO assert deskbell is better than doorbell
    assert('DeskBell' in set([x.class_predicted for x in res]))
