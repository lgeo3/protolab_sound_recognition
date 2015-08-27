from test_common import wget_file, _get_training_data
from sound_classification import classification_service
from sound_processing import io_sound

import glob
import os
import unittest
import time

class TestSpeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = 'https://www.dropbox.com/s/tcem6metr3ejp6y/2015_07_13-10h38m15s101111ms_Juliette__full_test_calm.wav?dl=0'
        cls.filename = 'test_huge_sound.wav'
        cls.wav_file = wget_file(cls.url)
        cls.dataset_path = _get_training_data()
        cls.file_regexp = os.path.join(cls.dataset_path, '*.wav')
        cls.file_regexp_bis = os.path.join(cls.dataset_path, '*/*.wav')
        cls.files = glob.glob(cls.file_regexp) + glob.glob(cls.file_regexp_bis)
        cls.sound_classification_obj = classification_service.SoundClassification(wav_file_list=cls.files, calibrate_score=True)
        cls.sound_classification_obj.learn()
        cls.wav_duration = io_sound.duration(cls.wav_file)


    def test_speed(self, max_duration_factor = 8.):
        start_time = time.time()
        self.res = self.sound_classification_obj.processed_wav(self.wav_file)
        duration = time.time() - start_time
        max_duration = self.wav_duration / max_duration_factor
        assert(duration < max_duration)

