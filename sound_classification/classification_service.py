"""
Main file for online classification service

"""
__author__ = 'lgeorge'

from sklearn import preprocessing
import sklearn
import sklearn.feature_extraction
import sklearn.svm
from collections import namedtuple

from sklearn_pandas import DataFrameMapper
import numpy as np

from sound_classification.generate_database_humavips import generate_aldebaran_dataset
from sound_processing.segmentaxis import segment_axis
from sound_processing.features_extraction import get_features
import sound_processing.io_sound

ClassificationResult = namedtuple("ClassificationResult", ["timestamp_start", "timestamp_end", "class_predicted", "confidence"])

def get_confidence_prediction(clf, val):
    """
    Compute a score for a prediction

    :param clf: the classifier, probability should be activated
    :param val: val to classify
    :return: a probability score between 0 and 1
    """
    return np.max(clf.predict_proba(val))


class SoundClassification(object):
    """
    the service object allowed to learn a specific dataset (default aldebaran sounds)
    and classify a new file
    """

    def __init__(self, wav_file_pattern=None, clf = None, confidence_threshold=0.2):
        """

        :param wav_file_pattern: , each files should be named  # TODO : replace that with a list of namedTuple (file, class) for example ?
        :param clf: default is SVC with rbf kernel
        :param confidence_threshold:
        :return:
        """
        if wav_file_pattern is None:
            wav_file_pattern = '/mnt/protolab_innov/data/sounds/dataset/*.wav'
        if clf is None:
            clf = sklearn.svm.SVC(kernel='rbf', probability=True, verbose=False)
        self.to_sklearn_features = DataFrameMapper([('features', sklearn.feature_extraction.DictVectorizer())])
        self.scaler = None  # init during learn
        self.wav_file_pattern = wav_file_pattern

        self.clf = clf
        self.confidence_threshold = confidence_threshold

    def learn(self):
        self.df = generate_aldebaran_dataset(glob_file_pattern=self.wav_file_pattern)
        self._learning_data_X = self.to_sklearn_features.fit_transform(self.df)
        self._learning_data_Y = self.df.expected_class

        # normalization
        self.scaler = preprocessing.StandardScaler().fit(self._learning_data_X)
        self._learning_data_X_scaled = self.scaler.transform(self._learning_data_X)

        self.clf.fit(self._learning_data_X_scaled, self._learning_data_Y)

    def processed_signal(self, data=None, fs=48000.):
        """

        :param data:
        :param fs:
        :return: list of ClassificationResult namedtuple
        """

        assert(np.ndarray == type(data))
        assert(len(data.shape) == 1)  # we only support one channel for now
        assert(data.size != 0)

        res = []
        block_size = min(1 * fs, data.size)
        overlap = block_size >> 1  # int(block_size / 2)
        # fs is 48000  Hz for now
        nfft = 1024

        for num, signal in enumerate(segment_axis(data, block_size, overlap=overlap, end='cut')):
            preprocessed_features = get_features(signal, nfft=nfft, scaler=self.scaler)
            confidence = get_confidence_prediction(self.clf, preprocessed_features)
            if confidence > self.confidence_threshold:
                class_predicted = self.clf.predict(preprocessed_features)[0]   # [0] : as asked by Alex we return only class in string not an np.array
                timestamp_start = num * (block_size - overlap) / float(fs)
                timestamp_end = timestamp_start + block_size / float(fs)
                new_result = ClassificationResult(timestamp_start, timestamp_end, class_predicted, confidence)
                res.append(new_result)
        return res

    def processed_wav(self, filename):
        data, fs = sound_processing.io_sound.load_sound(filename)
        if len(data.shape) > 1:
            data = data[:, 0]
        return self.processed_signal(data=data, fs=fs)


def main():
    """
    Just a short demo how to use the SoundClassification class
    """
    import time
    sound_classification_obj = SoundClassification()
    sound_classification_obj.learn()
    #test_file = "/mnt/protolab_innov/data/sounds/test_segment/2015_06_12-17h33m05s546ms_PepperAlex.wav"
    test_file = "test_data/bell_test.wav"
    rStart_time = time.time()
    res = sound_classification_obj.processed_wav(test_file)
    rDuration = time.time() - rStart_time
    print("duration of processing is {}".format(rDuration))
    print(res)
    return res

if __name__ == "__main__":
    main()

