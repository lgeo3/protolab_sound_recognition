"""
Main file for online classification service

"""
__author__ = 'lgeorge'

from collections import namedtuple
from sklearn import preprocessing
import glob
import sklearn
import sklearn.base
import sklearn.feature_extraction
import sklearn.svm
import pandas as pd
from sklearn.cross_validation import StratifiedKFold

from sklearn_pandas import DataFrameMapper
import numpy as np

from sound_classification.generate_database_humavips import generate_aldebaran_dataset
from sound_processing.segmentaxis import segment_axis
from sound_processing.features_extraction import get_features
from sound_classification.confidence_scaling_based_on_confusion import get_threshold_cum_precision
import sound_processing.io_sound

ClassificationResult = namedtuple("ClassificationResult",
                                  ["timestamp_start", "timestamp_end", "class_predicted", "confidence", "score"])


def get_confidence_prediction(clf, val):
    """
    Compute a score for a prediction

    :param clf: the classifier, probability should be activated
    :param val: val to classify
    :return: a probability score between 0 and 1
    """
    return np.max(clf.predict_proba(val))


class SoundClassificationException(Exception):
    pass


class SoundClassification(object):
    """
    the service object allowed to learn a specific dataset (default aldebaran sounds)
    and classify a new file
    """

    def __init__(self, wav_file_list=None, clf=None, confidence_threshold=0.2, window_block_learning=None,
                 calibrate_score=True):
        """

        :param wav_file_list: , each files should be named  # TODO : replace that with a list of namedTuple (file, class) for example ?
        :param clf: default is SVC with rbf kernel
        :param confidence_threshold:
        :return:
        """
        if wav_file_list is None:
            wav_file_list = glob.glob('/mnt/protolab_innov/data/sounds/dataset/*.wav')
        if clf is None:
            # TODO : try with linearSVC .. and one vs all
            clf = sklearn.svm.SVC(kernel='rbf', probability=True, verbose=False)
        print("CLF is %s" % clf)
        self.to_sklearn_features = DataFrameMapper([('features', sklearn.feature_extraction.DictVectorizer())])
        self.scaler = None  # init during learn
        self.wav_file_list = wav_file_list
        self.nfft = 1024
        self.fs = 48000.  # for now we force it .. TODO

        self.clf = clf
        self.confidence_threshold = confidence_threshold
        self.window_block_learning = window_block_learning

        self.calibrate_score = calibrate_score
        # initialized during learn :
        self.score_normalized_coefs = {}  # coefs to convert confidence into "normalized" score
        self.min_expected_cum_precision = 0.99
        self.confidence_thresholds = {}  # threshold of score for each classifier in order to have `min_expected_cum_precision` for this class

        self.df = None

    def learn(self):
        # TODO: ne pas utiliser une fonction annexe.. mais pouvoir comprendre rapidement
        self.df = generate_aldebaran_dataset(self.wav_file_list, nfft=self.nfft,
                                             window_block=self.window_block_learning)
        self._learning_data_X = self.to_sklearn_features.fit_transform(self.df)
        self._learning_data_Y = self.df.expected_class

        # normalization
        self.scaler = preprocessing.StandardScaler().fit(self._learning_data_X)
        self._learning_data_X_scaled = self.scaler.transform(self._learning_data_X)

        self.clf.fit(self._learning_data_X_scaled, self._learning_data_Y)

        if self.calibrate_score:
            self._learn_calibration()
            print("confidence threshold are %s" % self.confidence_thresholds)
            print("confidence coefficients are %s" % self.score_normalized_coefs)

    def _learn_calibration(self):
        """
        calibrate score/threshold in order to provide a high precision to the user

        Warning calibration is done on same dataset that the learning

        we use using a 3-fold scheme
        :return:
        """
        # computing threshold scores
        n_folds = 3
        stratified_fold = StratifiedKFold(self.df.expected_class,
                                          n_folds)  # we use only 3 fold.. as we have only 16 values on some data

        expected = []
        predicted = []
        filenames = []
        fold_num = 0
        for train_set, test_set in stratified_fold:
            train_files = self.df.iloc[train_set].full_filename
            cloned_clf = sklearn.base.clone(self.clf)
            # we build a clone classifier service.. to do the learning on a fold..
            new_classifier_service = self.__class__(wav_file_list=train_files.tolist(), clf=cloned_clf, calibrate_score=False,
                                                    window_block_learning=self.window_block_learning)
            new_classifier_service.learn()
            for index in test_set:
                val = self.df.iloc[index]
                try:
                    prediction = new_classifier_service.processed_wav(val.full_filename)
                    expected.extend([val.expected_class] * len(prediction))
                    predicted.extend(prediction)
                    # we append the num of fold to filename to have easy difference after that.
                    filenames.extend(['_'.join([val.file_name, '_fold%s' % fold_num])] * len(prediction))
                except SoundClassificationException as e:
                    print("Exception {} detected on {}".format(e, val.full_filename))
            fold_num += 1

        predicted_class = [x.class_predicted for x in predicted]
        predicted_confidence = [x.confidence for x in predicted]
        df = pd.DataFrame(zip(expected, predicted_class, predicted_confidence), columns=['class_expected', 'class_predicted', 'confidence'])
        for class_name in set(self.df.expected_class):
            self.confidence_thresholds[class_name] = get_threshold_cum_precision(df,
                                                                                 true_positive_class=class_name,
                                                                                 min_expected_cum_precision=self.min_expected_cum_precision)

        # computing coeficient to `normalized` based on threshold scores per class
        self.score_normalized_coefs = {predicted_class: 1. if val == 0 else 0.9 / float(val) for predicted_class, val in
                                       self.confidence_thresholds.iteritems()}

    def post_processed_score(self, confidence=None, class_predicted=None):
        score = -1  # -1 => not used
        if self.score_normalized_coefs:  # dict is not empty
            if class_predicted in self.score_normalized_coefs:
                score = confidence * self.score_normalized_coefs[class_predicted]
        return score

    def processed_signal(self, data=None, fs=48000., window_block=1.0):
        """
        :param data:
        :param fs:
        :param window_block: duration of window block to use, default : 1.0 second, if None, the full signal is used as
        one big window
        :return: list of ClassificationResult namedtuple
        """

        assert (np.ndarray == type(data))
        assert (len(data.shape) == 1)  # we only support one channel for now
        assert (data.size != 0)

        res = []
        if window_block is None:
            block_size = data.size
        else:
            block_size = min(window_block * fs, data.size)
        overlap = int(block_size) >> 1  # int(block_size / 2)

        for num, signal in enumerate(segment_axis(data, block_size, overlap=overlap, end='cut')):
            preprocessed_features = get_features(signal, nfft=self.nfft, scaler=self.scaler)
            confidence = get_confidence_prediction(self.clf, preprocessed_features)
            # if confidence > self.confidence_threshold:
            class_predicted = self.clf.predict(preprocessed_features)[
                0]  # [0] : as asked by Alex we return only class in string not an np.array
            timestamp_start = num * (block_size - overlap) / float(fs)
            # print("timestamp_start is %s" % timestamp_start)
            timestamp_end = timestamp_start + block_size / float(fs)
            score = self.post_processed_score(confidence=confidence, class_predicted=class_predicted)
            new_result = ClassificationResult(timestamp_start, timestamp_end, class_predicted, confidence, score)
            res.append(new_result)
        return res

    def processed_wav(self, filename, window_block=1.0, ignore_fs=False):
        data, fs = sound_processing.io_sound.load_sound(filename)
        if not ignore_fs and fs != self.fs:
            raise (SoundClassificationException('fs (%s) != self.fs (%s)' % (fs, self.fs)))
        if len(data.shape) > 1:
            data = data[:, 0]
        return self.processed_signal(data=data, fs=fs, window_block=window_block)


def main():
    """
    Just a short demo how to use the SoundClassification class
    """
    import time

    sound_classification_obj = SoundClassification()
    sound_classification_obj.learn()
    # test_file = "/mnt/protolab_innov/data/sounds/test_segment/2015_06_12-17h33m05s546ms_PepperAlex.wav"
    test_file = "test_data/bell_test.wav"
    start_time = time.time()
    res = sound_classification_obj.processed_wav(test_file)
    duration = time.time() - start_time
    print("duration of processing is {}".format(duration))
    print(res)
    return res


if __name__ == "__main__":
    main()
