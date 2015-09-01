__author__ = 'lgeorge'

import pytest
import os
import glob
import subprocess
import pandas
import sklearn.metrics
import numpy as np
import json
import unittest
import pprint

from test_common import _get_training_data, wget_file
from sound_classification import classification_service
from sound_classification import evaluate_classification

def _convert_min_sec_to_sec(val):
    """

    :param val:  val is a string in format 'XmYsZ' like '0m5s3' meaning at secong 5,3
    :return:
    >>> _convert_min_sec_to_sec('10m11s2')
    611.2
    """
    _min = val.split('m')[0]
    _sec = val.split('m')[1].split('s')[0]
    _dsec = val.split('s')[1]
    if len(_dsec) == 1:
        _dsec =  _dsec + '0'

    res =  int(_min) * 60 + int(_sec) + float(_dsec)/100.
    return res

def load_csv_annotation(csv_file):
    import pandas
    df_expected = pandas.read_csv(csv_file)
    df_expected['timestamp_start'] = df_expected['time begin'].apply(_convert_min_sec_to_sec)
    df_expected['timestamp_end'] = df_expected['time end'].apply(_convert_min_sec_to_sec)
    df_expected['class_expected'] = df_expected['name']
    #df_expected[['class_expected', 'timestamp_start', 'timestamp_end']]
    return df_expected


def _overlap(a_start, a_stop, b_start, b_stop):
    """
    :param a_start:
    :param a_stop:
    :param b_start:
    :param b_stop:
    :return:
    >>> _overlap(1, 5, 1.3, 2)
    True
    >>> _overlap(8, 9, 1.2, 2)
    False
    """
    # thx stack overflow
    overlap = max(0, min(a_stop, b_stop) - max(a_start, b_start))
    return (overlap >0)


def find_overlap(timestamps_1, timestamps_2):
    """

    :param timestamps_1: list of tuple (index, start, stop) with start and stop float values, stop >= start
    :param timestamps_2: list of tuple (index, start, stop) with start and stop float values, stop >= start
    :return:

    >>> find_overlap([(0, 1, 5), (1, 8, 9), (2, 11, 30)], [(0, 1.3, 2), (1, 2.3, 4), (2, 7, 8.5)])
    [(0, 0), (0, 1), (1, 2)]
    """
    associated = []
    for (index_1, a_start, a_stop) in (timestamps_1):
        for  (index_2, b_start, b_stop) in (timestamps_2):
            if _overlap(a_start, a_stop, b_start, b_stop):
                associated.append((index_1, index_2))
    return associated



def compute_tp_fp(df, df_expected):
    """
    return {'false_positives': false_positives, 'true_positives':true_positives, 'not_detected': not_detected, 'silence_not_detected': silence_not_detected}
    each elements correspond to a list of tuple, first item of tuple is equal to detection second item equal to expectation (i.e annotation)

    Warning if for same index in expected we have multiple values in predicted.. we count them all
    """
    timestamps_1 = df[['timestamp_start', 'timestamp_end']].to_records()
    timestamps_2 = df_expected[['timestamp_start', 'timestamp_end']].to_records()
    overlaps_index = find_overlap(timestamps_1, timestamps_2)

    index_expected_found = zip(*overlaps_index)[1]
    index_df_found = zip(*overlaps_index)[0]

    # for values
    overlap_values = [(df.iloc[x], df_expected.iloc[y]) for (x, y) in overlaps_index]
    false_positives = [(x, y) for (x, y) in overlap_values if x.class_predicted != y.class_expected]
    true_positives = [(x, y)  for (x, y) in overlap_values if x.class_predicted == y.class_expected]

    # TODO: if for the same song we detect multiple time the same classifier values.. we should maybee group them.. otherwise long song will get more points..
    # TODO : think about it

    # for silences.. :
    not_detected = [(None, df_expected.iloc[index]) for index in df_expected.index if index not in index_expected_found]  # silence detected (i.e nothing detected) but it should be something
    silence_not_detected = [(df.iloc[index], None) for index in df.index if index not in index_df_found]  # something detected but it should be a silence -> it's an errors

    return {'false_positives': false_positives, 'true_positives':true_positives, 'not_detected': not_detected, 'silence_not_detected': silence_not_detected}


def convert_tp_fp_to_confusion_matrix(true_positives, false_positives, not_detected, silence_not_detected):
    """
    Now we compute confusion matrix like list
    :param true_positives:
    :param false_positives:
    :param not_detected:
    :param silence_not_detected:
    :return: expected_list, predicted_list, labels  which could easily be used with confusion matrix plot
    """

    expected_list, predicted_list, labels = [], [], set()
    nothing_value = 'NOTHING'

    for (predicted, expected) in true_positives + false_positives:
        expected_list.append(expected.class_expected)
        predicted_list.append(predicted.class_predicted)

    # now we handle silences case :
    #first when a silence is detected when there is something
    for (_, expected) in not_detected:
        expected_list.append(expected.class_expected)
        predicted_list.append(nothing_value)

    # second when a silence is detected as something
    for (predicted, _) in silence_not_detected:
        expected_list.append(nothing_value)
        predicted_list.append(predicted.class_predicted)

    labels = list(set(predicted_list).union(set(expected_list)))
    return expected_list, predicted_list, labels


def generate_html5(output_html_fname, df, wav_file_url):
    from flask import Flask, render_template

    annotation_tuples = []
    for index, row in df.iterrows():
        text = []
        if 'class_expected' in row:
            text.append('expected:')
            text.append(row.class_expected)

        if 'class_predicted' in row:
            text.append('predicted:')
            text.append(row.class_predicted)

        val = {'text':str(text), 'start':row.timestamp_start, 'end':row.timestamp_end}
        annotation_tuples.append(val)


    #annotation_tuples = [{'text':'5 to 15', 'start':'5', 'end':'15'}, {'text':'text 2 to 3 BELL', 'start':'10.0', 'end':'30.0'}]
    app = Flask(__name__)
    with app.app_context():
        output_text =  render_template('index.html', annotations=annotation_tuples, wav_file_url=wav_file_url)

    with open(output_html_fname, 'w') as f:
        f.write(output_text)



class TestMultipleDetectionsDefaultDatasetWithCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls, dataset_url=None, wav_file_url=None, csv_url=None):
        cls.min_precision = 0.7
        cls.min_recall = 0.7
        cls.enable_calibration_of_score = True
        cls.dataset_url = dataset_url
        cls.dataset_path = _get_training_data(cls.dataset_url)
        cls.file_regexp = os.path.join(cls.dataset_path, '*.wav')
        cls.file_regexp_bis = os.path.join(cls.dataset_path, '*/*.wav')
        cls.files = glob.glob(cls.file_regexp) + glob.glob(cls.file_regexp_bis)
        cls.sound_classification_obj = classification_service.SoundClassification(wav_file_list=cls.files, calibrate_score=cls.enable_calibration_of_score)
        cls.test_file = "test.wav"
        if wav_file_url is None:
            wav_file_url = 'https://www.dropbox.com/s/tcem6metr3ejp6y/2015_07_13-10h38m15s101111ms_Juliette__full_test_calm.wav?dl=0'
        cls.wav_file_url = wav_file_url
        cls.test_file = wget_file(cls.wav_file_url)
        cls.test_file = os.path.abspath(cls.test_file)
        #cls.test_file = '/home/lgeorge/tests/2015_07_13-10h38m15s101111ms_Juliette__full_test_calm.wav'

        if csv_url is None:
            csv_url = 'https://www.dropbox.com/s/umohtewtn6l5275/2015_07_13-10h38m15s101111ms_Juliette__full_test_calm.csv?dl=0'
        cls.csv_url = csv_url
        cls.csv_file = wget_file(cls.csv_url)
        cls.df_expected = load_csv_annotation(cls.csv_file)

        cls.csv_file = os.path.abspath(cls.csv_file)

        cls.sound_classification_obj.learn()
        cls.res = cls.sound_classification_obj.processed_wav(cls.test_file)

        cls.df = pandas.DataFrame([rec.__dict__ for rec in cls.res])
        if cls.enable_calibration_of_score:
            #df.class_predicted[df.score <= 0.9] = 'NOTHING'
            cls.df = cls.df[cls.df.score >= 2.5] #we drop low score
            cls.df = cls.df.reset_index(drop=True)
        cls.df.to_csv('output.csv')  # just for later check if needed

        cls.labels_present_in_wavfile = set(cls.df_expected.class_expected)

        detection_dict = compute_tp_fp(cls.df, cls.df_expected)

        #print("True positive count: {}".format(len(detection_dict['true_positives'])))
        #print("False positive count: {}".format(len(detection_dict['false_positives'])))
        #print("Silence detected as something: {}".format(len(detection_dict['silence_not_detected'])))
        #print("Not detected count: {}".format(len(detection_dict['not_detected'])))

        cls.expected, cls.predicted, cls.labels = convert_tp_fp_to_confusion_matrix(detection_dict['true_positives'], detection_dict['false_positives'], detection_dict['not_detected'], detection_dict['silence_not_detected'])
        report = sklearn.metrics.classification_report(cls.expected, cls.predicted, labels=cls.labels, target_names=None, sample_weight=None, digits=2)
        matrix = sklearn.metrics.confusion_matrix(cls.expected, cls.predicted, cls.labels)
        print("Confusion Matrix")
        pprint.pprint(matrix)

        #res.savefig('confusion_mat.png')
        print(report)


        cls.precisions = sklearn.metrics.precision_score(cls.expected, cls.predicted, labels=cls.labels, average=None)
        cls.recalls = sklearn.metrics.recall_score(cls.expected, cls.predicted, labels=cls.labels, average=None)

        cls.labels_to_consider = [l for l in cls.labels if l in cls.sound_classification_obj.clf.classes_]
        cls.labels_to_ignore = [l for l in cls.labels if l not in cls.sound_classification_obj.clf.classes_]
        cls.labels_to_consider_index = [num for (num, val) in enumerate(cls.labels) if val in cls.labels_to_consider]

        generate_html5('debug_predicted.html', cls.df, "http://127.0.0.1/out.mp3")

    def test_setup(self):
        pass

    def test_precision(self):
        print("ignoring labels %s, not present in learning dataset" % str(self.labels_to_ignore))
        for index in self.labels_to_consider_index:
            if self.precisions[index] == 0 and self.labels[index] not in self.predicted:
                self.precisions[index] = 1.  # MY precision comprhension
        np.testing.assert_array_less(self.min_precision, self.precisions[self.labels_to_consider_index], "labels considered are {}, predicted are {}, expected are {}".format(self.labels_to_consider, self.predicted, self.expected))
        assert(len(self.predicted) > 0)


    def test_recall(self):
        # for recall we also ignore labels not present in the test_wav_file
        print("ignoring labels %s, not present in learning dataset" % str(self.labels_to_ignore))
        print("considering only labels present in wavfile %s" % str(self.labels_present_in_wavfile))
        labels_to_ignore = self.labels_to_ignore + [label for label in self.labels if label not in self.labels_present_in_wavfile]
        labels_to_consider = [l for l in self.labels if l not in labels_to_ignore]
        labels_to_consider_index = [num for (num, val) in enumerate(self.labels) if val in labels_to_consider]
        # we use assert array less because it provide a pecent mismatch, easier to read.. it's equivalent to checking all values above min_recall
        np.testing.assert_array_less(self.min_recall, self.recalls[labels_to_consider_index], "labels considered are {} predicted are {}, expected are {}".format(self.labels_to_consider, self.predicted, self.expected))



class TestMultipleDetectionsWithCalibrationEuropythonDatasetFull(TestMultipleDetectionsDefaultDatasetWithCalibration):
    @classmethod
    def setUpClass(cls):
        dataset_all_sound_europython = "https://www.dropbox.com/s/8t427pyszfhkfm4/dataset_aldebaran_allsounds.tar.gz?dl=0"
        super(TestMultipleDetectionsWithCalibrationEuropythonDatasetFull, cls).setUpClass(dataset_all_sound_europython)

class TestMultipleDetectionsWithCalibrationEuropythonDatasetSimpleBell(TestMultipleDetectionsDefaultDatasetWithCalibration):
    @classmethod
    def setUpClass(cls):
        simple_sound = "https://www.dropbox.com/s/8dlr28s9gby46h1/bell_test.wav?dl=0"
        simple_bell_sound_csv = 'https://www.dropbox.com/s/hvjyvmexq8gn8r0/bell.csv?dl=0'
        super(TestMultipleDetectionsWithCalibrationEuropythonDatasetSimpleBell, cls).setUpClass(wav_file_url=simple_sound, csv_url=simple_bell_sound_csv)

def main():
    csv_file = '2015_07_13-10h38m15s101111ms_Juliette__full_test_calm.csv?dl=0'
    wav_file_url = "http://127.0.0.1/out.mp3"
    df = load_csv_annotation(csv_file)
    generate_html5('debug.html', df, wav_file_url)

    import unittest
    unittest.main()

#if __name__ == "__main__":
    #main()
