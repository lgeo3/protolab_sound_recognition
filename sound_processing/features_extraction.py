__author__ = 'lgeorge'


import numpy as np
import scipy.stats

from features import mfcc, logfbank, fbank  #  to compute mel frequency we use this package ->  https://github.com/jameslyons/python_speech_features
from sklearn import preprocessing
import sklearn.feature_extraction
from collections import namedtuple

# TODO : it should works with a multichannel signal ?
Cross_validation_split = namedtuple('cross_validation_split', ['X_training', 'X_testing', 'Y_training', 'Y_testing'])

def extract_mfcc_features(signal, win_len=0.0232, win_overlap=0.5, n_mel_bands=40, n_coefs=25, fs=48000, nfft=1024):
    """
    Return feature vector for a one channel signal

    Return same features as the one defined in the paper
    Salamon, J., Jacoby, C., & Bello, J. (2014). A Dataset and Taxonomy for Urban Sound Research. ACM International Conference Onf Multimedia, (3). doi:10.1145/2647868.2655045

    :param signal: one dimension array
    :param win_len: length of window to split the signal into
    :param win_overlap: overlap over window, 1 > win_overlap >= 0
    :param n_mel_bands: numbers of mel bands to use
    :param n_coefs: number of dct coefs to return
    :param fs: signal sampling rate
    :return: a dict of features array
    """
    win_step = win_len * win_overlap # 50%
    features = {}
    res = mfcc(signal, samplerate=fs, winlen=win_len, winstep = win_step, nfilt = n_mel_bands, lowfreq=0,
               highfreq = 22050, numcep=n_coefs, nfft=nfft) ## TODO revoir nfft.. je ne suis pas certain de comprendre a quoi ca correspond pour mel dans le papier il n'en parle pas.. surtour revoir si ca fonctionne avec nfft et fs... car bon
    #print("fs {}, signal.shape {}".format(fs,signal.shape))
    #print(res.shape)
    features["minimum"] = np.min(res, axis=0)
    features["maximum"] = np.max(res, axis=0)
    features["median"] = np.median(res, axis=0)
    features["mean"] = np.mean(res, axis=0)
    features["variance"] = np.var(res, axis=0)
    features["skewness"] = scipy.stats.skew(res, axis=0)
    features["kurtosis"] = scipy.stats.kurtosis(res, axis=0)
    features["mean_first_diff"] = np.mean(np.diff(res, axis=0), axis=0)
    features["variance_first_diff"] = np.var(np.diff(res, axis=0), axis=0)
    features["mean_second_diff"] = np.mean(np.diff(res, axis=0, n=2), axis=0)
    features["var_second_diff"] = np.var(np.diff(res, axis=0, n=2), axis=0)
    return features

def  extract_mfcc_features_one_channel(signal, **kwargs):
    if len(signal.shape)>1:
        signal_ = signal[:,0]
    else:
        signal_ = signal
    return extract_mfcc_features(signal_, **kwargs)


def _flatten_features_dict(u):
    # getting features as we want
    accu = {}
    for key, val in u.iteritems():

        for num, v in enumerate(val):
            accu['_'.join([key, str(num)])] = v
    return accu


def preprocess(cross_validation_tuple, preprocess_correlation=False, preprocess_scaling=False):

    X = cross_validation_tuple.X_training
    X_test = cross_validation_tuple.X_testing
    if preprocess_scaling:
        scaler = preprocessing.StandardScaler().fit(cross_validation_tuple.X_training)
        X = scaler.transform(X)
        X_test = scaler.transform(X_test)
    if preprocess_correlation:
        from sklearn.decomposition import RandomizedPCA
        pca = RandomizedPCA(n_components=0.99, whiten=True)  # n between 0 and 1 to select number of componnents to explain 99 percents of the data
        pca.fit(X)
        print("PCA component keeping {}".format(pca.n_components))
        X = pca.transform(X)
        X_test = pca.transform(X_test)
    return Cross_validation_split(X, X_test, cross_validation_tuple.Y_training, cross_validation_tuple.Y_testing)

def get_features(signal_chunck, nfft=1024, sklearn_dict_vectorizer = None, scaler=None):
    """
    Simple feature generator for online process
    :param signal_chunck: the minimal lenght is nfft
    :param nfft: the window size for fft
    :param scaler: a sklearn preprocessing scaler fitted on training data, e.g scaler = preprocessing.StandardScaler().fit(data_X)
    :return: a feature vector
    """
    features = extract_mfcc_features_one_channel(signal_chunck, nfft=nfft)
    features = _flatten_features_dict(features) # flat
    if sklearn_dict_vectorizer is None:
        # we create a feature vectorizer
        sklearn_dict_vectorizer = sklearn.feature_extraction.DictVectorizer()
        features = sklearn_dict_vectorizer.fit_transform(features)
    else:
        ## if a sklearn dictVectorizer is provided we use it directly
        features = sklearn_dict_vectorizer.transform(features)
    features = features.toarray()

    if scaler is None:
        return features
    return scaler.transform(features)
