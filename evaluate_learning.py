__author__ = 'lgeorge'

import sklearn.ensemble
import sklearn.neighbors
import sklearn.dummy
import sklearn.svm
import sklearn.feature_extraction
from collections import namedtuple
from sklearn import cross_validation

from sklearn_pandas import DataFrameMapper
import numpy as np
import pylab
import pandas as pd

import sound_classification.confusion_matrix
from sound_processing.features_extraction import preprocess


classification_prediction = {}
clfs = []
clfs_name = ['random_forest'] #, 'nearest neighbors']
clfs.append(sklearn.ensemble.RandomForestClassifier(n_estimators=500))

#clfs.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=5))
accuracy = {}
Cross_validation_split = namedtuple('cross_validation_split', ['X_training', 'X_testing', 'Y_training', 'Y_testing'])

def compute_cross_validation_fold(df, fold):
    """
    Compute testing/training data as in the paper

    :param df: a dataframe containing 'fold', 'expected_class' and 'features'
    :return:
    """
    # convert dataframe with a features entry that contains a dict of features array into a format useable by sklearn

    to_sklearn_features = DataFrameMapper([('features', sklearn.feature_extraction.DictVectorizer())])

    test_mask = (df.fold == fold)
    training_mask = (df.fold != fold)

    training_data_X = to_sklearn_features.fit_transform(df[training_mask])
    training_data_Y = df.expected_class[training_mask]

    testing_data_X = to_sklearn_features.fit_transform(df[test_mask])
    testing_data_Y = df.expected_class[test_mask]


    return Cross_validation_split(X_training=training_data_X, X_testing=testing_data_X, Y_training=training_data_Y, Y_testing=testing_data_Y)

def predict_cross_validation(clf, cross_validatation_tuple=None):
    Classification_results = namedtuple("Classification_results", ["predicted_class", "expected_class", "classifier_name"] )

    clf.fit(cross_validatation_tuple.X_training, cross_validatation_tuple.Y_training)
    prediction = clf.predict(cross_validatation_tuple.X_testing)

    name = [clf.__module__]
    try:
        name.append(clf.kernel)
    except AttributeError:
        print('attribute error')

    return Classification_results(prediction, cross_validatation_tuple.Y_testing, name)


def compute_score(classification_results):
    """

    :param classification_results: a namedtuple with .predicted_class and expected_class vectors
    :param cross_validatation_tuple:
    :return:
    """
    res = classification_results
    Score = namedtuple("Score", ["accuracy"])

    accuracy = np.sum( res.predicted_class == res.expected_class) / float(res.expected_class.size)

    return Score(accuracy)

def generate_score(clf, cross_valid_data=None, fold=0):
    entry = {'classifier_name':'', 'accuracy':-1, 'fold':-1}
    classification_results = predict_cross_validation(clf, cross_valid_data)
    score = compute_score(classification_results)
    name = [clf.__module__]
    try:
        name.append(clf.kernel)
    except AttributeError:
        print('attribute error')
    entry['classifier_name'] = '_'.join(name)
    entry['accuracy'] = score.accuracy
    entry['fold'] = fold
    print(entry)
    return entry, classification_results


def generate_res_as_in_paper(df, list_of_classifiers, preprocess_scaling=True, preprocess_correlation=False):
    res = []
    for fold in set(df.fold):
        cross_valid_data = compute_cross_validation_fold(df, fold)
        cross_valid_data = preprocess(cross_valid_data, preprocess_scaling=preprocess_scaling, preprocess_correlation=preprocess_correlation)


        for clf in list_of_classifiers:
            res.append(generate_score(clf, cross_valid_data)[0])
    return res


def compute_cross_correlation_score(df, clfs, preprocess_scaling=True, nFold=10):
    """
    return an iterator with cross validation data
    :param df:
    :param clfs:
    :param preprocess_scaling:
    :param nFold:
    :return:
    """

    to_sklearn_features = DataFrameMapper([('features', sklearn.feature_extraction.DictVectorizer())])

    data_X = to_sklearn_features.fit_transform(df)
    data_Y = df.expected_class

    skf = cross_validation.StratifiedKFold(data_Y, n_folds=nFold)
    classification_results = []
    scores = []
    for num, (train_index, test_index) in enumerate(skf):
        X_train, X_test = data_X[train_index], data_X[test_index]
        Y_train, Y_test = data_Y[train_index], data_Y[test_index]
        print("Len train{}, Len test{}".format(Y_train.size, Y_test.size))
        cross_valid_data = Cross_validation_split(X_train, X_test, Y_train, Y_test)
        cross_valid_data = preprocess(cross_valid_data, preprocess_scaling=preprocess_scaling, preprocess_correlation=False)

        for clf in clfs:
            score, classification = generate_score(clf, cross_valid_data, fold=num)
            scores.append(score)
            classification_results.append(classification)
    return scores, classification_results



def plot_res_paper_fold(df):
    """

    :param df:  contain field classifier_name, accuarcy, and fold
    :return:
    """
    for g, v in df.groupby(df.classifier_name):
        pylab.plot(v['fold'], v['accuracy'], label=g, marker='^')
        print v

    pylab.gca().invert_xaxis()
    pylab.ylabel('Classification accuracy')
    pylab.xlabel('Fold (cross validation fold for test)')
    pylab.gca().yaxis.set_ticks(np.arange(0, 1, 0.1))
    pylab.ylim((0,1))
    pylab.legend()
    pylab.show()
    return

def plot_res_paper_fold(df):
    """

    :param df:  contain field classifier_name, accuarcy, and fold
    :return:
    """
    for g, v in df.groupby(df.classifier_name):
        pylab.plot(v['fold'], v['accuracy'], label=g, marker='^')
        print v

    pylab.gca().invert_xaxis()
    pylab.ylabel('Classification accuracy')
    pylab.xlabel('Fold (cross validation fold for test)')
    pylab.gca().yaxis.set_ticks(np.arange(0, 1, 0.1))
    pylab.ylim((0,1))
    pylab.legend()
    pylab.show()
    return

def plot_res_paper(df):
    """

    :param df:  contain field classifier_name, accuarcy, and fold
    :return:
    """
    ticks = []
    i = 0
    data_to_plot = []
    for g, v in df.groupby(df.classifier_name):
        data_to_plot.append(v['accuracy'].values)
        ticks.append(g)
        print v
    pylab.boxplot(data_to_plot)
    pylab.xticks(range(1, 1+ len(data_to_plot)), ticks)



    pylab.gca().invert_xaxis()
    pylab.ylabel('Classification accuracy')
    pylab.xlabel('Fold (cross validation fold for test)')
    pylab.gca().yaxis.set_ticks(np.arange(0, 1, 0.1))
    pylab.ylim((0,1))
    pylab.legend()
    pylab.show()
    return



def evaluate_database_8k():
    database_fname = 'database_8k_mmc_features_nfft_1024_fs_44100.h5'
    hdf = pd.HDFStore(database_fname)
    try:
        df_to_plot = hdf['res']
        plot_res_paper(df_to_plot)
        pylab.show()
    except Exception as e:
        print("error {}".format(e))

    df = hdf['df']
    clfs = []
    clfs.append(sklearn.ensemble.RandomForestClassifier(n_estimators=500, random_state=0))
    clfs.append(sklearn.svm.SVC(kernel='linear'))
    clfs.append(sklearn.svm.SVC(kernel='rbf'))
    clfs.append(sklearn.svm.SVC(kernel='poly', degree=3))
    clfs.append(sklearn.dummy.DummyClassifier())
    clfs.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=5))
    res_with_scaling = generate_res_as_in_paper(df, clfs, preprocess_scaling=True)
    hdf['res_with_scaling'] = pd.DataFrame(res_with_scaling)

    pylab.figure("with_scaling")
    plot_res_paper(hdf['res_with_scaling'])

    res_without_scaling = generate_res_as_in_paper(df, clfs, preprocess_scaling=False)
    hdf['res_without_scaling'] = pd.DataFrame(res_without_scaling)
    pylab.figure("without_scaling")
    plot_res_paper(hdf['res_without_scaling'])

    res_with_pca_withening_and_scaling = generate_res_as_in_paper(df, clfs, preprocess_scaling=False)
    hdf['res_with_pca_and_scaling'] = pd.DataFrame(res_with_pca_withening_and_scaling)
    pylab.figure("with_scaling_and_pca")
    plot_res_paper(hdf['res_with_pca_and_scaling'])


    hdf.close()

def evaluate_database_humavips(database_fname):
    hdf = pd.HDFStore(database_fname)
    df = hdf['df']
    clfs = []
    #clfs.append(sklearn.ensemble.RandomForestClassifier(n_estimators=500, random_state=0))
    clfs.append(sklearn.svm.SVC(kernel='linear'))
    clfs.append(sklearn.svm.SVC(kernel='rbf'))
    clfs.append(sklearn.svm.SVC(kernel='poly', degree=3))
    clfs.append(sklearn.dummy.DummyClassifier())
    clfs.append(sklearn.neighbors.KNeighborsClassifier(n_neighbors=5))

    res, classifications = compute_cross_correlation_score(df, clfs, preprocess_scaling=True, nFold=10)

    #hdf['res_with_scaling'] = pd.DataFrame(res)
    res = pd.DataFrame(res)
    pylab.figure("with_scaling")
    plot_res_paper(res)

    filebis = pd.HDFStore('results_humavips')
    filebis['res'] = res

## TODO: refactor le code car la c'est vraiment du one shot pourri pour le compte rendu :
    filter = ['sklearn.svm.classes', 'rbf']
    predicted=[]
    expected=[]
    labels = clfs[1].classes_
    for c in classifications:
        for predicted_class, expected_class in zip(np.array(c.predicted_class), np.array(c.expected_class)):
            predicted.append(predicted_class)
            expected.append(expected_class)

    print(len(predicted))


    matrix = sklearn.metrics.confusion_matrix(predicted, expected, labels=labels)
    sound_classification.confusion_matrix.displayConfusionMatrix(matrix, labels=labels)



    ## TODO:  le fait que classifications soit des namedtuple c'est chiant.. une dataframe ca serait miexu
    ## genre pour filter sur le classifier -> a regarder demain
    #import IPython
    #IPython.embed()

def main():
    evaluate_database_humavips('test_database_aldebaran_features_1024_48000Hz.h5')
    evaluate_database_humavips('test_database_humavips_features_1024_48000Hz.h5')


if __name__ == "__main__":
    main()
