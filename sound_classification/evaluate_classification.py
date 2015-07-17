__author__ = 'lgeorge'

#import seaborn as sns
import pylab
import classification_service
import numpy as np
import sound_classification.confusion_matrix
from sklearn.metrics import confusion_matrix

def plot_distribution_true_false(prediction_df):
    """

    :param prediction_df:
    :return:
    """
    mask_well_classified = prediction_df.expected == prediction_df.class_predicted
    for group_name, g in prediction_df.groupby('expected'):
        print("group_name %s" % group_name)
        pylab.figure()
        try:
            v = g.confidence[mask_well_classified]
            pylab.hist(list(v), color='g', alpha=0.3, normed=0, range=(0,1), bins=10)
            #sns.distplot(g.confidence[mask_well_classified], color='g', bins=11)  # TRUE POSITIVE
        except Exception as e:
            print(e)
        mask_wrong = (prediction_df.class_predicted == group_name) & (prediction_df.expected != group_name)  # FALSE POSITIVE
        #v = g.confidence[~mask_well_classified]
        try:
            v = prediction_df.confidence[mask_wrong]
            pylab.hist(list(v), color='r', alpha=0.3, normed=0, range=(0,1), bins=10)
            #sns.distplot(v, color='r', bins=11)
        except Exception as e:
            print(e)
            #print(len(v))
            pass
        print("FIN figure %s" % group_name)
        pylab.show()
        print("")
    #return pylab.gcf()


def get_expected_predicted_stratified_fold(stratified_fold, df, window_block=None, clf=None, window_block_learning=None, calibrate_score=False):
    """
    Tool function to report classification accuracy for our classification tools
    """
    predicted=[]
    expected=[]
    filenames = []

    fold_num = 0
    for train_set, test_set in stratified_fold:
        train_files = df.iloc[train_set].filename
        sound_classification_obj = classification_service.SoundClassification(train_files.tolist(), clf=clf, window_block_learning=window_block_learning , calibrate_score=calibrate_score)
        sound_classification_obj.learn()
        labels = sound_classification_obj.clf.classes_
        for index in test_set:
            val = df.iloc[index]
            try:
                prediction = sound_classification_obj.processed_wav(val.filename, window_block=window_block, ignore_fs=True)
                print(len(prediction))
                expected.extend([val.classname]*len(prediction))
                predicted.extend(prediction)
                filenames.extend(['_'.join([val.filename, '_fold%s' % fold_num])]*len(prediction))  # we append the num of fold to filename to have easy difference after that.... TODO: use another column
            except classification_service.SoundClassificationException as e:
                print("Exception %s detected on %s" % (e, val.filename))
        fold_num += 1
    return expected, predicted, labels, filenames   # ca commence a faire beaucoup, pourquoi ne pas renvoyer un dictionnaire, ou un pandas DataFrame: TODO

def filter_out_based_on_threshold(prediction_df, score_threshold_dict):
    # we use threshold_dict to filter out value with confidence bellow threshold and assigned class UNKNOWN
    for name, threshold in score_threshold_dict.iteritems():
        mask = (prediction_df.predicted_class == name) & (prediction_df.confidence < threshold)
        prediction_df.predicted_class[mask] = 'UNKNOWN'
    return prediction_df

def print_report(expected, predicted_class, labels, score_threshold_dict=None):
    # compute confusion matrix
    matrix = confusion_matrix(expected, predicted_class, labels=labels)
    print(matrix)

    # plot confusion matrix
    fig = sound_classification.confusion_matrix.displayConfusionMatrix(matrix, labels=labels)
    return fig

    # print report
    #import sklearn
    #print(sklearn.metrics.classification_report(expected, predicted_class))




# Warning this methods are not optimal at all.. the complexity is high.. but we don't really care here..



