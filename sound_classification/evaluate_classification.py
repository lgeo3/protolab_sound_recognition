__author__ = 'lgeorge'

import seaborn as sns
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


def get_expected_predicted_stratified_fold(stratified_fold, df, window_block=None, clf=None, window_block_learning=None):
    """
    Tool function to report classification accuracy for our classification tools
    """
    predicted=[]
    expected=[]
    filenames = []

    fold_num = 0
    for train_set, test_set in stratified_fold:
        train_files = df.iloc[train_set].filename
        sound_classification_obj = classification_service.SoundClassification(train_files.tolist(), clf=clf, window_block_learning=window_block_learning )
        sound_classification_obj.learn()
        labels = sound_classification_obj.clf.classes_
        for index in test_set:
            val = df.iloc[index]
            try:
                prediction = sound_classification_obj.processed_wav(val.filename, window_block=window_block, ignore_fs=True)
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
    sound_classification.confusion_matrix.displayConfusionMatrix(matrix, labels=labels)

    # print report
    import sklearn
    print(sklearn.metrics.classification_report(expected, predicted_class))




# Warning this methods are not optimal at all.. the complexity is high.. but we don't really care here..



def compute_precision_cumulative_curve(prediction_df, true_positive_class=None, step=0.01):
    mask_true = prediction_df.expected == prediction_df.class_predicted  # TRUE POSITIVE
    mask_wrong = (prediction_df.class_predicted == true_positive_class) & (prediction_df.expected != true_positive_class)  # FALSE POSITIVE
    bins = np.arange(0, 1, step)
    res = []
    for prediction_threshold in bins:
        false_cumulative = np.sum(prediction_df.confidence[mask_wrong] >= prediction_threshold)
        true_cumulative = np.sum(prediction_df.confidence[mask_true] >= prediction_threshold)
        precision_cumulative = true_cumulative / float(false_cumulative + true_cumulative)
        res.append(precision_cumulative)
    return bins, res

def plot_precision_cumulative_curve(prediction_df, true_positive_class=None, step=0.01):
    bins, res = compute_precision_cumulative_curve(prediction_df, true_positive_class=true_positive_class, step=step)
    pylab.scatter(bins, res)


def get_threshold_cum_precision(prediction_df, true_positive_class=None, min_expected_cum_precision=0.99):
    bins, precision_cumulative = compute_precision_cumulative_curve(prediction_df, true_positive_class=true_positive_class)
    valid_entry = np.argwhere(np.array(precision_cumulative) >= min_expected_cum_precision)[0]
    if valid_entry == []:
        return 1  # worst threshold..
    else:
        return bins[valid_entry[0]] # first position

