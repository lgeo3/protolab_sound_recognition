__author__ = 'lgeorge'

import numpy as np

def compute_precision_cumulative_curve(df, true_positive_class=None, step=0.01):
    """
    Compute `cumulative precision` based on predicted/expected of a specific class

    :param df: a dataframe with columns class_expected, class_predicted, confidence
    :param true_positive_class: a class name
    :param step: step for the histogram
    :return: bins_indices, and cumulative precision value

    precision of bin 1 should be 100 % for instance
    precision of bin 0 could be 0% (if we have wrong values in class_predicted for instance)
    """
    mask_true = df.class_expected == df.class_predicted  # TRUE POSITIVE
    mask_wrong = (df.class_predicted == true_positive_class) & (df.class_expected != true_positive_class)  # FALSE POSITIVE
    bins = np.arange(0, 1, step)
    res = []
    for prediction_threshold in bins:
        false_cumulative = np.sum(df.confidence[mask_wrong] >= prediction_threshold)
        true_cumulative = np.sum(df.confidence[mask_true] >= prediction_threshold)
        precision_cumulative = true_cumulative / float(false_cumulative + true_cumulative)
        res.append(precision_cumulative)
    return bins, res

def plot_precision_cumulative_curve(prediction_df, true_positive_class=None, step=0.01):
    import pylab
    bins, res = compute_precision_cumulative_curve(prediction_df, true_positive_class=true_positive_class, step=step)
    pylab.scatter(bins, res)


def get_threshold_cum_precision(prediction_df, true_positive_class=None, min_expected_cum_precision=0.99):
    """
    Compute `cumulative precision` based on predicted/expected of a specific class

    :param df: a dataframe with columns class_expected, class_predicted, confidence
    :param true_positive_class: a class name
    :param step: step for the histogram
    :return: bins_indices, and cumulative precision value
    """
    bins, precision_cumulative = compute_precision_cumulative_curve(prediction_df, true_positive_class=true_positive_class)
    valid_entry = np.argwhere(np.array(precision_cumulative) >= min_expected_cum_precision)[0]
    if valid_entry == []:
        return 1  # worst threshold..
    else:
        return bins[valid_entry[0]] # first position

