# -*- coding: utf-8 -*-
__author__ = 'lgeorge'

import argparse
import logging
import pylab
import numpy as np
from sklearn.metrics import confusion_matrix

def displayConfusionMatrix(aConfusion_matrix, labels=None):
    #ax.set_xticklabels([''] + labels)
    """
    :param aConfusion_matrix: confusion matrix not normalized
    :param labels: labels of each class (one per row of matrix)
    :return:

    usage:  sklearn.metrics.confusion_matrix(Y, clf.predict(X))
    """

    aConfusion_matrix = np.array(aConfusion_matrix)
    normalized_confusion_matrix = np.array(aConfusion_matrix.copy(), dtype=np.float)
    for i in range(normalized_confusion_matrix.shape[0]):
        sum_line = float(np.sum(aConfusion_matrix[i, :]))
        normalized_confusion_matrix[i,:] = normalized_confusion_matrix[i,:] / sum_line


    fig = pylab.figure('confusion matrix')
    ax = fig.add_subplot(111)

    # colormap = pylab.cm.jet  # bad # colormap = pylab.cm.bwr  # ok
    colormap = pylab.cm.coolwarm  # seems to be good
    cax = ax.matshow(normalized_confusion_matrix, cmap=colormap, interpolation='None')
    ax.xaxis.set_label_position('top')
    if labels is not None:
        #ax.set_xticklabels([''] + labels)
        #ax.set_yticklabels([''] + labels)
        pylab.xticks(np.arange(len(labels)), [''] + labels, rotation=90)
        pylab.yticks(np.arange(len(labels)), [''] + labels)
    cb = fig.colorbar(cax)
    #cb.set_clim(0,1)
    #cb.solids.set_edgecolor("face")
    cb.set_label('Percentage')

    for i in range(normalized_confusion_matrix.shape[0]):
        for j in range(normalized_confusion_matrix.shape[1]):
            val = aConfusion_matrix[i, j]
            if val == 0:
                cell = ""
            else:
                cell = "{0:.0f}".format( val )
            #pylab.text(j - .2, i + .2, cell, fontsize=14,  va='center', ha='center')
            pylab.text(j , i , cell, fontsize=14,  va='center', ha='center')
    pylab.xlabel('Predicted')
    pylab.ylabel('Expected')
    pylab.tight_layout()

    pylab.show()
    return fig


