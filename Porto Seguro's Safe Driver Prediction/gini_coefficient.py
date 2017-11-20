'''
This script calculates gini coefficient as an accuracy metric for learning model predictions.
Originally submitted by Kaggle user Patrick (https://www.kaggle.com/patrick7) on the 
Allstate Claim Prediction Challenge Discussion boards
(https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703)
'''

import numpy as np

def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)