# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:53:10 2017

@author: 301339
"""

import numpy as np

actual = [0, 1, 0, 0, 1, 1, 0]
pred = [0.1, 0.2, 0.9, 0.15, 0.2, 0.65, 0]


all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
print(all)
all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
print(all)
totalLosses = all[:,0].sum()
print(totalLosses)
print(all[:,0].cumsum())
giniSum = all[:,0].cumsum().sum() / totalLosses
print(giniSum)

giniSum -= (len(actual) + 1) / 2.
print(giniSum)
print(giniSum / len(actual))
