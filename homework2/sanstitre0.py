# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:42:06 2016

@author: Vincent Lan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp
from sklearn import linear_model, datasets

x=0
i=0
while i<100:
    i=i+1
    x=x+3
print(x)

#K=3
#
#grad=0
#for n in range(0,len(X)):
#    den=0
#    for i in range(0,K):
#        den=den+np.exp(w[i].dot(X[n].reshape(len(X[n]),1)))
#    if Y[n]==1:
#        print("Yes")
#        print(np.exp(np.array(w[1]).dot(X[n].reshape(len(X[n]),1)))/den)
#        grad=grad+((np.exp(np.array(w[1]).dot(X[n].reshape(len(X[n]),1)))/den)-1)*X[n]
#print(grad)
