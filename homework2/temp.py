# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from numpy.linalg import inv

N=int(X.shape[0])
tn = pd.DataFrame(0,index=range(N),columns=['C0','C1','C2'])
tn.loc[Y==0,'C0']=1
tn.loc[Y==1,'C1']=1
tn.loc[Y==2,'C2']=1

N0=int(tn['C0'].sum())
N1=int(tn['C1'].sum())
N2=int(tn['C2'].sum())

X0=np.array(tn['C0'])*X.T
X0=X0[X0 != 0]
X0=np.reshape(X0,(2,N0))

X1=np.array(tn['C1'])*X.T
X1=X1[X1 != 0]
X1=np.reshape(X1,(2,N1))

X2=np.array(tn['C2'])*X.T
X2=X2[X2 != 0]
X2=np.reshape(X2,(2,N2))

MU0=X0.mean(axis=1)
MU1=X1.mean(axis=1)
MU2=X2.mean(axis=1)

SIGMA0=np.cov(X0)
SIGMA1=np.cov(X1)
SIGMA2=np.cov(X2)

SIGMA=SIGMA0

PI0_MLE=N0/float(N)
PI1_MLE=N1/float(N)
PI2_MLE=N2/float(N)

SIGMAINV=inv(SIGMA)

W00=-0.5*(MU0.T).dot(SIGMAINV).dot(MU0)+PI0_MLE
W10=-0.5*(MU1.T).dot(SIGMAINV).dot(MU1)+PI1_MLE
W20=-0.5*(MU2.T).dot(SIGMAINV).dot(MU2)+PI2_MLE

W0=np.dot(SIGMAINV,MU0)
W1=np.dot(SIGMAINV,MU1)
W2=np.dot(SIGMAINV,MU2)

A0=(W0.T).dot(X.T)+W00
A1=(W1.T).dot(X.T)+W10
A2=(W2.T).dot(X.T)+W20

Y0=np.exp(A0)/(np.exp(A0)+np.exp(A1)+np.exp(A2))
Y1=np.exp(A1)/(np.exp(A0)+np.exp(A1)+np.exp(A2))
Y2=np.exp(A2)/(np.exp(A1)+np.exp(A1)+np.exp(A2))

print Y0, Y1, Y2

print X