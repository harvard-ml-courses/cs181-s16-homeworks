from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import pandas as pd
from numpy.linalg import inv
from numpy.linalg import det

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
    
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
        
        MU0=X0.mean(axis=0)
        MU1=X1.mean(axis=0)
        MU2=X2.mean(axis=0)
        
        SIGMA0=np.cov(X0.T)
        SIGMA1=np.cov(X1.T)
        SIGMA2=np.cov(X2.T)
        
        PI0_MLE=N0/float(N)
        PI1_MLE=N1/float(N)
        PI2_MLE=N2/float(N)
        
        X0s=X0-MU0
        X1s=X1-MU1
        X2s=X2-MU2
        SIGMA0_MLE=(1/float(N0))*X0s.T.dot(X0s)
        SIGMA1_MLE=(1/float(N1))*X1s.T.dot(X1s)
        SIGMA2_MLE=(1/float(N2))*X2s.T.dot(X2s)
        
        SIGMA_MLE=(N0/float(N))*SIGMA0_MLE+(N1/float(N))*SIGMA1_MLE+(N2/float(N))*SIGMA2_MLE
        SIGMAINV=inv(SIGMA_MLE)
        
        SIGMA0INV=inv(SIGMA0_MLE)
        SIGMA1INV=inv(SIGMA1_MLE)
        SIGMA2INV=inv(SIGMA2_MLE)
        SIGMA0DET=det(SIGMA0_MLE)
        SIGMA1DET=det(SIGMA1_MLE)
        SIGMA2DET=det(SIGMA2_MLE)
        
        W00=-0.5*(MU0).dot(SIGMAINV).dot(MU0.T)+PI0_MLE
        W10=-0.5*(MU1.T).dot(SIGMAINV).dot(MU1)+PI1_MLE
        W20=-0.5*(MU2.T).dot(SIGMAINV).dot(MU2)+PI2_MLE
        
        W0=np.dot(SIGMAINV,MU0)
        W1=np.dot(SIGMAINV,MU1)
        W2=np.dot(SIGMAINV,MU2)


        V20=-0.5*SIGMA0INV
        V10=SIGMA0INV.dot(MU0)
        V00=-0.5*MU0.T.dot(SIGMA0INV).dot(MU0)-0.5*np.log(SIGMA0DET)+np.log(PI0_MLE)
        
        V21=-0.5*SIGMA1INV
        V11=SIGMA1INV.dot(MU1)
        V01=-0.5*MU1.T.dot(SIGMA1INV).dot(MU1)-0.5*np.log(SIGMA1DET)+np.log(PI1_MLE)
        
        V22=-0.5*SIGMA2INV
        V12=SIGMA2INV.dot(MU2)
        V02=-0.5*MU2.T.dot(SIGMA2INV).dot(MU2)-0.5*np.log(SIGMA2DET)+np.log(PI2_MLE)

        if isSharedCovariance==True:
            return W00, W10, W20, W0, W1, W2
        else:
            return V20, V10, V00, V21, V11, V01, V22, V12, V02

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        if isSharedCovariance==True:        
            W00, W10, W20, W0, W1, W2 = nb2.fit(X,Y)       
            A0=(W0.T).dot(X_to_predict.T)+W00
            A1=(W1.T).dot(X_to_predict.T)+W10
            A2=(W2.T).dot(X_to_predict.T)+W20
        
        else:
            V20, V10, V00, V21, V11, V01, V22, V12, V02 = nb2.fit(X,Y)
            A0=X_to_predict.dot(V20).dot(X_to_predict.T)+(V10.T).dot(X_to_predict.T)+V00
            A1=X_to_predict.dot(V21).dot(X_to_predict.T)+(V11.T).dot(X_to_predict.T)+V01
            A2=X_to_predict.dot(V22).dot(X_to_predict.T)+(V12.T).dot(X_to_predict.T)+V02
 
        Z0=np.exp(A0)/(np.exp(A0)+np.exp(A1)+np.exp(A2))
        Z1=np.exp(A1)/(np.exp(A0)+np.exp(A1)+np.exp(A2))
        Z2=np.exp(A2)/(np.exp(A1)+np.exp(A1)+np.exp(A2))       
        
        Z=pd.DataFrame([Z0,Z1,Z2])
        Z=Z.idxmax(axis=0) 
            
        return Z
    

    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .05))

        # Flatten the grid so the values match spec for self.predict
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        X_topredict = np.vstack((xx_flat,yy_flat)).T

        # Get the class predictions
        Y_hat = self.predict(X_topredict)
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))

        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
