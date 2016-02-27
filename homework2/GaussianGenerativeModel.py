from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c

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
        self.X=X
        self.Y=Y
        N=len(X)
        K=3
        self.mu=[0,0,0]
        for i in range (0, K):
            numerator=0
            denomenator=0
            for j in range(0,N):
                if Y[j]==i:
                    numerator=numerator+X[j]
                    denomenator=denomenator+1
            if denomenator>0:
                self.mu[i]=numerator/denomenator
        if self.isSharedCovariance:
            self.E=0
            for i in range (0, K):
                for j in range(0,N):
                    if Y[j]==i:
                        print((X[j]-self.mu[i])*(np.array(X[j]-self.mu[i]).reshape(len(X[j]),1)))
                        self.E=self.E+(X[j]-self.mu[i])*(np.array(X[j]-self.mu[i]).reshape(len(X[j]),1))/N
                        print(self.E)
            print(self.E,"asdfa")
        else:
            self.E=[0,0,0]
            for i in range (0, K):
                numerator=0
                denomenator=0
                for j in range(0,N):
                    if Y[j]==i:
#                        print((np.array(X[j]-self.mu[i]).reshape(len(X[j]),1)))
                        numerator=numerator+(X[j]-self.mu[i])*(np.array(X[j]-self.mu[i]).reshape(len(X[j]),1))
                        denomenator=denomenator+1
                if denomenator>0:
                    self.E[i]=numerator/denomenator
                    print(self.E[i])

        return

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        Y=[0]*len(X_to_predict)
        for i in range(0,len(X_to_predict)):
            est_lik=[0,0,0]
            if self.isSharedCovariance:
                for k in range(0,len(self.mu)):
                    est_lik[k]=multivariate_normal.pdf(X_to_predict[i], mean=self.mu[k], cov=self.E)
#                    est_lik[k]=(X_to_predict[i]-self.mu[k]).reshape(1,len(X_to_predict[i]))*self.E^{-1}*(X_to_predict[i]-self.mu[k])
                maxind=np.argmax(est_lik)
                Y[i]=maxind
            else:
                for k in range(0,len(self.mu)):
                    est_lik[k]=multivariate_normal.pdf(X_to_predict[i], mean=self.mu[k], cov=self.E[k])
#                    est_lik[k]=pow(np.linalg.det(self.E[k]),-1/2)*np.exp(0.5*((X_to_predict[i]-self.mu[k]).reshape(1,len(X_to_predict[i])))*self.E[k]^{-1})
                maxind=np.argmax(est_lik)
                Y[i]=maxind
        return(np.array(Y))
                
                
            

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
