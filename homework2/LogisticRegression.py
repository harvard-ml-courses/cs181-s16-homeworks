import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp
from sklearn import linear_model, datasets

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
    
    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, C):
        self.X=X
        self.C=C
        self.w=np.array([[1.]*len(X[1]),[1.]*len(X[1]),[1.]*len(X[1])])
        self.w1=np.array([[1.]*len(X[1]),[1.]*len(X[1]),[1.]*len(X[1])])
        K=3
        cont=0
        while cont<5000:
            cont=cont+1
            print(cont)
            self.w=self.w1
            for k in range (0,K):
                grad=0
                for n in range(0,len(X[0])):
                    den=0
                    for i in range(0,K):
                        den=den+np.exp(self.w[i].dot(X[n].reshape(len(X[n]),1)))
    #                    print(den,n,k)
                    if C[n]==k:
                        grad=grad+((np.exp(self.w[k].dot(X[n].reshape(len(X[n]),1)))/den)-1)*X[n]
                    else:
                        grad=grad+(np.exp(self.w[k].dot(X[n].reshape(len(X[n]),1)))/den)*X[n]
#                print(self.eta*grad)
#                print(self.w[k])
#                print(self.w1[k],"w1k prior")
                self.w1[k]-=self.eta*grad
                print(self.w1[k],"w1k posterior")
    #                print(self.w,self.w1,"2")
        return

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        Y=[0]*len(X_to_predict)
        for n in range(0,len(X_to_predict)):
            lik=[0,0,0]
            for k in range(0,len(self.w1)):
                lik[k]=np.array(self.w1[k]).reshape(len(self.w1[k]),1)*X_to_predict[n]
            maxind=np.argmax(lik)
            Y[n]=maxind
        return (np.array(Y))

    def visualize(self, output_file, width=2, show_charts=False):
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
        plt.pcolormesh(xx,yy, Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
