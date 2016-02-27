import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp
import pandas as pd

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
        self.X = X
        self.C = C       
        
        eta = .0001
        lambda_parameter = .01
        max_iter=5000        
        
        N=X.shape[0]
        tn = pd.DataFrame(0,index=range(N),columns=['C0','C1','C2'])
        tn.loc[C==0,'C0']=1
        tn.loc[C==1,'C1']=1
        tn.loc[C==2,'C2']=1

        w0=np.random.random(X.shape[1])
        w1=np.random.random(X.shape[1])
        w2=np.random.random(X.shape[1])

        a0=np.dot(X,w0)
        a1=np.dot(X,w1)
        a2=np.dot(X,w2)
        yn0=np.exp(a0)/(np.exp(a0)+np.exp(a1)+np.exp(a2))
        yn1=np.exp(a1)/(np.exp(a0)+np.exp(a1)+np.exp(a2))
        yn2=np.exp(a2)/(np.exp(a0)+np.exp(a1)+np.exp(a2))

        L=-(tn['C0']*np.log(yn0)+tn['C1']*np.log(yn1)+tn['C2']*np.log(yn2)).sum()
        iter =1
        converged= False

        while not converged: 
        
            grad0= (np.array(yn0-tn['C0'])*X.T).sum(1)
            grad1= (np.array(yn1-tn['C1'])*X.T).sum(1)
            grad2= (np.array(yn2-tn['C2'])*X.T).sum(1)
        
            w0=w0-eta*grad0 
            w1=w1-eta*grad1
            w2=w2-eta*grad2
            #print w0, w1, w2    
            
            a0=np.dot(X,w0)
            a1=np.dot(X,w1)
            a2=np.dot(X,w2)
            yn0=np.exp(a0)/(np.exp(a0)+np.exp(a1)+np.exp(a2))
            yn1=np.exp(a1)/(np.exp(a0)+np.exp(a1)+np.exp(a2))
            yn2=np.exp(a2)/(np.exp(a0)+np.exp(a1)+np.exp(a2))
        
            Lnew=-(tn['C0']*np.log(yn0)+tn['C1']*np.log(yn1)+tn['C2']*np.log(yn2)).sum()
        
            if abs(L-Lnew)<lambda_parameter:
                print 'Converged in ', iter, ' iterations'
                converged = True
            
            L=Lnew 
            iter += 1  
            #print L    
            
            if iter == max_iter:
                print 'Max interactions exceeded!'
                converged = True
        
        return w0, w1, w2

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        W0,W1,W2=lr.fit(X,Y)

        A0=np.dot(X_to_predict,W0)
        A1=np.dot(X_to_predict,W1)
        A2=np.dot(X_to_predict,W2)
                
        zn0=np.exp(A0)/(np.exp(A0)+np.exp(A1)+np.exp(A2))
        zn1=np.exp(A1)/(np.exp(A0)+np.exp(A1)+np.exp(A2))
        zn2=np.exp(A2)/(np.exp(A0)+np.exp(A1)+np.exp(A2))
                
        Z=pd.DataFrame([zn0,zn1,zn2])
        Z1=Z.idxmax(axis=0) 

        return Z1

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
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
