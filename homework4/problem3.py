# CS 181, Spring 2016
# Homework 4: Clustering
# Name: Yohann Smadja
# Email: yohann.smadja@gmail.com

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from numpy import random, array
import sys

class KMeans(object):
	# K is the K in KMeans
	# useKMeansPP is a boolean. If True, you should initialize using KMeans++
	def __init__(self, K, useKMeansPP):
		self.K = K
        	self.useKMeansPP = useKMeansPP

	# X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
	def fit(self, X):
		max_iter=100
		iter=0
		convergence=False
		N=len(X)
		if useKMeansPP==False:              
			Z=np.random.randint(low=0,high=K,size=N) 
		else:
			MU=[0]*10  
			Z0=[0]*10
			k0=np.random.randint(low=0,high=N)
			Z0[0]=k0    
			MU[0]=X[k0] 
			Z=[0]*N   
			for k in np.arange(1,K):
				dn=[]        
				for n in range(N):
					d=[]
					for k1 in np.arange(0,k):
						d.append(np.linalg.norm(X[n]-MU[k1]))
					dn.append(min(d)) 
				pn=dn/np.sum(dn)
				Z0[k]=np.random.choice(range(N),p=pn)
				MU[k]=X[Z0[k]]      
			#for j in range(K):
				#plt.figure()
				#plt.imshow(MU[j], cmap='Greys_r')
				#plt.savefig(r'H:\Harvard ML\HW4\plusplus_%i.png' % j)    
				#plt.show()  
			for n in range(N):    
        			opti=[] 
        			for k2 in range(K):
        				opti.append(np.linalg.norm(X[n]-MU[k2]))
        			Z[n]= np.argmin(opti)
			Z=np.array(Z)
		sum_norm=[0]
		while (convergence==False or iter>max_iter):
        		MU=[]
        		for k in range(K):
        			condition = (Z==k)
        			MU.append(np.average(X[condition],0))
        		sn=0           
        		for n in range(N):    
        			opti=[] 
        			for k2 in range(K):
        				opti.append(np.linalg.norm(X[n]-MU[k2]))
        			Z[n]= np.argmin(opti)
        			sn+=min(opti)
        		if (sum_norm[len(sum_norm)-1]==sn):
                 		convergence=True
        		sum_norm.append(sn) 
		print len(sum_norm)-1  
		plt.figure()
		plt.plot(sum_norm[1:])
		plt.show()   
		return Z, MU    

	# This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
	def get_mean_images(self,X):
		Z, MU = KMeansClassifier.fit(X) 
		for j in range(K): 
        		plt.figure()
        		plt.imshow(MU[j], cmap='Greys_r')
        		plt.savefig(r'C:\Users\Yohann\Documents\Machine Learning\hw4\hw4complete\homework4\kmean20_%i.png' % j)    
        		plt.show()
        		#KMeansClassifier.create_image_from_array(MU[j])
          
	# This should return the arrays for D images from each cluster that are representative of the clusters.
	def get_representative_images(self, X):
		Z, MU = KMeansClassifier.fit(X)
		dist=[]
		picsD=[]
		N=len(pics)  
		for n in range(N):
   			if Z[n]==D:
  				picsD.append(pics[n])
		ND=len(picsD)
		for n in range(ND):
			dist.append(np.linalg.norm(picsD[n]-MU[Z[n]])) 
		#print dist
		min_distances=np.sort(dist)[0:9]
		for n in range(ND):
			if dist[n] in min_distances:
				plt.figure()
				plt.imshow(picsD[n], cmap='Greys_r')
				plt.show()    

	# img_array should be a 2D (square) numpy array.
	# Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
	# However, we do ask that any images in your writeup be grayscale images, just as in this example.
	def create_image_from_array(self, img_array):
		plt.figure()
		plt.imshow(img_array, cmap='Greys_r')
		plt.show()
		return

# This line loads the images for you. Don't change it! 
pics = np.load(r"C:\Users\Yohann\Documents\Machine Learning\hw4\hw4\homework4\images.npy", allow_pickle=False)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.
K = 10
useKMeansPP=False
D=0
KMeansClassifier = KMeans(K=10, useKMeansPP=True)
KMeansClassifier.fit(pics)
#KMeansClassifier.get_mean_images(pics)
#KMeansClassifier.get_representative_images(pics)
#KMeansClassifier.create_image_from_array(pics[1])

