# CS 181, Spring 2016
# Homework 5: EM
# Name: Yohann Smadja
# Email: smadja@post.harvard.edu

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import time


class LDA(object):

	# Initializes with the number of topics
	def __init__(self, num_topics):
		self.num_topics = num_topics

	# This should run the M step of the EM algorithm
	def M_step(self):      
		pass  

	# This should run the E step of the EM algorithm
	def E_step(self):
		pass       

	# This should print the topics that you find
	def print_topics(self):      
		pass      
        

# This line loads the text for you. Don't change it! 
text_data = np.load(r"C:\Users\Yohann\Documents\Machine Learning\hw5\hw5\homework5\text.npy", allow_pickle=False)
with open('C:\Users\Yohann\Documents\Machine Learning\hw5\hw5\homework5\words.txt', 'r') as f:
	word_dict_lines = f.readlines()

# Feel free to add more functions as needed for the LDA class. You are welcome to change anything below this line. 
# However, your code should be contained in the constructor for the LDA class, and should be executed in a way 
# similar to the below.
LDAClassifier = LDA(num_topics=10)
LDAClassifier.print_topics()


opti=[]
num_topics=10
num_iter=30
D=text_data[len(text_data)-1][0].astype('int')  #number of docs
t0=random.sample(range(1, 100), 10)
t1=[float(i) for i in t0]  
theta=[i/sum(t1) for i in t1]  

b0=np.random.random((len(word_dict_lines), num_topics))
b1=b0/b0.sum(0)
beta=pd.DataFrame(b1,index=word_dict_lines,columns=range(num_topics))
beta['word_id']=range(len(beta))

df_text_data=pd.DataFrame(text_data,columns=['doc_id','word_id','count'])
df_text_data=df_text_data.astype('float')

df_Nd=pd.pivot_table(df_text_data,index="doc_id",values="count", aggfunc=np.sum)

pt=pd.pivot_table(df_text_data,values='count',index=['word_id','doc_id'],aggfunc=np.sum)
pt=pt.to_frame()
pt['doc_id']=pt.index.get_level_values('doc_id')
pt['word_id']=pt.index.get_level_values('word_id')

for j in range(num_iter):
    start=time.time()
    #E step
    df_doc=df_text_data.merge(beta,on='word_id',how='left')
    for i in range(10):
        df_doc[i]=df_doc[i]*df_doc['count']
    df_q=pd.pivot_table(df_doc,index='doc_id',aggfunc=np.sum)
    df_q=df_q.drop(['count','word_id'],1)
    df_q=df_q*theta
    df_q=df_q.div(df_q.sum(1),0)
    
    #M-stetp
    theta=df_q.sum(0)/D
    df_q['doc_id']=df_q.index
    denK=df_q.multiply(df_Nd,axis=0).sum(0) 
    denK=denK[:num_topics]
    ptM=pt.merge(df_q,on='doc_id',how='left')
    df_q=df_q.drop('doc_id',1)
    for i in range(num_topics):
        ptM[i]=ptM['count']*ptM[i]
    ptM=ptM.drop(['count','doc_id','word_id'],1)
    ptM.index=pt.index
    ptM['word_id']=ptM.index.get_level_values('word_id')
    ptM=pd.pivot_table(ptM,index='word_id',aggfunc=np.sum)
    beta=ptM/denK  
    
    #Loss function
    beta['word_id']=beta.index
    ptEM=pt.merge(beta,on='word_id',how='left')
    for i in range(num_topics):
        ptEM[i]=ptEM['count']*np.log(ptEM[i])
    ptEM=ptEM.drop(['count','word_id'],1)    
    ptEM=pd.pivot_table(ptEM,index='doc_id',aggfunc=np.sum)
    for i in range(i):
        ptEM[i]=ptEM[i]+np.log(theta[i])
    opti.append((df_q*ptEM).sum().sum())
    print j, round(time.time()-start,0)


plt.plot(opti)

opti

for i in range(10):
    k0=int(beta[i].argmax())
    print word_dict_lines[k0]

