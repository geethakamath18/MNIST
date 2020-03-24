#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mnist.loader import MNIST
import cvxpy as cp
import numpy as np
from random import randrange
import matplotlib.pyplot as plt


# In[3]:


def compute_loss(W, X, y, reg):
  
  loss=0.0
  gradient=np.zeros(W.shape) #initialize the gradient as zero
  num_classes=W.shape[1] #Number of classes
  num_train=X.shape[0] #Number of Training examples=60000
  scores=X.dot(W)
  correct_class_scores=scores[np.arange(num_train),y].reshape(num_train,1)
  margin=np.maximum(0,scores-correct_class_scores+1)
  margin[np.arange(num_train),y] = 0
  loss=margin.sum()/num_train
  loss+=reg*np.sum(W*W) #Loss+Regularization

  # Compute gradient
  margin[margin>0]=1
  valid_margin_count=margin.sum(axis=1)
  # Subtract in correct class (-s_y)
  margin[np.arange(num_train),y]-=valid_margin_count
  gradient=(X.T).dot(margin)/num_train

  gradient=gradient+reg*2*W #Regularization gradient update

  return loss,gradient


# In[4]:


class LinearClassifier(object):
    def predict(self, X):
        y_pred = np.zeros(X.shape[0]) #Predicted labels
        scores=X.dot(self.W)
        y_pred=scores.argmax(axis=1)
        return y_pred

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train, dim=X.shape
        num_classes=np.max(y)+1
        if self.W is None:
          self.W=0.001*np.random.randn(dim, num_classes) #Initialization for W

        
        loss_history=[] #Stochastic gradient descent
        for it in range(num_iters):
          X_batch=None
          y_batch=None
          batch_indices=np.random.choice(num_train, batch_size, replace=False)
          X_batch=X[batch_indices]
          y_batch=y[batch_indices]

          
          loss, grad=self.loss(X_batch, y_batch, reg) #Compute loss and gradient
          loss_history.append(loss)

          
          self.W=self.W-learning_rate*grad #Updating weight 
          if it%1000==0: #Checking if iteration%1000=0 and performing prediction
            y_train_pred=self.predict(X_test)
            plotnum1.append((np.mean(Y_test==y_train_pred),))
            plt.plot()
        return loss_history


class LinearSVM(LinearClassifier):

      def loss(self, X_batch, y_batch, reg):
        return compute_loss(self.W, X_batch, y_batch, reg)


# In[5]:


mndata=MNIST('E:/Grad School/Semester 2/ML/Homeworks/MNIST/')


# In[6]:


X_train, Y_train=mndata.load_training() #60000 samples
X_test, Y_test=mndata.load_testing()    #10000 samples


# In[7]:


X_train= np.asarray(X_train).astype(np.float32)
Y_train=np.asarray(Y_train).astype(np.int32)
X_test=np.asarray(X_test).astype(np.float32)
Y_test=np.asarray(Y_test).astype(np.int32)


# In[8]:


W = np.random.randn(784, 10)*0.0001 
plotnum1=[]


# In[9]:


plt.figure(figsize=(15,10))
plt.title('Accuracy')
plt.xlabel('Iteration number')
plt.ylabel('Prediction accuracy')
plotnum1.clear()
svm1=LinearSVM()
loss_hist=svm1.train(X_train, Y_train, learning_rate=1e-6, reg=10, num_iters=1000000, verbose=True)
plt.plot(plotnum1,label='10')
plotnum1.clear()
svm2=LinearSVM()
loss_hist=svm2.train(X_train, Y_train, learning_rate=1e-6, reg=1, num_iters=1000000, verbose=True)
plt.plot(plotnum1,label='1')
plotnum1.clear()
svm3=LinearSVM()
loss_hist=svm3.train(X_train, Y_train, learning_rate=1e-6, reg=0.1, num_iters=1000000, verbose=True)
plt.plot(plotnum1,label='0.1')
plotnum1.clear()
svm4=LinearSVM()
loss_hist=svm4.train(X_train, Y_train, learning_rate=1e-6, reg=0.01, num_iters=1000000, verbose=True)
plt.plot(plotnum1,label='0.01')
plt.plot([], [], ' ', label="1 unit=1000 iterations")
plt.legend(loc="lower right")
plt.show()


# In[ ]:




