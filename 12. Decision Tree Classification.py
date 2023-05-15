#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Creating the dataset 
   
dataset = pd.read_csv('C:\\Users\\Dell\\Downloads\\dataset\\Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values


# In[3]:


print(x)


# In[4]:


print(y)


# In[6]:


# Splitting the dataset onto training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size=0.2 , random_state=0)


# In[7]:


print(x_train)


# In[8]:


print(y_train)


# In[9]:


print(x_test)


# In[10]:


print(y_test)


# In[11]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[12]:


print(x_train)


# In[13]:


print(x_test)


# In[14]:


# Training the decision tree model on the training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)
classifier.fit(x_train,y_train)


# In[15]:


# Predicting the new result
print(classifier.predict(sc.transform([[30,87000]])))


# In[17]:


#Predicting the test set results
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[18]:


# Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:




