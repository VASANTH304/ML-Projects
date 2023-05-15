#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importig the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#importing the dataset
dataset = pd.read_csv('C:\\Users\\Dell\\Downloads\\dataset\\Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


# In[3]:


print(x)


# In[4]:


print(y)


# In[5]:


#Training the decision tree regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)


# In[8]:


#Predicting the new result
regressor.predict([[6.5]])


# In[9]:


#Visualizing the decision tree regression results (Higher Resolution)
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid, regressor.predict(x_grid),color='black')
plt.title('truth vs dare(Decision Tree Regression)')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()


# In[ ]:




