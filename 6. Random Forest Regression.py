#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#Importing the dataset
dataset = pd.read_csv('C:\\Users\\Dell\\Downloads\\dataset\\Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values


# In[3]:


print(x)


# In[4]:


print(y)


# In[6]:


#Training the random forest regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(x,y)


# In[7]:


#Predicting the new result
regressor.predict([[6.5]])


# In[8]:


#Visualising the random forest regression results
x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid, regressor.predict(x_grid),color='black')
plt.title('truth vs dare(Decision Tree Regression)')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()


# In[ ]:




