
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
import seaborn as sns

data_file = pd.read_csv('flight_data.csv', float_precision='round_trip') # reading csv file
data_file.dropna(inplace=True) # dropping NA values and actual dta is modified
data_file.head() # printing first 5 lines in data frames


# In[2]:


data_file.tail()  # printing last 5 lines in data frames


# In[ ]:


plt.rcParams['figure.figsize'] = (80,40)
sns.barplot(x='dest', y='month', hue='origin', data=data_file)
plt.ylabel('month')
plt.xlabel('destination airport')
# we can see that number of flights per month from one point to another


# In[4]:


# creating a table by grouping data in 2 level basis, first based on origin and second on destination
table1 = pd.pivot_table(data_file, values='distance', index=['origin','dest'], columns=['month'])
table1.fillna(0, inplace=True)  # filling NA values with zeros
table1.head()


# In[5]:


# creating a new column dep_in% by calculating percentage of dep_delay variable by sched_dep_time variable 
# to know delay percentage of flight
table2 = data_file
table2['dep_in'] = table2['dep_delay']/data_file['sched_dep_time']
table2['dep_in%'] = table2['dep_delay']*100/data_file['sched_dep_time']


# In[6]:


# a new data frame obtained by grouping based on origin variable and finding minimum
best_air = table2.groupby('origin').min()
# best origin airport is obtained interms of departure time
print(best_air[best_air['dep_in%'] == best_air['dep_in%'].min()].index.values + ' is best airport interms of depature time')


# In[7]:


plt.rcParams['figure.figsize'] = (25,6)
sns.distplot(a=table2['dep_delay'], hist=True)
# flights depart mostly around 10mins early or late to the scheduled time


# In[8]:


# new column speed is created by calculating speed of airplane using distance and time taken to travel
table3 = data_file
table3['speed'] = table3['distance']/(table3['arr_time'] - table3['dep_time'])
table3.head()


# In[9]:


#table3[table3.pivot_table(values='speed', index=['flight']) == table3.pivot_table(values='speed', index=['flight']).max()].index.values

# obtaining charactersitics of column speed taking flight column as index
print('Aircraft speed analysis')
table3.pivot_table(values='speed', index=['flight']).describe()


# In[10]:


sns.jointplot(x='dep_time', y='speed', data=table3)
# speed of the flight is on an average between 0 to 12.5miles/hr whatever maybe the departure time


# In[11]:


sns.jointplot(x='arr_time', y='speed', data=table3)
# speed of the flight is on an average between 0 to 12.5miles/hr whatever maybe the arrival time


# In[12]:


# trying to find how many flights are on time and percentage of it among all flights
counter = 0
for i in data_file['arr_delay']:
    if i == 0:
        counter += 1

on_time = (counter/data_file.arr_delay.count())*100
print('the percentage of flights arriving on time is {}%'.format(round(on_time,2)))


# In[13]:


# data frame is created containing total number of flights going to a particular destination
print('the total number of flights headed to a destination')
no_flight=data_file.pivot_table(values='flight', index=['dest'], aggfunc='count')


# In[14]:


plt.rcParams['figure.figsize'] = (50,20)
sns.pointplot(data=no_flight.T)
# total number of flights going to a ATL destination is higher

