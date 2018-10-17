
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import pandas_profiling
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


fb_data = pd.read_csv('pseudo_facebook.csv', sep='\s+')
fb_data.head()


# In[3]:


report = pandas_profiling.ProfileReport(fb_data)
report.to_file("fb_data.html")


# In[4]:


gender = fb_data.groupby('gender').count()


# In[5]:


max_gen = gender[gender.userid.max() == gender.userid].index.values
fb_data.dropna(inplace=True)
fb_data.head()


# In[6]:


report = pandas_profiling.ProfileReport(fb_data)
report.to_file("fb_data_after.html")


# In[7]:


cnt_1900,cnt_1920,cnt_1940,cnt_1960,cnt_1980,cnt_2000,other = 0,0,0,0,0,0,0

for i in fb_data['dob_year']:
    if i>=1900 and i<1920:
        cnt_1900 += 1
    elif i>=1920 and i<1940:
        cnt_1920 +=1
    elif i>=1940 and i<1960:
        cnt_1940 +=1
    elif i>=1960 and i<1980:
        cnt_1960 +=1
    elif i>=1980 and i<2000:
        cnt_1980 +=1
    elif i>=2000:
        cnt_2000 +=1
    else:
        other +=1

year_wise = pd.DataFrame({'1900-1920':[cnt_1900], '1920-1940':[cnt_1920], '1940-1960':[cnt_1940], '1960-1980':[cnt_1960], 
                         '1980-2000':[cnt_1980], '2000-more':[cnt_2000]})
year_wise = year_wise.T
year_wise.rename(columns={0:'ppl_count'})


# In[8]:


year_wise.plot.bar()
plt.title('Number of people on Facebook based on their the birth year')
plt.legend(loc='upper left')
plt.show()


# In[9]:


gender.plot.barh(y='userid')
plt.xlabel('number of people')
plt.title('number of people on FB based on Gender')
plt.legend('no ppl',loc='upper left')
plt.show()


# In[10]:


age_data = fb_data.groupby('age').mean()
age_data.head()


# In[11]:


age_data.plot.line(y='tenure')
plt.ylabel('tenure')
plt.title('on average how long people on FB')
plt.show()


# In[12]:


age_data.plot.line(y='friend_count')
plt.ylabel('friend_count')
plt.title('on average number of friends based on age')
plt.show()


# In[13]:


data_below_18 = fb_data[fb_data['age']<=18]
data_bw_19_30 = fb_data[(fb_data['age']<=30)&(fb_data['age']>18)]
data_bw_31_60 = fb_data[(fb_data['age']<=60)&(fb_data['age']>30)]
data_bw_61_100 = fb_data[(fb_data['age']<=100)&(fb_data['age']>60)]
data_more_101 = fb_data[fb_data['age']>=101]


# In[14]:


data_below_18.head()


# In[15]:


data_bw_19_30.head()


# In[16]:


data_bw_31_60.head()


# In[17]:


data_bw_61_100.head()


# In[18]:


data_more_101.head()


# In[19]:


sns.barplot(x='age',y='likes',data=data_more_101)
plt.title('max likes given by a age group more than 100')
plt.show()


# In[20]:


sns.factorplot(x='age', y='likes_received', data=data_bw_19_30)
plt.title('max likes received by a age group between 19 and 30')
plt.show()


# In[21]:


sns.boxplot(x='gender', y='age', data=data_bw_31_60)
plt.title('no of male and female between the age group 31 to 60')
plt.show()


# In[22]:


sns.pairplot(data_below_18[['gender','likes_received','mobile_likes_received','www_likes_received']],
             hue='gender', diag_kind="hist")
# The histogram on the diagonal allows us to see the distribution of a single variable 
# while the scatter plots on the upper and lower triangles show the relationship (or lack thereof) between two variables
plt.show()


# In[23]:


sns.lmplot(x='age', y='friendships_initiated', data=data_bw_61_100,fit_reg=False, aspect=2.5, x_jitter=.01)
plt.title('relation between age and friend request initiated')
plt.show()


# In[24]:


sns.pairplot(fb_data[['gender','likes','mobile_likes','www_likes','likes_received','mobile_likes_received','www_likes_received']]
            , hue='gender', diag_kind="hist")
# The histogram on the diagonal allows us to see the distribution of a single variable 
# while the scatter plots on the upper and lower triangles show the relationship (or lack thereof) between two variables
plt.show()


# In[25]:


gender_on_likes = fb_data.groupby('gender').sum()
gender_on_likes


# In[26]:


gender_on_likes.plot.bar(y='likes_received')
plt.ylabel('number of likes received')
plt.title('number of likes received on FB based on Gender')
plt.legend('no likes',loc='upper left')
plt.show()


# In[27]:


fb_details = pd.read_csv('pseudo_facebook.csv', sep='\s+', parse_dates=[['dob_year','dob_month','dob_day']], 
                         index_col='dob_year_dob_month_dob_day')
#fb_details.rename(index={'dob_year_dob_month_dob_day':'dateOfBirth'},inplace=True)
fb_details.head()


# In[35]:


age_mean = np.mean(fb_data['age'])
age_std = np.std(fb_data['age'])
pdf = stats.norm.pdf(fb_data['age'], age_mean, age_std)
plt.plot(fb_data['age'], pdf)
plt.hist(fb_data['age'], density=True)
plt.show()


# In[36]:


frndcnt_mean = np.mean(fb_data['friend_count'])
frndcnt_std = np.std(fb_data['friend_count'])
pdf = stats.norm.pdf(fb_data['friend_count'], frndcnt_mean, frndcnt_std)
plt.plot(fb_data['friend_count'], pdf)
plt.hist(fb_data['friend_count'], density=True)
plt.show()

