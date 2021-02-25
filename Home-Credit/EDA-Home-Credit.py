#!/usr/bin/env python
# coding: utf-8

# ## CREDIT EDA : Problem Statement
# 
# #### The loan providing companies find it hard to give loans to the people due to their insufficient or non-existent credit history. Because of that, some consumers use it as their advantage by becoming a defaulter. Suppose you work for a consumer finance company which specialises in lending various types of loans to urban customers. You have to use EDA to analyse the patterns present in the data. This will ensure that the applicants are capable of repaying the loan are not rejected.

# #### - Importing the required packages

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 250)


# #### - Reading data from application_data.csv

# In[2]:


application_data = pd.read_csv('application_data.csv')
application_data.head()


# #### - Checking the shape of data

# In[3]:


application_data.shape


# #### - Checking the data types of data

# In[4]:


application_data.dtypes


# #### - Describing Statistical details of numerical data in Dataframe

# In[5]:


application_data.describe()


# ### Null value check and reshaping the data
# #### - Inspecting number of Null values columnwise and finding percentage of same

# In[6]:


application_data.isnull().sum()


# In[7]:


cent_null = round(100*(application_data.isnull().sum()/len(application_data.index)),2)
cent_null


# #### - Dropping columns with Null values pecentage greater than 13%

# In[8]:


miss_cols = cent_null[cent_null > 13].index

clean_app_data = application_data.drop(miss_cols, axis=1)
clean_app_data.head()


# #### - Checking the shape of data after dropping the columns

# In[9]:


clean_app_data.shape


# #### - Information about the data types, number of columns and rows, number of non-null rows for each column after dropping the columns

# In[10]:


clean_app_data.info()


# #### - Checking the Null values percentage of data after dropping the columns

# In[11]:


after_drop_cent = round(100*(clean_app_data.isnull().sum()/len(clean_app_data.index)),2)


# #### - Picking the columns with missing value more than 0% and less than 13% from dataframe after dropping the columns

# In[12]:


missing_cols = after_drop_cent[after_drop_cent > 0].index

clean_app_data[missing_cols].head(10)


# ### Handling missing values and imputation metric
# #### - Plotting the below columns to see the distribution of data in these columns

# In[13]:


plt.figure(figsize=(10,10))
plt.subplot(3,2,1)
sns.boxplot(clean_app_data.EXT_SOURCE_2)
plt.subplot(3,2,2)
sns.boxplot(clean_app_data.OBS_30_CNT_SOCIAL_CIRCLE)
plt.subplot(3,2,3)
sns.boxplot(clean_app_data.DEF_30_CNT_SOCIAL_CIRCLE)
plt.subplot(3,2,4)
sns.boxplot(clean_app_data.OBS_60_CNT_SOCIAL_CIRCLE)
plt.subplot(3,2,5)
sns.boxplot(clean_app_data.DEF_60_CNT_SOCIAL_CIRCLE)
plt.show()


# #### - Statistical description of EXT_SOURCE_2 column

# In[14]:


clean_app_data.EXT_SOURCE_2.describe()


# - **Observation:** The null values can be imputed using mean or median in above column

# #### - Statistical description of below columns

# In[15]:


print('{}\n'.format(clean_app_data.OBS_30_CNT_SOCIAL_CIRCLE.describe()))
print('{}\n'.format(clean_app_data.OBS_60_CNT_SOCIAL_CIRCLE.describe()))
print('{}\n'.format(clean_app_data.DEF_30_CNT_SOCIAL_CIRCLE.describe()))
print(clean_app_data.DEF_60_CNT_SOCIAL_CIRCLE.describe())


# - **Observation:** The outliers are spread over 75th and 100th quantile

# #### - Statistical description of below columns by considering the data lying within 99th quantile

# In[16]:


print('{}\n'.format(clean_app_data.OBS_30_CNT_SOCIAL_CIRCLE[clean_app_data.OBS_30_CNT_SOCIAL_CIRCLE <= 
                                        clean_app_data.OBS_30_CNT_SOCIAL_CIRCLE.quantile(0.99)].describe()))
print('{}\n'.format(clean_app_data.OBS_60_CNT_SOCIAL_CIRCLE[clean_app_data.OBS_60_CNT_SOCIAL_CIRCLE <= 
                                        clean_app_data.OBS_60_CNT_SOCIAL_CIRCLE.quantile(0.99)].describe()))
print('{}\n'.format(clean_app_data.DEF_30_CNT_SOCIAL_CIRCLE[clean_app_data.DEF_30_CNT_SOCIAL_CIRCLE <= 
                                        clean_app_data.DEF_30_CNT_SOCIAL_CIRCLE.quantile(0.99)].describe()))
print(clean_app_data.DEF_60_CNT_SOCIAL_CIRCLE[clean_app_data.DEF_60_CNT_SOCIAL_CIRCLE <= 
                                        clean_app_data.DEF_60_CNT_SOCIAL_CIRCLE.quantile(0.99)].describe())


# - **Observation:** Till 99th quantile there isn't any significant impact.

# #### - Plotting the below columns taking data points lying within 99th quantile

# In[17]:


plt.figure(figsize=(20,6))
plt.subplot(2,2,1)
sns.boxplot(clean_app_data.OBS_30_CNT_SOCIAL_CIRCLE[clean_app_data.OBS_30_CNT_SOCIAL_CIRCLE <= 
                                        clean_app_data.OBS_30_CNT_SOCIAL_CIRCLE.quantile(0.99)])
plt.subplot(2,2,2)
sns.boxplot(clean_app_data.OBS_60_CNT_SOCIAL_CIRCLE[clean_app_data.OBS_60_CNT_SOCIAL_CIRCLE <= 
                                        clean_app_data.OBS_60_CNT_SOCIAL_CIRCLE.quantile(0.99)])
plt.subplot(2,2,3)
sns.boxplot(clean_app_data.DEF_30_CNT_SOCIAL_CIRCLE[clean_app_data.DEF_30_CNT_SOCIAL_CIRCLE <= 
                                        clean_app_data.DEF_30_CNT_SOCIAL_CIRCLE.quantile(0.99)])
plt.subplot(2,2,4)
sns.boxplot(clean_app_data.DEF_60_CNT_SOCIAL_CIRCLE[clean_app_data.DEF_60_CNT_SOCIAL_CIRCLE <= 
                                        clean_app_data.DEF_60_CNT_SOCIAL_CIRCLE.quantile(0.99)])
plt.show()


# **Observation:**
# - The null values can be imputed using mean after treating the outliers for the columns OBS_30_CNT_SOCIAL_CIRCLE and OBS_60_CNT_SOCIAL_CIRCLE. 
# - The null values can be imputed with 0 for columns DEF_30_CNT_SOCIAL_CIRCLE and DEF_60_CNT_SOCIAL_CIRCLE

# #### - Checking the unique categories and top category in Categorical data and also plotting the same  

# In[18]:


clean_app_data.NAME_TYPE_SUITE.describe(include=object)


# In[19]:


sns.countplot(clean_app_data.NAME_TYPE_SUITE)
plt.xticks(rotation=90)
plt.show()


# - **Observation:** The null values can be imputed with `Unaccompanied` as it has highest frequency

# #### - Statistical description of below column

# In[20]:


clean_app_data.AMT_GOODS_PRICE.describe()


# #### - Plotting the below columns to see the spread of data

# In[21]:


sns.boxplot(clean_app_data.AMT_GOODS_PRICE)
plt.xticks(rotation=45)
plt.show()


# - **Observation:** Since the outliers is found to be in the higher quantile, it would be good to clip at the 95th quantile.

# #### - Statistical description taking data points lying within 95th quantile

# In[22]:


clean_app_data.AMT_GOODS_PRICE.clip_upper(clean_app_data.AMT_GOODS_PRICE.quantile(0.95)).describe()


# - **Observation:** The null values can be imputed using mean

# ### Conversion or formating the columns
# #### - Converting the data type of columns or format of data for selected columns

# In[23]:


clean_app_data.CNT_FAM_MEMBERS.fillna(0, inplace=True)

clean_app_data.CNT_FAM_MEMBERS = clean_app_data.CNT_FAM_MEMBERS.astype('int64')


# #### - Derived column AGE from DAYS_BIRTH column

# In[24]:


clean_app_data['AGE'] = round(abs(clean_app_data.DAYS_BIRTH)/365.25, 2)
clean_app_data.head()


# #### - Creating new column LOAN_PERIOD from AMT_CREDIT and AMT_ANNUITY

# In[25]:


clean_app_data['LOAN_PERIOD'] = round(clean_app_data.AMT_CREDIT/clean_app_data.AMT_ANNUITY, 4)
clean_app_data.head()


# ### Outlier treatment
# #### - Imputing the AMT_ANNUITY column using suitable methods then Treating the outliers

# In[26]:


clean_app_data.AMT_ANNUITY.describe()


# #### -  Imputing using mean value

# In[27]:


clean_app_data.AMT_ANNUITY.fillna(clean_app_data.AMT_ANNUITY.mean(),inplace=True)


# In[28]:


sns.boxplot(clean_app_data.AMT_ANNUITY)
plt.show()


# - **Observation:** There are huge number of outliers which can be treated

# #### - Treating the outliers using IQR method

# In[29]:


q1_annuity=clean_app_data.AMT_ANNUITY.quantile(0.25)
q3_annuity=clean_app_data.AMT_ANNUITY.quantile(0.75)
iqr_annuity=q3_annuity-q1_annuity
annuity_low=q1_annuity-1.5*iqr_annuity
annuity_high=q3_annuity+1.5*iqr_annuity


# #### - Clippping outliers instead of removing them so that the outliers influence would be limited to the maximum of the clip percentile values

# In[30]:


clean_app_data.AMT_ANNUITY = clean_app_data.AMT_ANNUITY.clip(annuity_low,annuity_high)
sns.boxplot(clean_app_data.AMT_ANNUITY)
plt.show()


# - **Observation:** Outliers are handled after clipping

# #### - Imputing the AMT_INCOME_TOTAL column using suitable methods then Treating the outliers

# In[31]:


clean_app_data.AMT_INCOME_TOTAL.describe()


# In[32]:


sns.boxplot(clean_app_data.AMT_INCOME_TOTAL)
plt.show()


# - **Observation:** The outlier has huge impact on the column data values

# #### - Treating the outliers using IQR method

# In[33]:


q1_income=clean_app_data.AMT_INCOME_TOTAL.quantile(0.25)
q3_income=clean_app_data.AMT_INCOME_TOTAL.quantile(0.75)
iqr_income=q3_income-q1_income
income_low=q1_income-1.5*iqr_income
income_high=q3_income+1.5*iqr_income


# #### - Clippping outliers instead of removing them so that the outliers influence would be limited to the maximum of the clip percentile values

# In[34]:


clean_app_data.AMT_INCOME_TOTAL = clean_app_data.AMT_INCOME_TOTAL.clip(income_low,income_high)
sns.boxplot(clean_app_data.AMT_INCOME_TOTAL)
plt.show()


# - **Observation:** The outlier has been handled after clipping

# #### - Imputing the AMT_CREDIT column using suitable methods then Treating the outliers

# In[35]:


clean_app_data.AMT_CREDIT.describe()


# In[36]:


sns.boxplot(clean_app_data.AMT_CREDIT)
plt.xticks(rotation=45)
plt.show()


# - **Observation:** There are huge number of outliers which can be treated

# #### - Treating the outliers using IQR method

# In[37]:


q1_credit=clean_app_data.AMT_CREDIT.quantile(0.25)
q3_credit=clean_app_data.AMT_CREDIT.quantile(0.75)
iqr_credit=q3_credit-q1_credit
credit_low=q1_credit-1.5*iqr_credit
credit_high=q3_credit+1.5*iqr_credit


# #### - Clippping outliers instead of removing them so that the outliers influence would be limited to the maximum of the clip percentile values

# In[38]:


clean_app_data.AMT_CREDIT = clean_app_data.AMT_CREDIT.clip(credit_low,credit_high)
sns.boxplot(clean_app_data.AMT_CREDIT)
plt.xticks(rotation=45)
plt.show()


# - **Observation:** The outliers are treated after clipping

# #### - As mentioned previously the outliers in AMT_GOODS_PRICE column are treating by clipping to 95th quantile

# In[39]:


clean_app_data.AMT_GOODS_PRICE=clean_app_data.AMT_GOODS_PRICE.clip_upper(clean_app_data.AMT_GOODS_PRICE.quantile(0.95))


# ### Binning the Continuous variables
# #### - Binning the AGE column with bin size being 10

# In[40]:


clean_app_data.AGE.describe()


# In[41]:


bins = [20,30,40,50,60,70]
labels = ['21-30','31-40','41-50','51-60','61-70']
clean_app_data['AGE_BIN'] = pd.cut(clean_app_data['AGE'], bins=bins, labels=labels)
clean_app_data.head()


# #### - Binning the LOAN_PERIOD column with bin size being 5

# In[42]:


clean_app_data.LOAN_PERIOD.describe()


# In[43]:


bins_loan = [i for i in range(5,56,5)]
labels_loan = ['6-10','11-15','16-20','21-25','26-30','31-35','36-40','41-45','46-50','51-55']
clean_app_data['LOAN_PERIOD_BIN'] = pd.cut(clean_app_data['LOAN_PERIOD'], bins=bins_loan, labels=labels_loan)
clean_app_data.head()


# ### Balance Check and Splitting of dataframe
# #### - Imbalance percentage

# In[44]:


print('Percentage of Target=1 : {}\n'
      .format((clean_app_data.TARGET[clean_app_data.TARGET==1].count()/clean_app_data.TARGET.count())*100))
print('Percentage of Target=0 : {}'
      .format((clean_app_data.TARGET[clean_app_data.TARGET==0].count()/clean_app_data.TARGET.count())*100))


# #### -Splitting the Dataframe based on TARGET value

# In[45]:


app_data_tgt_1=clean_app_data[clean_app_data.TARGET==1]


# In[85]:


app_data_tgt_1.shape


# In[46]:


app_data_tgt_0=clean_app_data[clean_app_data.TARGET==0]


# In[86]:


app_data_tgt_0.shape


# ### Univariate Analysis
# #### - Univariate analysis for categorical variables for both 0 and 1

# In[47]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(app_data_tgt_1.NAME_CONTRACT_TYPE)
plt.title('For Target= 1')
plt.subplot(1,2,2)
sns.countplot(app_data_tgt_0.NAME_CONTRACT_TYPE)
plt.title('For Target= 0')
plt.show()


# - **Observation:** The contract type with `cash loans` are preferred more irrespective whether the applicants are defaulters or not

# In[48]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(app_data_tgt_1.NAME_INCOME_TYPE,order=app_data_tgt_1.NAME_INCOME_TYPE.value_counts().index)
plt.title('For Target= 1')
plt.xticks(rotation=90)
plt.subplot(1,2,2)
sns.countplot(app_data_tgt_0.NAME_INCOME_TYPE,order=app_data_tgt_0.NAME_INCOME_TYPE.value_counts().index)
plt.title('For Target= 0')
plt.xticks(rotation=90)
plt.show()


# - **Observation:** From the above, we can see that irrespective of the target, `working` class applied the highest for loan

# In[49]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(app_data_tgt_1.NAME_EDUCATION_TYPE,order=app_data_tgt_1.NAME_EDUCATION_TYPE.value_counts().index)
plt.title('For Target= 1')
plt.xticks(rotation=90)
plt.subplot(1,2,2)
sns.countplot(app_data_tgt_0.NAME_EDUCATION_TYPE,order=app_data_tgt_0.NAME_EDUCATION_TYPE.value_counts().index)
plt.title('For Target= 0')
plt.xticks(rotation=90)
plt.show()


# - **Observation:** From the above, we can see that irrespective of the target, `Secondary education` class applied the highest for loan

# In[50]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(app_data_tgt_1.NAME_FAMILY_STATUS,order=app_data_tgt_1.NAME_FAMILY_STATUS.value_counts().index)
plt.title('For Target= 1')
plt.xticks(rotation=90)
plt.subplot(1,2,2)
sns.countplot(app_data_tgt_0.NAME_FAMILY_STATUS,order=app_data_tgt_0.NAME_FAMILY_STATUS.value_counts().index)
plt.title('For Target= 0')
plt.xticks(rotation=90)
plt.show()


# - **Observation:** From the above, we can see that irrespective of the target, `married` applicants have applied the highest for loan

# In[51]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(app_data_tgt_1.NAME_HOUSING_TYPE,order=app_data_tgt_1.NAME_HOUSING_TYPE.value_counts().index)
plt.title('For Target= 1')
plt.xticks(rotation=90)
plt.subplot(1,2,2)
sns.countplot(app_data_tgt_0.NAME_HOUSING_TYPE,order=app_data_tgt_0.NAME_HOUSING_TYPE.value_counts().index)
plt.title('For Target= 0')
plt.xticks(rotation=90)
plt.show()


# - **Observation:** From the above, we can see that irrespective of the target, applicants owning a `house/apartment` have applied the highest for loan

# In[52]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(app_data_tgt_1.WEEKDAY_APPR_PROCESS_START,order=app_data_tgt_1.WEEKDAY_APPR_PROCESS_START.value_counts().index)
plt.title('For Target= 1')
plt.xticks(rotation=90)
plt.subplot(1,2,2)
sns.countplot(app_data_tgt_0.WEEKDAY_APPR_PROCESS_START,order=app_data_tgt_0.WEEKDAY_APPR_PROCESS_START.value_counts().index)
plt.title('For Target= 0')
plt.xticks(rotation=90)
plt.show()


# - **Observation:** From the above, we can see that irrespective of the target, most of the application has been started for processing on `Tuesday`

# #### - Univariate for numerical variables for both 0 and 1

# In[53]:


plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.boxplot(app_data_tgt_1.AMT_INCOME_TOTAL)
plt.title('For Target= 1')
plt.xticks(rotation=90)
plt.subplot(1,2,2)
sns.boxplot(app_data_tgt_0.AMT_INCOME_TOTAL)
plt.title('For Target= 0')
plt.xticks(rotation=90)
plt.show()


# - **Observation:** From the above, we can see that irrespective of the target, the `income total` of applicants are in same range

# In[54]:


plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
sns.boxplot(app_data_tgt_1.AMT_CREDIT)
plt.title('For Target= 1')
plt.xticks(rotation=90)
plt.subplot(1,2,2)
sns.boxplot(app_data_tgt_0.AMT_CREDIT)
plt.title('For Target= 0')
plt.xticks(rotation=90)
plt.show()


# - **Observation:** From the above, we can see that irrespective of the target, the `credit` of applicants are in same range

# In[55]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(app_data_tgt_1.AGE_BIN)
plt.title('For Target= 1')
plt.xticks(rotation=90)
plt.subplot(1,2,2)
sns.countplot(app_data_tgt_0.AGE_BIN)
plt.title('For Target= 0')
plt.xticks(rotation=90)
plt.show()


# - **Observation:** From the above, we can see that irrespective of the target, the `age group` applying for loan is similar

# In[56]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.countplot(app_data_tgt_1.LOAN_PERIOD_BIN)
plt.title('For Target= 1')
plt.xticks(rotation=90)
plt.subplot(1,2,2)
sns.countplot(app_data_tgt_0.LOAN_PERIOD_BIN)
plt.title('For Target= 0')
plt.xticks(rotation=90)
plt.show()


# - **Observation:** From the above, we can see that irrespective of the target, the `loan period` is similar

# ### Correlation
# #### - Correlation for numerical columns for TARGET=0

# In[57]:


round(app_data_tgt_0.corr(),2)


# In[58]:


plt.figure(figsize=(40,40))
sns.heatmap(round(app_data_tgt_0.corr(),2), cmap='Reds', annot=True)
plt.show()


# #### - Correlation for numerical columns for TARGET=1

# In[59]:


round(app_data_tgt_1.corr(),2)


# In[60]:


plt.figure(figsize=(40,40))
sns.heatmap(round(app_data_tgt_1.corr(),2), cmap='Blues', annot=True)
plt.show()


# **Observation:** Both 0 and 1 have same variables with highest correlation
# - AMT_CREDIT, AMT_ANNUITY and AMT_GOODS_PRICE are highly correlated to each other
# - LOAN_PERIOD, AMT_CREDIT and AMT_GOODS_PRICE are correlated to each other

# ### Bivariate Analysis
# #### - Taking AMT_CREDIT, AMT_ANNUITY columns to analyse for TARGET=0

# In[61]:


plt.figure(figsize=(10,6))
plt.scatter(x='AMT_CREDIT', y='AMT_ANNUITY', data=app_data_tgt_0)
plt.xlabel('AMT_CREDIT')
plt.ylabel('AMT_ANNUITY')
plt.show()


# #### - Taking AMT_CREDIT, AMT_ANNUITY columns to analyse for TARGET=1

# In[62]:


plt.figure(figsize=(10,6))
plt.scatter(x='AMT_CREDIT', y='AMT_ANNUITY', data=app_data_tgt_1)
plt.xlabel('AMT_CREDIT')
plt.ylabel('AMT_ANNUITY')
plt.show()


# - **Observation:** From the above, we can see for both 0 and 1 `credit` and `annuity` varies in similar fashion

# #### - Taking AMT_CREDIT, AMT_GOODS_PRICE columns to analyse for TARGET=0

# In[63]:


plt.figure(figsize=(10,6))
plt.scatter(x='AMT_CREDIT', y='AMT_GOODS_PRICE', data=app_data_tgt_0)
plt.xlabel('AMT_CREDIT')
plt.ylabel('AMT_GOODS_PRICE')
plt.show()


# #### - Taking AMT_CREDIT, AMT_GOODS_PRICE columns to analyse for TARGET=1

# In[64]:


plt.figure(figsize=(10,6))
plt.scatter(x='AMT_CREDIT', y='AMT_GOODS_PRICE', data=app_data_tgt_1)
plt.xlabel('AMT_CREDIT')
plt.ylabel('AMT_GOODS_PRICE')
plt.show()


# - **Observation:** From the above, we can see for both 0 and 1 `credit` and `goods price` varies in similar fashion

# #### - Taking AMT_GOODS_PRICE, AMT_ANNUITY columns to analyse for TARGET=0

# In[65]:


plt.figure(figsize=(10,6))
plt.scatter(x='AMT_ANNUITY', y='AMT_GOODS_PRICE', data=app_data_tgt_0)
plt.xlabel('AMT_ANNUITY')
plt.ylabel('AMT_GOODS_PRICE')
plt.show()


# #### - Taking AMT_GOODS_PRICE, AMT_ANNUITY columns to analyse for TARGET=1

# In[66]:


plt.figure(figsize=(10,6))
plt.scatter(x='AMT_ANNUITY', y='AMT_GOODS_PRICE', data=app_data_tgt_1)
plt.xlabel('AMT_ANNUITY')
plt.ylabel('AMT_GOODS_PRICE')
plt.show()


# - **Observation:** From the above, we can see for both 0 and 1 `annuity` and `goods price` varies in similar fashion

# #### - Taking AMT_GOODS_PRICE, LOAN_PERIOD columns to analyse for TARGET=0

# In[67]:


plt.figure(figsize=(10,6))
plt.scatter(x='AMT_GOODS_PRICE', y='LOAN_PERIOD', data=app_data_tgt_0)
plt.xlabel('AMT_GOODS_PRICE')
plt.ylabel('LOAN_PERIOD')
plt.show()


# #### - Taking AMT_GOODS_PRICE, LOAN_PERIOD columns to analyse for TARGET=1

# In[68]:


plt.figure(figsize=(10,6))
plt.scatter(x='AMT_GOODS_PRICE', y='LOAN_PERIOD', data=app_data_tgt_1)
plt.xlabel('AMT_GOODS_PRICE')
plt.ylabel('LOAN_PERIOD')
plt.show()


# - **Observation:** From the above, we can see for both 0 and 1 `loan period` and `goods price` varies in similar fashion

# #### - Taking AMT_CREDIT, LOAN_PERIOD columns to analyse for TARGET=0

# In[69]:


plt.figure(figsize=(10,6))
plt.scatter(x='AMT_CREDIT', y='LOAN_PERIOD', data=app_data_tgt_0)
plt.xlabel('AMT_CREDIT')
plt.ylabel('LOAN_PERIOD')
plt.show()


# #### - Taking AMT_CREDIT, LOAN_PERIOD columns to analyse for TARGET=1

# In[70]:


plt.figure(figsize=(10,6))
plt.scatter(x='AMT_CREDIT', y='LOAN_PERIOD', data=app_data_tgt_1)
plt.xlabel('AMT_CREDIT')
plt.ylabel('LOAN_PERIOD')
plt.show()


# - **Observation:** From the above, we can see for both 0 and 1 `loan period` and `credit` varies in similar fashion

# ### Analysing previous and current data
# #### - Reading the previous_application.csv

# In[71]:


previous_application = pd.read_csv('previous_application.csv')
previous_application.head()


# #### - Shape of the previous data

# In[72]:


previous_application.shape


# #### - Information about the previous data

# In[73]:


previous_application.info()


# #### - Statistical description of the previous data

# In[74]:


previous_application.describe()


# #### - Merging the previous data and current data

# In[75]:


merged_data=previous_application.merge(clean_app_data,how='outer',on='SK_ID_CURR')
merged_data.head()


# #### - Shape of the merged data set

# In[76]:


merged_data.shape


# #### - Univarate analysis on selected columns

# In[77]:


sns.countplot(merged_data.NAME_CONTRACT_STATUS)
plt.show()


# - **Observation:** Most of the previous application are in `approved` status

# In[78]:


sns.countplot(merged_data.NAME_CLIENT_TYPE)
plt.show()


# - **Observation:** Most of the previous application are of `repeaters`

# In[79]:


plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(merged_data.NAME_CONTRACT_TYPE_x)
plt.xlabel('NAME_CONTRACT_TYPE in Previous')
plt.subplot(1,2,2)
sns.countplot(merged_data.NAME_CONTRACT_TYPE_y)
plt.xlabel('NAME_CONTRACT_TYPE in Current')
plt.show()


# - **Observation:** In both previous and current data, `cash loans` are high

# #### - Correlation of merged data

# In[80]:


plt.figure(figsize=(40,40))
sns.heatmap(round(merged_data.corr(),2), annot=True, cmap='Greens')


# **Observation:** Few of the columns are highly correlated. The columns being:
# - DAYS_LAST_DUE and DAYS_TERMINATION
# - AMT_ANNUITY_x, AMT_APPLICATION, AMT_CREDIT_x and AMT_GOODS_PRICE_x
# - AMT_ANNUITY_y, AMT_CREDIT_y and AMT_GOODS_PRICE_y

# #### - Bivariant analysis

# In[81]:


plt.figure(figsize=(8,4))
plt.scatter(previous_application.AMT_CREDIT, previous_application.AMT_APPLICATION)
plt.xlabel('AMT_CREDIT')
plt.ylabel('AMT_APPLICATION')
plt.show()


# - **Observation:** The expected `loan amount` was not received everytime

# In[82]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(previous_application.AMT_GOODS_PRICE, previous_application.AMT_APPLICATION)
plt.xlabel('AMT_GOODS_PRICE')
plt.ylabel('AMT_APPLICATION')
plt.subplot(1,2,2)
plt.scatter(previous_application.AMT_GOODS_PRICE, previous_application.AMT_CREDIT)
plt.xlabel('AMT_GOODS_PRICE')
plt.ylabel('AMT_CREDIT')
plt.show()


# - **Observation:** From above, we can say that `AMT_APPLICATION/AMT_CREDIT` and `AMT_GOODS_PRICE` are linearly increasing

# In[83]:


plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(previous_application.AMT_GOODS_PRICE, previous_application.AMT_ANNUITY)
plt.xlabel('AMT_GOODS_PRICE')
plt.ylabel('AMT_ANNUITY')
plt.subplot(1,2,2)
plt.scatter(previous_application.AMT_CREDIT, previous_application.AMT_ANNUITY)
plt.xlabel('AMT_CREDIT')
plt.ylabel('AMT_ANNUITY')
plt.show()


# - **Observation:** From the previous graph we concluded that `AMT_CREDIT` and `AMT_GOODS_PRICE` are directly proportional which is the same in case of `AMT_ANNUITY` as well

# In[84]:


sns.barplot(merged_data.NAME_CONTRACT_STATUS, merged_data.AMT_APPLICATION, hue=merged_data.TARGET)
plt.show()


# - **Observation:** We can see that loan has been `approved` for the `defaulters` in previous application and most of the `non-defaulters` were `refused`.

# #### So based on the above explained insights, the bank can restructure its plans so that it can result in more profit and gains by approving loans to non-defaulting clients and making sure less credit loans are approved to defaulters or defaulters loan application are refused.

# In[ ]:




