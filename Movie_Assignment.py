
# coding: utf-8

# ## Problem Statement
# 
# ### Trying to find some interesting insights into a few movies released between 1916 and 2016, using Python. Explore the data, gain insights into the movies, actors, directors, and collections, and submit the code.

# In[1]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Import the matplotlib, seaborn, numpy and pandas packages

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ## Task 1: Reading and Inspection
# 
# -  ### Subtask 1.1: Import and read
# 
# Import and read the movie database. Store it in a variable called `movies`.

# In[3]:


# Write your code for importing the csv file here

movies = pd.read_csv('Movie_Assignment_Data.csv')
movies.head()


# -  ### Subtask 1.2: Inspect the dataframe
# 
# Inspect the dataframe's columns, shapes, variable types etc.

# **Columns in dataframe :**

# In[4]:


# Write your code for inspection here

movies.columns


# **Shape of dataframe :**

# In[5]:


movies.shape


# **Variable types in dataframe**

# In[6]:


movies.dtypes


# **Information of the dataframe about the data types, number of columns and rows, number of non-null columns, memory column names :**

# In[7]:


movies.info()


# **Statistical description of dataframe :**

# In[8]:


movies.describe()


# ## Task 2: Cleaning the Data
# 
# -  ### Subtask 2.1: Inspect Null values
# 
# Find out the number of Null values in all the columns and rows. Also, find the percentage of Null values in each column. Round off the percentages upto two decimal places.

# In[9]:


# Write your code for column-wise null count here

movies.isnull().sum()


# - **Observation :** We can see that gross, budget, aspect_ratio and content_rating have more number of null values where gross being highest

# In[10]:


# Write your code for row-wise null count here

movies.isnull().sum(axis=1)


# - **Observation :** There are few rows which have null values

# In[11]:


# Write your code for column-wise null percentages here

round(100*(movies.isnull().sum()/len(movies.index)),2)


# - **Observation :** gross has highest percentage of null values.

# -  ### Subtask 2.2: Drop unecessary columns
# 
# For this assignment, you will mostly be analyzing the movies with respect to the ratings, gross collection, popularity of movies, etc. So many of the columns in this dataframe are not required. So it is advised to drop the following columns.
# -  color
# -  director_facebook_likes
# -  actor_1_facebook_likes
# -  actor_2_facebook_likes
# -  actor_3_facebook_likes
# -  actor_2_name
# -  cast_total_facebook_likes
# -  actor_3_name
# -  duration
# -  facenumber_in_poster
# -  content_rating
# -  country
# -  movie_imdb_link
# -  aspect_ratio
# -  plot_keywords

# In[12]:


# Write your code for dropping the columns here. It is advised to keep inspecting the dataframe after each set of operations 

col_drop = ['color','director_facebook_likes','actor_1_facebook_likes','actor_2_facebook_likes',
            'actor_3_facebook_likes','actor_2_name','cast_total_facebook_likes','actor_3_name','duration',
            'facenumber_in_poster','content_rating','country','movie_imdb_link','aspect_ratio','plot_keywords']

new_movies = movies.drop(columns=col_drop)


# **Columns in dataframe :**

# In[13]:


# Write your code for inspection here

new_movies.columns


# **Shape of dataframe :**

# In[14]:


new_movies.shape


# **Variable types in dataframe**

# In[15]:


new_movies.dtypes


# **Information of the dataframe about the data types, number of columns and rows, number of non-null columns, memory column names :**

# In[16]:


new_movies.info()


# **Null percentage of columns :**

# In[17]:


round(100*(new_movies.isnull().sum()/len(new_movies.index)),2)


# -  ### Subtask 2.3: Drop unecessary rows using columns with high Null percentages
# 
# Now, on inspection you might notice that some columns have large percentage (greater than 5%) of Null values. Drop all the rows which have Null values for such columns.

# In[18]:


# Write your code for dropping the rows here

null_columns=new_movies.columns[new_movies.isnull().any()]
new_movies.dropna(subset=null_columns, inplace=True)


# **Columns in dataframe :**

# In[19]:


# Write your code for inspection here

new_movies.columns


# **Shape of dataframe :**

# In[20]:


new_movies.shape


# **Null percentage of columns :**

# In[21]:


round(100*(new_movies.isnull().sum()/len(new_movies.index)),2)


# -  ### Subtask 2.4: Fill NaN values
# 
# You might notice that the `language` column has some NaN values. Here, on inspection, you will see that it is safe to replace all the missing values with `'English'`.

# In[22]:


# Write your code for filling the NaN values in the 'language' column here

new_movies.loc[pd.isnull(new_movies['language']), ['language']] = 'English'


# -  ### Subtask 2.5: Check the number of retained rows
# 
# You might notice that two of the columns viz. `num_critic_for_reviews` and `actor_1_name` have small percentages of NaN values left. You can let these columns as it is for now. Check the number and percentage of the rows retained after completing all the tasks above.

# In[23]:


# Write your code for checking number of retained rows here

newRow = len(new_movies.index)
print('Number of Rows : {}'.format(newRow))
oldRow = len(movies.index)
centrow = 100*newRow/oldRow
print('Percentage of rows left : {}'.format(centrow))


# **Checkpoint 1:** You might have noticed that we still have around `77%` of the rows!

# - Number of rows left after removing null values is 3884 and the percentage is 77.02%

# ## Task 3: Data Analysis
# 
# -  ### Subtask 3.1: Change the unit of columns
# 
# Convert the unit of the `budget` and `gross` columns from `$` to `million $`.

# In[24]:


# Write your code for unit conversion here

new_movies['gross'] = new_movies['gross']/1000000
new_movies['budget'] = new_movies['budget']/1000000


# -  ### Subtask 3.2: Find the movies with highest profit
# 
#     1. Create a new column called `profit` which contains the difference of the two columns: `gross` and `budget`.
#     2. Sort the dataframe using the `profit` column as reference.
#     3. Plot `profit` (y-axis) vs `budget` (x- axis) and observe the outliers using the appropriate chart type.
#     4. Extract the top ten profiting movies in descending order and store them in a new dataframe - `top10`

# In[25]:


# Write your code for creating the profit column here

new_movies['profit'] = new_movies['gross'] - new_movies['budget']
new_movies


# In[26]:


# Write your code for sorting the dataframe here

new_movies.sort_values(by='profit', ascending=False)


# In[27]:


# Write code for profit vs budget plot here

sns.jointplot(x='budget', y='profit', data=new_movies, color='g')


# - **Observation :** We can see that lower budget movies have done well in box office than higher budget movies

# In[28]:


# Write your code to get the top 10 profiting movies here

top10 = new_movies.sort_values(by='profit', ascending=False).head(10)
top10


# -  ### Subtask 3.3: Drop duplicate values
# 
# After you found out the top 10 profiting movies, you might have noticed a duplicate value. So, it seems like the dataframe has duplicate values as well. Drop the duplicate values from the dataframe and repeat `Subtask 3.2`. Note that the same `movie_title` can be there in different languages. 

# In[29]:


# Write your code for dropping duplicate values here

no_duplicate = new_movies.drop_duplicates(subset='movie_title')
no_duplicate.head()


# **Columns in dataframe :**

# In[30]:


# Write your code for inspection here

no_duplicate.columns


# **Shape of dataframe :**

# In[31]:


no_duplicate.shape


# **Null percentage of columns :**

# In[32]:


round(100*(no_duplicate.isnull().sum()/len(no_duplicate.index)),2)


# In[33]:


# Write code for repeating subtask 2 here
# sorting dataframe using profit columnn

no_duplicate.sort_values(by='profit', ascending=False)


# In[34]:


# Write code for profit vs budget plot here

sns.jointplot(x='budget', y='profit', data=no_duplicate)


# - **Observation :** We can see that lower budget movies have done well in box office than higher budget movies

# In[35]:


# Write your code to get the top 10 profiting movies here

top_10 = no_duplicate.sort_values(by='profit', ascending=False).head(10)
top_10


# **Checkpoint 2:** You might spot two movies directed by `James Cameron` in the list.
# - YES can see `James Cameron` twice in top 10 list

# -  ### Subtask 3.4: Find IMDb Top 250
# 
#     1. Create a new dataframe `IMDb_Top_250` and store the top 250 movies with the highest IMDb Rating (corresponding to the column: `imdb_score`). Also make sure that for all of these movies, the `num_voted_users` is greater than 25,000.
# Also add a `Rank` column containing the values 1 to 250 indicating the ranks of the corresponding films.
#     2. Extract all the movies in the `IMDb_Top_250` dataframe which are not in the English language and store them in a new dataframe named `Top_Foreign_Lang_Film`.

# In[36]:


# Write your code for extracting the top 250 movies as per the IMDb score here. Make sure that you store it in a new dataframe 
# and name that dataframe as 'IMDb_Top_250'

IMDb_Top_250 = no_duplicate.sort_values(by=['imdb_score'], ascending=False)
IMDb_Top_250 = IMDb_Top_250[IMDb_Top_250.num_voted_users>25000].head(250)
IMDb_Top_250['Rank'] = np.arange(1,len(IMDb_Top_250.index)+1)
IMDb_Top_250.set_index('Rank', inplace=True)
IMDb_Top_250


# In[37]:


# Write your code to extract top foreign language films from 'IMDb_Top_250' here

Top_Foreign_Lang_Film = IMDb_Top_250[IMDb_Top_250.language.ne('English')]
Top_Foreign_Lang_Film


# **Checkpoint 3:** Can you spot `Veer-Zaara` in the dataframe?
# - Yes, `Veer-Zaara` is in the IMDB top 250 non English movies

# - ### Subtask 3.5: Find the best directors
# 
#     1. Group the dataframe using the `director_name` column.
#     2. Find out the top 10 directors for whom the mean of `imdb_score` is the highest and store them in a new dataframe `top10director`.  Incase of a tie in IMDb score between two directors, sort them alphabetically. 

# In[38]:


# Write your code for extracting the top 10 directors here

top10director = no_duplicate.groupby(by=['director_name']).mean()
top10director = top10director.sort_values(by='imdb_score', ascending=False).head(10)
top10director


# **Checkpoint 4:** No surprises that `Damien Chazelle` (director of Whiplash and La La Land) is in this list.
# - Yes, `Damien Chazelle` (director of Whiplash and La La Land) is part of the top 10 directors.

# -  ### Subtask 3.6: Find popular genres
# 
# You might have noticed the `genres` column in the dataframe with all the genres of the movies seperated by a pipe (`|`). Out of all the movie genres, the first two are most significant for any film.
# 
# 1. Extract the first two genres from the `genres` column and store them in two new columns: `genre_1` and `genre_2`. Some of the movies might have only one genre. In such cases, extract the single genre into both the columns, i.e. for such movies the `genre_2` will be the same as `genre_1`.
# 2. Group the dataframe using `genre_1` as the primary column and `genre_2` as the secondary column.
# 3. Find out the 5 most popular combo of genres by finding the mean of the gross values using the `gross` column and store them in a new dataframe named `PopGenre`.

# In[39]:


# Write your code for extracting the first two genres of each movie here

no_duplicate['genre_1'] = no_duplicate['genres'].str.split('|', expand=True)[0]
no_duplicate['genre_2'] = no_duplicate['genres'].str.split('|', expand=True)[1]

no_duplicate.genre_2.fillna(no_duplicate.genre_1, inplace=True)
no_duplicate


# In[40]:


# Write your code for grouping the dataframe here

movies_by_segment = no_duplicate.groupby(by=['genre_1','genre_2'])


# In[41]:


# Write your code for getting the 5 most popular combo of genres here

PopGenre = movies_by_segment.mean()
PopGenre = PopGenre.sort_values(by='gross', ascending=False).head(5)
PopGenre


# **Checkpoint 5:** Well, as it turns out. `Family + Sci-Fi` is the most popular combo of genres out there!
# - Yes, the most popular combo of geners is `Family + Sci-Fi`

# -  ### Subtask 3.7: Find the critic-favorite and audience-favorite actors
# 
#     1. Create three new dataframes namely, `Meryl_Streep`, `Leo_Caprio`, and `Brad_Pitt` which contain the movies in which the actors: 'Meryl Streep', 'Leonardo DiCaprio', and 'Brad Pitt' are the lead actors. Use only the `actor_1_name` column for extraction. Also, make sure that you use the names 'Meryl Streep', 'Leonardo DiCaprio', and 'Brad Pitt' for the said extraction.
#     2. Append the rows of all these dataframes and store them in a new dataframe named `Combined`.
#     3. Group the combined dataframe using the `actor_1_name` column.
#     4. Find the mean of the `num_critic_for_reviews` and `num_users_for_review` and identify the actors which have the highest mean.
#     5. Observe the change in number of voted users over decades using a bar chart. Create a column called `decade` which represents the decade to which every movie belongs to. For example, the  `title_year`  year 1923, 1925 should be stored as 1920s. Sort the dataframe based on the column `decade`, group it by `decade` and find the sum of users voted in each decade. Store this in a new data frame called `df_by_decade`.

# In[42]:


# Write your code for creating three new dataframes here
# Include all movies in which Meryl_Streep is the lead

Meryl_Streep = no_duplicate[no_duplicate.actor_1_name == 'Meryl Streep']


# In[43]:


# Include all movies in which Leo_Caprio is the lead

Leo_Caprio = no_duplicate[no_duplicate.actor_1_name == 'Leonardo DiCaprio']


# In[44]:


# Include all movies in which Brad_Pitt is the lead

Brad_Pitt = no_duplicate[no_duplicate.actor_1_name == 'Brad Pitt']


# In[45]:


# Write your code for combining the three dataframes here

Combined = pd.concat([Meryl_Streep,Leo_Caprio,Brad_Pitt])


# In[46]:


# Write your code for grouping the combined dataframe here

Combined = Combined.groupby(by='actor_1_name')


# In[47]:


# Write the code for finding the mean of critic reviews and audience reviews here

Combined[['num_critic_for_reviews','num_user_for_reviews']].mean().sort_values(by=['num_critic_for_reviews','num_user_for_reviews'],ascending=False)


# **Checkpoint 6:** `Leonardo` has aced both the lists!
# - Yes, `Leonardo` has highest number of critic reviews and user reviews

# In[48]:


# Write the code for calculating decade here

no_duplicate['decade'] = pd.DataFrame(no_duplicate['title_year']//10*10, dtype=int)
no_duplicate.head()


# In[49]:


# Write your code for creating the data frame df_by_decade here

df_by_decade = no_duplicate.sort_values(by='decade').groupby(by=['decade'])
df_by_decade = df_by_decade[['num_voted_users']].sum()
df_by_decade.reset_index()


# In[50]:


# Write your code for plotting number of voted users vs decade

sns.barplot(x='decade', y='num_voted_users', data=no_duplicate, palette='rocket', estimator=sum)


# - We could see that 200 has highest number of voted users over the decades.
# - We could observe that there is increase in voted users from 1920 to 1930 and then there is very drastic drop in voted users in 1940.
# - From 1940 is a steady increase in voted users until 1980.
# - Then there is a steep increase in voted users in 1990 and also steep increase in 2000.
# - There is a slight drop in the voted users in the year 2010.
