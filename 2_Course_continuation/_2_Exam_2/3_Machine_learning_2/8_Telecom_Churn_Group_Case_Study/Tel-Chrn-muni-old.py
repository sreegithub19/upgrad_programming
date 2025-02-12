#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn Prediction

#  > Developers - <br>
#  Munirathinam Duraisamy : munirathinamd1985@gmail.com , <br>
#  Sreedhar K : cpadmaja2003@gmail.com

# # Problem Statement
# 
# > In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.
# 
# > For many incumbent operators, retaining high profitable customers is the number one business
# goal. To reduce customer churn, telecom companies need to predict which customers are at high risk of churn. In this project, you will analyze customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn.
# 
# > In this competition, your goal is to build a machine learning model that is able to predict churning customers based on the features provided for their usage.
# 
# 
# # Problem data
# <br>
# <a href="https://www.kaggle.com/competitions/telecom-churn-case-study-hackathon-c41/overview">Competition link</a>
# </br>
# <br>
# <a href="https://www.kaggle.com/competitions/telecom-churn-case-study-hackathon-c41/data">Dataset</a>
# <br>
# <br>
# 
# 
# # Business Objective
# > The main goal of the case study is to build ML models to predict churn. The predictive model that you’re going to build will the following purposes:
# <br>
# 1) It will be used to predict whether a high-value customer will churn or not, in near future (i.e. churn phase). By knowing this, the company can take action steps such as providing special plans, discounts on recharge etc.
# <br>
# 2) It will be used to identify important variables that are strong predictors of churn. These variables may also indicate why customers choose to switch to other networks.
# <br>
# 3) Even though overall accuracy will be your primary evaluation metric, you should also mention other metrics like precision, recall, etc. for the different models that can be used for evaluation purposes based on different business objectives. For example, in this problem statement, one business goal can be to build an ML model that identifies customers who'll definitely churn with more accuracy as compared to the ones who'll not churn. Make sure you mention which metric can be used in such scenarios.
# <br>
# 4) Recommend strategies to manage customer churn based on your observations.

#  ## Steps:
#  
# 0. [EDA](#EDA)<br>
# <ul>
#     <li>Load library</li>
#     <li>Data Load</li>
#     <li>Data Overview</li>
#     <li>Metadata Information</li>
# </ul>
# 1. [Data_Cleaning_and_Missing_Data_Analysis](#Data_Cleaning_and_Missing_Data_Analysis)<br>
# 2. [Outlier Analysis & Treatment Assumption values > Q3+1.5IQR and values < Q1-1.5IQR will be treated](#Outlier_Analysis_and_Treatment_Assumption_values)<br>
# 3. [Transforming_Categorical_Columns](#Transforming_Categorical_Columns)<br>
# <ul>
#     <li>Filter High-Value Customers</li>
#     <ul>
#         <li>[calculate total data recharge amount](#Transforming_Categorical_Columns)</li>
#     </ul>
#     <li>Display the correlation matrix again to analyze correlation coefficient between features</li>
# </ul>
# 4. [Univariate_Analysis](#Univariate_Analysis)<br>
# 5. [Bivariate_Analysis](#Bivariate_Analysis)<br>
# 
# 
# > Model Preperation
# 
# - Training and Test data split
# - Feature Scaling - StandardScaler
# - Strategy steps
# - Handle Imbalance dataset using SMOTE
# - PCA - Dimensionality Reduction
# - Case1 : 
# - - Split train data into train and test split
# - - Created below models using Hyper Parameter Tuning
# - - - LOGISTICREGRESSION
# - - - RANDOMFOREST
# - - - ADABOOST
# - - - XGBBOOST
# - - - Made predictions by using combination of Random Forest + Adaboost + XGBOOST
# - Case2 : 
# - - Use entire train dataset for model building using K Cross Validation
# - - Created below models on entire train set
# - - - RANDOMFOREST
# - - - ADABOOST
# - - - XGBBOOST
# - - - Made predictions by using combination of Random Forest + Adaboost + XGBOOST
# - Model Evaluation & Assessment
# - Prediction
# - - - Made predictions on combination of case1 and case2 
# - - - Important Features
# - Conclusion & Analysis
# 
# <hr>
# <hr>

# <h1><a id='EDA'>EDA</a><br></h1>
# <ul>
#     <li>Load library</li>
#     <li>Data Load</li>
#     <li>Data Overview</li>
#     <li>Metadata Information</li>
# </ul>

# 
# # Import Libraries

# In[1]:


get_ipython().system('pip uninstall -y scikit-learn')
get_ipython().system('pip install scikit-learn')


# In[2]:


#Importing reqried libraries
import pandas as pd 
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from IPython.display import display,HTML

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold,StratifiedKFold
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import imblearn
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import sensitivity_specificity_support

import plotly.offline as py 
import plotly.graph_objs as go
import plotly.tools as tls


# # Data Load

# In[3]:


#Loading data into data frame
data = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")


# # Data Overview

# In[4]:


pd.set_option('display.max_columns',200)
pd.set_option('display.max_rows',200)


# In[5]:


#Displaying the first 10 records with all columns
data.head(10)


# In[6]:


data.describe()


# In[7]:


#Check the column data
data.info()


# In[8]:


#Get DF Shape
print ("Rows     : " ,data.shape[0])
print ("Columns  : " ,data.shape[1])


# # Metadata Information

# In[9]:


# List the Features
print (data.columns.tolist())


# In[10]:


#Check the missing values
data.isnull().sum().values.sum()


# In[11]:


#Check the unique values
data.nunique()


# In[12]:


# Number of rows have the all null values

print("Number of rows have null values in all columns= {} ".format(data.isnull().all(axis=1).sum()))


# In[13]:


# Number of colummns have the all null values

print("Number of colummns have the all null values = {}".format(data.isnull().all(axis=0).sum()))


# In[14]:


# Check Duplicate Rows

data[data.duplicated()]


# <hr>
# <h1><a id='Data_Cleaning_and_Missing_Data_Analysis'>Data Cleaning and Missing Data Analysis</a><br></h1>
# <ul>
#     <li>Data_Cleaning</li>
#     <li>Missing Data Treatment Analysis</li>
# </ul>

# In[15]:


# Lets do EDA and further activities without disturbing the orginal dataset.
# As we are doing data treatment, we will do for both train and test data
#Droping id as it has only unique values
# Dropping listed columns as these columns have only single value : 'circle_id', loc_og_t2o_mou','std_og_t2o_mou','loc_ic_t2o_mou','std_og_t2c_mou_6','std_og_t2c_mou_7','std_og_t2c_mou_8','std_ic_t2o_mou_6','std_ic_t2o_mou_7','std_ic_t2o_mou_8

test_v1 = test.drop(columns=['id','circle_id','loc_og_t2o_mou','std_og_t2o_mou','loc_ic_t2o_mou','std_og_t2c_mou_6','std_og_t2c_mou_7','std_og_t2c_mou_8','std_ic_t2o_mou_6',
                               'std_ic_t2o_mou_7','std_ic_t2o_mou_8'],axis=1)

data_v1 = data.drop(columns=['id','circle_id','loc_og_t2o_mou','std_og_t2o_mou','loc_ic_t2o_mou','std_og_t2c_mou_6','std_og_t2c_mou_7','std_og_t2c_mou_8','std_ic_t2o_mou_6',
                               'std_ic_t2o_mou_7','std_ic_t2o_mou_8'],axis=1)


# In[16]:


display(data_v1.shape)


# ##### We are now having 69999 rows and 161 columns

# In[17]:


# Percentage of Missing values in all Columns
pd.options.display.max_rows = None 
round(100 * data_v1.isnull().sum()/len(data_v1.index),2).sort_values(ascending=False)


# In[18]:


#Segregating categorical , continuous and date columns
catg = []
cont = []
for i in data_v1.columns:
    if data_v1[i].dtype == 'object':
        catg.append(i)
    else:
        cont.append(i)

dtcols = catg

catg = ['night_pck_user_6' , 'night_pck_user_8' , 'night_pck_user_7' , 'monthly_2g_6' , 'monthly_2g_7'
       ,'monthly_2g_8','sachet_2g_6','sachet_2g_7','sachet_2g_8','monthly_3g_6','monthly_3g_7',
       'monthly_3g_8','sachet_3g_6','sachet_3g_7','sachet_3g_8','fb_user_6','fb_user_7', 'fb_user_8']

cont = [column for column in cont if column not in catg]

display("Numerical Columns" ,len(cont),"Categorical Columns",len(catg),"Date Columns" ,len(dtcols))


# ## Insights of Missing Data Treatment ##

# > ### Replacing null value with -1 for below variables assuming no recharge has been made
# - - 'night_pck_user_6', 'night_pck_user_8', 'night_pck_user_7','fb_user_6', 'fb_user_7', 'fb_user_8'
# 
# > ### Replacing null value with 0 for all recharge amount columns assuming no recharge has been made. As its amount column, better to keep value 0 for further analysis
# 
# - - 'total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8','max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8','av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8'
# 
# > ### Imputing Count recharge columns with 0 considering no recharge has been made by the customer
# 
# - - 'count_rech_2g_6', 'count_rech_2g_7', 'count_rech_2g_8','count_rech_3g_6', 'count_rech_3g_7', 'count_rech_3g_8'
# 
# 
# 
# > ### As % of missing values are more than 70% and also not adding values to our analysis, dropping below columns
# 
# - - 'arpu_3g_6', 'arpu_3g_7', 'arpu_3g_8', 'arpu_2g_6', 'arpu_2g_7','arpu_2g_8'
# 
# > ### Dropping below date columns as it has same value and will not be used much. For other date columns will calculate age
# - - 'last_date_of_month_6','last_date_of_month_7', 'last_date_of_month_8'
# > ### Replacing Missing age data with -1 considering no recharge has been made by the user
# 
# - - 'date_of_last_rech_6age', 'date_of_last_rech_7age', 'date_of_last_rech_8age','date_of_last_rech_data_6age', 'date_of_last_rech_data_7age','date_of_last_rech_data_8age'
# 
# 
# > ### For other Continuous columns missing values has been replaced with median since we have outliers

# In[19]:


# Replacing null value with -1 for below variables assuming no recharge has been made

data_v1['night_pck_user_6'] = data_v1['night_pck_user_6'].fillna(-1)
data_v1['night_pck_user_7'] = data_v1['night_pck_user_7'].fillna(-1)
data_v1['night_pck_user_8'] = data_v1['night_pck_user_8'].fillna(-1)
data_v1['fb_user_6'] = data_v1['fb_user_6'].fillna(-1)
data_v1['fb_user_7'] = data_v1['fb_user_7'].fillna(-1)
data_v1['fb_user_8'] = data_v1['fb_user_8'].fillna(-1)

test_v1['night_pck_user_6'] = data_v1['night_pck_user_6'].fillna(-1)
test_v1['night_pck_user_7'] = data_v1['night_pck_user_7'].fillna(-1)
test_v1['night_pck_user_8'] = data_v1['night_pck_user_8'].fillna(-1)
test_v1['fb_user_6'] = data_v1['fb_user_6'].fillna(-1)
test_v1['fb_user_7'] = data_v1['fb_user_7'].fillna(-1)
test_v1['fb_user_8'] = data_v1['fb_user_8'].fillna(-1)


# In[20]:


#Replacing null value with 0 for all recharge amount columns assuming no recharge has been made. As its amount column, better to keep value 0 for further analysis
rchrg_amt_col=['total_rech_data_6', 'total_rech_data_7', 'total_rech_data_8','max_rech_data_6', 'max_rech_data_7', 'max_rech_data_8','av_rech_amt_data_6', 'av_rech_amt_data_7', 'av_rech_amt_data_8']

for column in rchrg_amt_col:
    data_v1[column] = data_v1[column].fillna(0)
    test_v1[column] = test_v1[column].fillna(0)


# In[21]:


# Check % of missing values after the replacing
round(100 * data_v1[rchrg_amt_col].isnull().sum()/len(data_v1.index),2)


# In[22]:


# Imputing Count recharge columns with 0 considering no recharge has been made by the customer

count_col = ['count_rech_2g_6', 'count_rech_2g_7', 'count_rech_2g_8','count_rech_3g_6', 'count_rech_3g_7', 'count_rech_3g_8']

for column in count_col:
    data_v1[column] = data_v1[column].fillna(0)
    test_v1[column] = test_v1[column].fillna(0)

# Re-check % of missing values after the imputation
round(100 * data_v1[count_col].isnull().sum()/len(data_v1.index),2)


# In[23]:


#As % of missing values are more than 70% and also not adding values to our analysis, dropping below columns
data_v1 = data_v1.drop(columns=['arpu_3g_6', 'arpu_3g_7', 'arpu_3g_8', 'arpu_2g_6', 'arpu_2g_7','arpu_2g_8'],axis=1)
test_v1 = test_v1.drop(columns=['arpu_3g_6', 'arpu_3g_7', 'arpu_3g_8', 'arpu_2g_6', 'arpu_2g_7','arpu_2g_8'],axis=1)


# In[24]:


#Dropping below date columns as it has unique data and will not be in use much 

data_v1 = data_v1.drop(columns=['last_date_of_month_6','last_date_of_month_7','last_date_of_month_8'],axis=1)
test_v1 = test_v1.drop(columns=['last_date_of_month_6','last_date_of_month_7','last_date_of_month_8'],axis=1)


# In[25]:


dtcols.remove('last_date_of_month_6')
dtcols.remove('last_date_of_month_7')
dtcols.remove('last_date_of_month_8')


# In[26]:


test_v1[dtcols].head()


# In[27]:


# compute age on above date columns.
# Replace missing age data with -1 considering no recharge has been made

for column in dtcols:
    colstr = column + 'age'
    display(colstr)
    data_v1[column] = pd.to_datetime(data_v1[column])
    data_v1[colstr] = data_v1[column].max() - data_v1[column]
    data_v1[colstr] = data_v1[colstr].dt.days
    data_v1 = data_v1.drop(columns=column,axis=1)
    data_v1[colstr] = data_v1[colstr].fillna(-1)
    

for column in dtcols:
    colstr = column + 'age'
    display(colstr)
    test_v1[column] = pd.to_datetime(test_v1[column])
    test_v1[colstr] = test_v1[column].max() - test_v1[column]
    test_v1[colstr] = test_v1[colstr].dt.days
    test_v1 = test_v1.drop(columns=column,axis=1)
    test_v1[colstr] = test_v1[colstr].fillna(-1)
    


# In[28]:


#Segregating cat , con and date columns
catg = []
cont = []
for i in data_v1.columns:
    if data_v1[i].dtype == 'object':
        catg.append(i)
    else:
        cont.append(i)

dtcols = catg

catg = ['night_pck_user_6' , 'night_pck_user_8' , 'night_pck_user_7' , 'monthly_2g_6' , 'monthly_2g_7'
       ,'monthly_2g_8','sachet_2g_6','sachet_2g_7','sachet_2g_8','monthly_3g_6','monthly_3g_7',
       'monthly_3g_8','sachet_3g_6','sachet_3g_7','sachet_3g_8','fb_user_6','fb_user_7', 'fb_user_8']

cont = [column for column in cont if column not in catg]

display("Numerical Columns" ,len(cont),"\nCategorical Columns",len(catg),"\nDate Columns" ,len(dtcols))


# In[29]:


data_v1[cont].describe()


# In[30]:


# Missing Value data treatment onnet_mou
onnet_mou = data_v1.columns[data_v1.columns.str.contains('onnet_mou')]

#Replacing the values with median since we have outliers here

for column in onnet_mou:
    data_v1[column] = data_v1[column].fillna(data_v1[column].median())
    test_v1[column] = test_v1[column].fillna(data_v1[column].median())


# In[31]:


# Missing Value data treatment offnet_mou
offnet_mou = data_v1.columns[data_v1.columns.str.contains('offnet_mou')]

#Replacing the values with median since we have outliers here

for column in offnet_mou:
    data_v1[column] = data_v1[column].fillna(data_v1[column].median())
    test_v1[column] = test_v1[column].fillna(data_v1[column].median())



# In[32]:


# Missing Value data treatment roaming
roam_col = data_v1.columns[data_v1.columns.str.contains('roam')]

#Replacing the values with median since we have outliers here

for column in roam_col:
    data_v1[column] = data_v1[column].fillna(data_v1[column].median())
    test_v1[column] = test_v1[column].fillna(data_v1[column].median())


# In[33]:


# Missing Value data treatment loc
local_calls = data_v1.columns[data_v1.columns.str.contains('loc')]

#Replacing the values with median since we have outliers here

for column in local_calls:
    data_v1[column] = data_v1[column].fillna(data_v1[column].median())
    test_v1[column] = test_v1[column].fillna(data_v1[column].median())


# In[34]:


# Missing Value data treatment std
std_calls = data_v1.columns[data_v1.columns.str.contains('std')]

#Replacing the values with median since we have outliers here

for column in std_calls:
    data_v1[column] = data_v1[column].fillna(data_v1[column].median())
    test_v1[column] = test_v1[column].fillna(data_v1[column].median())


# In[35]:


# Missing Value data treatment isd
isd_calls = data_v1.columns[data_v1.columns.str.contains('isd')]

# #Replacing the values with median since we have outliers here

for column in isd_calls:
    data_v1[column] = data_v1[column].fillna(data_v1[column].median())
    test_v1[column] = test_v1[column].fillna(data_v1[column].median())


# In[36]:


# Missing Value data treatment spl
spl_list = data_v1.columns[data_v1.columns.str.contains('spl')]

# #Replacing the values with median since we have outliers here

for column in spl_list:
    data_v1[column] = data_v1[column].fillna(data_v1[column].median())
    test_v1[column] = test_v1[column].fillna(data_v1[column].median())


# In[37]:


# Missing Value data treatment other
other_list = data_v1.columns[data_v1.columns.str.contains('other')]

# #Replacing the values with median since we have outliers here

for column in other_list:
    data_v1[column] = data_v1[column].fillna(data_v1[column].median())
    test_v1[column] = test_v1[column].fillna(data_v1[column].median())


# In[38]:


#Re-check null values
display(data_v1.isnull().sum())


# In[39]:


display ("Rows     : " ,data_v1.shape[0])
display ("Columns  : " ,data_v1.shape[1])
display ("Features : " ,data_v1.columns.tolist())
display ("Missing values :  ", data_v1.isnull().sum().values.sum())
display ("Unique values :  ",data_v1.nunique())


# In[40]:


data_v1[cont].describe()


# In[41]:


display ("Rows     : " ,test_v1.shape[0])
display ("Columns  : " ,test_v1.shape[1])
display ("Features : " ,test_v1.columns.tolist())
display ("Missing values :  ", test_v1.isnull().sum().values.sum())
display ("Unique values :  ",test_v1.nunique())


# <hr>
# <h1><a id='Outlier_Analysis_and_Treatment_Assumption_values'>Outlier Analysis and Treatment</a><br></h1>

# In[42]:


#Removing age column from outliers treatment
col = ['churn_probability','date_of_last_rech_data_6age', 'date_of_last_rech_data_7age', 'date_of_last_rech_8age', 'date_of_last_rech_6age', 'date_of_last_rech_data_8age', 'date_of_last_rech_7age']

for i in col:
    cont.remove(i)


# In[43]:


print(cont)


# In[44]:


# Continuous variable - Outlier Analysis

outliers = []
out_cols = cont
out_summary = []

for i in out_cols:
    Q3 = data_v1[i].quantile(.95)
    Q1 = data_v1[i].quantile(.05)
    IQR = Q3-Q1
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR
    
    if ((data_v1[i].min() < lower_bound) or (data_v1[i].max() > upper_bound)):
        out_summary.append("attribute \"{}\" with min value : {} -> max value : {} -> IQR {} -> lower_bound : {} match is {} -> upper_bound : {} match is {}".format(i,data_v1[i].min(),data_v1[i].max(),IQR,Q1-1.5*IQR,(data_v1[i].min() < lower_bound),Q3+1.5*IQR,data_v1[i].max() > upper_bound))
        outliers.append(i)


# List of outliers satisfying lower or upper bound        
for i in range(0,len(out_summary)):
    print("\nOutlier column with stats : \n\n{}\n".format(out_summary[i]))
    

#Outliers Treatment

for i in outliers:
    Q3 = data_v1[i].quantile(.95)
    Q1 = data_v1[i].quantile(.05)
    IQR = Q3-Q1
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR
    data_v1[i][data_v1[i]<=lower_bound] = lower_bound
    data_v1[i][data_v1[i]>=upper_bound] = upper_bound
    test_v1[i][test_v1[i]<=lower_bound] = lower_bound
    test_v1[i][test_v1[i]>=upper_bound] = upper_bound

print(outliers)

print("After outliers treatment\n\n",data_v1[out_cols].describe())


# <h1><a id='Transforming_Categorical_Columns'>Deriving Columns</a><br></h1>
# <ul>   
#     <li>calculate total data recharge amount</li>
#     <li>Filter High-Value Customers</li>
#     
# </ul>

# In[45]:


display(data_v1.head())


# <h2><a id='Total_data_recharge'>Calculate total data recharge amount</a><br></h2>

# In[46]:


# calculate the total data recharge amount for June and July 

data_v1['total_data_rech_6'] = data_v1.total_rech_data_6 * data_v1.av_rech_amt_data_6
data_v1['total_data_rech_7'] = data_v1.total_rech_data_7 * data_v1.av_rech_amt_data_7



# In[47]:


# calculate total recharge amount for June and July 
data_v1['amt_data_6'] = data_v1.total_rech_amt_6 + data_v1.total_data_rech_6
data_v1['amt_data_7'] = data_v1.total_rech_amt_7 + data_v1.total_data_rech_7



# In[48]:


# calculate average recharge done by customer in June and July
data_v1['av_amt_data_6_7'] = (data_v1.amt_data_6 + data_v1.amt_data_7)/2
test_v1['av_amt_data_6_7'] = (data_v1.amt_data_6 + data_v1.amt_data_7)/2


# In[49]:


# look at the 70th percentile recharge amount
display("Recharge amount at 70th percentile: {0}".format(data_v1.av_amt_data_6_7.quantile(0.7)))


# ## Filter High-Value Customers

# In[50]:


# Retain Customers who have made the recharge equivalent to 70th Percentile


data_v1_hvc = data_v1.loc[data_v1['av_amt_data_6_7'] >= data_v1['av_amt_data_6_7'].quantile(0.7),:]
data_v1_hvc = data_v1_hvc.reset_index(drop=True)
data_v1_hvc.shape


# In[51]:


#Dropping variables that are used to filter high-value customers as these will have same value

data_v1_hvc = data_v1_hvc.drop(['total_data_rech_6','total_data_rech_7','amt_data_6','amt_data_7'],axis=1)


# In[52]:


data_v1_hvc.shape


# In[53]:


#Derive difference in 8th month recharge in compare to avg of 6th and 7th month fo High value customer DF
data_v1_hvc['onnet_mou_diff'] = data_v1_hvc.onnet_mou_8 - ((data_v1_hvc.onnet_mou_6 + data_v1_hvc.onnet_mou_7)/2)

data_v1_hvc['offnet_mou_diff'] = data_v1_hvc.offnet_mou_8 - ((data_v1_hvc.offnet_mou_6 + data_v1_hvc.offnet_mou_7)/2)

data_v1_hvc['roam_ic_mou_diff'] = data_v1_hvc.roam_ic_mou_8 - ((data_v1_hvc.roam_ic_mou_6 + data_v1_hvc.roam_ic_mou_7)/2)

data_v1_hvc['roam_og_mou_diff'] = data_v1_hvc.roam_og_mou_8 - ((data_v1_hvc.roam_og_mou_6 + data_v1_hvc.roam_og_mou_7)/2)

data_v1_hvc['loc_og_mou_diff'] = data_v1_hvc.loc_og_mou_8 - ((data_v1_hvc.loc_og_mou_6 + data_v1_hvc.loc_og_mou_7)/2)

data_v1_hvc['std_og_mou_diff'] = data_v1_hvc.std_og_mou_8 - ((data_v1_hvc.std_og_mou_6 + data_v1_hvc.std_og_mou_7)/2)

data_v1_hvc['isd_og_mou_diff'] = data_v1_hvc.isd_og_mou_8 - ((data_v1_hvc.isd_og_mou_6 + data_v1_hvc.isd_og_mou_7)/2)

data_v1_hvc['spl_og_mou_diff'] = data_v1_hvc.spl_og_mou_8 - ((data_v1_hvc.spl_og_mou_6 + data_v1_hvc.spl_og_mou_7)/2)

data_v1_hvc['total_og_mou_diff'] = data_v1_hvc.total_og_mou_8 - ((data_v1_hvc.total_og_mou_6 + data_v1_hvc.total_og_mou_7)/2)

data_v1_hvc['loc_ic_mou_diff'] = data_v1_hvc.loc_ic_mou_8 - ((data_v1_hvc.loc_ic_mou_6 + data_v1_hvc.loc_ic_mou_7)/2)

data_v1_hvc['std_ic_mou_diff'] = data_v1_hvc.std_ic_mou_8 - ((data_v1_hvc.std_ic_mou_6 + data_v1_hvc.std_ic_mou_7)/2)

data_v1_hvc['isd_ic_mou_diff'] = data_v1_hvc.isd_ic_mou_8 - ((data_v1_hvc.isd_ic_mou_6 + data_v1_hvc.isd_ic_mou_7)/2)

data_v1_hvc['spl_ic_mou_diff'] = data_v1_hvc.spl_ic_mou_8 - ((data_v1_hvc.spl_ic_mou_6 + data_v1_hvc.spl_ic_mou_7)/2)

data_v1_hvc['total_ic_mou_diff'] = data_v1_hvc.total_ic_mou_8 - ((data_v1_hvc.total_ic_mou_6 + data_v1_hvc.total_ic_mou_7)/2)

data_v1_hvc['total_rech_num_diff'] = data_v1_hvc.total_rech_num_8 - ((data_v1_hvc.total_rech_num_6 + data_v1_hvc.total_rech_num_7)/2)

data_v1_hvc['total_rech_amt_diff'] = data_v1_hvc.total_rech_amt_8 - ((data_v1_hvc.total_rech_amt_6 + data_v1_hvc.total_rech_amt_7)/2)

data_v1_hvc['max_rech_amt_diff'] = data_v1_hvc.max_rech_amt_8 - ((data_v1_hvc.max_rech_amt_6 + data_v1_hvc.max_rech_amt_7)/2)

data_v1_hvc['total_rech_data_diff'] = data_v1_hvc.total_rech_data_8 - ((data_v1_hvc.total_rech_data_6 + data_v1_hvc.total_rech_data_7)/2)

data_v1_hvc['max_rech_data_diff'] = data_v1_hvc.max_rech_data_8 - ((data_v1_hvc.max_rech_data_6 + data_v1_hvc.max_rech_data_7)/2)

data_v1_hvc['av_rech_amt_data_diff'] = data_v1_hvc.av_rech_amt_data_8 - ((data_v1_hvc.av_rech_amt_data_6 + data_v1_hvc.av_rech_amt_data_7)/2)

data_v1_hvc['vol_2g_mb_diff'] = data_v1_hvc.vol_2g_mb_8 - ((data_v1_hvc.vol_2g_mb_6 + data_v1_hvc.vol_2g_mb_7)/2)

data_v1_hvc['vol_3g_mb_diff'] = data_v1_hvc.vol_3g_mb_8 - ((data_v1_hvc.vol_3g_mb_6 + data_v1_hvc.vol_3g_mb_7)/2)


# In[54]:


#For test data set: Derive difference in 8th month recharge in compare to avg of 6th and 7th month fo High value customer DFtest_v1['onnet_mou_diff'] = test_v1.onnet_mou_8 - ((test_v1.onnet_mou_6 + test_v1.onnet_mou_7)/2)
test_v1['onnet_mou_diff'] = test_v1.onnet_mou_8 - ((test_v1.onnet_mou_6 + test_v1.onnet_mou_7)/2)

test_v1['offnet_mou_diff'] = test_v1.offnet_mou_8 - ((test_v1.offnet_mou_6 + test_v1.offnet_mou_7)/2)

test_v1['roam_ic_mou_diff'] = test_v1.roam_ic_mou_8 - ((test_v1.roam_ic_mou_6 + test_v1.roam_ic_mou_7)/2)

test_v1['roam_og_mou_diff'] = test_v1.roam_og_mou_8 - ((test_v1.roam_og_mou_6 + test_v1.roam_og_mou_7)/2)

test_v1['loc_og_mou_diff'] = test_v1.loc_og_mou_8 - ((test_v1.loc_og_mou_6 + test_v1.loc_og_mou_7)/2)

test_v1['std_og_mou_diff'] = test_v1.std_og_mou_8 - ((test_v1.std_og_mou_6 + test_v1.std_og_mou_7)/2)

test_v1['isd_og_mou_diff'] = test_v1.isd_og_mou_8 - ((test_v1.isd_og_mou_6 + test_v1.isd_og_mou_7)/2)

test_v1['spl_og_mou_diff'] = test_v1.spl_og_mou_8 - ((test_v1.spl_og_mou_6 + test_v1.spl_og_mou_7)/2)

test_v1['total_og_mou_diff'] = test_v1.total_og_mou_8 - ((test_v1.total_og_mou_6 + test_v1.total_og_mou_7)/2)

test_v1['loc_ic_mou_diff'] = test_v1.loc_ic_mou_8 - ((test_v1.loc_ic_mou_6 + test_v1.loc_ic_mou_7)/2)

test_v1['std_ic_mou_diff'] = test_v1.std_ic_mou_8 - ((test_v1.std_ic_mou_6 + test_v1.std_ic_mou_7)/2)

test_v1['isd_ic_mou_diff'] = test_v1.isd_ic_mou_8 - ((test_v1.isd_ic_mou_6 + test_v1.isd_ic_mou_7)/2)

test_v1['spl_ic_mou_diff'] = test_v1.spl_ic_mou_8 - ((test_v1.spl_ic_mou_6 + test_v1.spl_ic_mou_7)/2)

test_v1['total_ic_mou_diff'] = test_v1.total_ic_mou_8 - ((test_v1.total_ic_mou_6 + test_v1.total_ic_mou_7)/2)

test_v1['total_rech_num_diff'] = test_v1.total_rech_num_8 - ((test_v1.total_rech_num_6 + test_v1.total_rech_num_7)/2)

test_v1['total_rech_amt_diff'] = test_v1.total_rech_amt_8 - ((test_v1.total_rech_amt_6 + test_v1.total_rech_amt_7)/2)

test_v1['max_rech_amt_diff'] = test_v1.max_rech_amt_8 - ((test_v1.max_rech_amt_6 + test_v1.max_rech_amt_7)/2)

test_v1['total_rech_data_diff'] = test_v1.total_rech_data_8 - ((test_v1.total_rech_data_6 + test_v1.total_rech_data_7)/2)

test_v1['max_rech_data_diff'] = test_v1.max_rech_data_8 - ((test_v1.max_rech_data_6 + test_v1.max_rech_data_7)/2)

test_v1['av_rech_amt_data_diff'] = test_v1.av_rech_amt_data_8 - ((test_v1.av_rech_amt_data_6 + test_v1.av_rech_amt_data_7)/2)

test_v1['vol_2g_mb_diff'] = test_v1.vol_2g_mb_8 - ((test_v1.vol_2g_mb_6 + test_v1.vol_2g_mb_7)/2)

test_v1['vol_3g_mb_diff'] = test_v1.vol_3g_mb_8 - ((test_v1.vol_3g_mb_6 + test_v1.vol_3g_mb_7)/2)


# In[55]:


display(    test_v1.shape)


# In[56]:


display(    data_v1_hvc.shape)


# In[57]:


display(
    test_v1.isnull().sum()
)


# In[58]:


display(
    data_v1_hvc.shape
)


# In[59]:


display(    data_v1_hvc['churn_probability'].value_counts())


# <h1><a id='Univariate_Analysis'>Univariate Analysis</a><br></h1>

# In[60]:


display(
    data_v1_hvc.describe(percentiles=[.10,.25,.50,.75,.90,.95,.99])
)


# In[61]:


#Draw histogram to see the distribution of below variables
cont_ctg_var=['night_pck_user_6', 'night_pck_user_8', 'night_pck_user_7', 'monthly_2g_6', 'monthly_2g_7', 'monthly_2g_8', 'sachet_2g_6', 'sachet_2g_7', 'sachet_2g_8', 'monthly_3g_6',
 'monthly_3g_7', 'monthly_3g_8', 'sachet_3g_6', 'sachet_3g_7', 'sachet_3g_8', 'fb_user_6', 'fb_user_7', 'fb_user_8']
plt.figure(figsize=(5,5))
for i in cont_ctg_var:
    print("Variable Name: ",i)
    plt.hist(data_v1_hvc[i])
    plt.show()


# In[62]:


display(sb.countplot(x="churn_probability",data = data_v1_hvc))


# ##### Insight: Percent of churn is very low in compared to non churn

# <h1><a id='Bivariate_Analysis'>Bivariate Analysis</a><br></h1>

# In[63]:


# Set the figure size
sb.set(rc={'figure.figsize':(11.7, 8.27)})

# Define the bins and labels
bins = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
labels = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Create a new column for 'age_on_network_binned'
data_v1_hvc['age_on_network_binned'] = pd.cut(round(((data_v1_hvc['aon'] / 30) / 12), 1), bins=bins, labels=labels)

# Plot the count plot
sb.countplot(x='age_on_network_binned', hue='churn_probability', data=data_v1_hvc)
plt.show()


# #### Insight:
# ##### With the increase in age on network, churn count is decreasing and hence probability of getting churn will decrease if customer stays long with the netwrok

# In[64]:


#Writing generic function for displaying box plot based on attributes given as input vs churn probability
def colboxplt(cols):
    plt.figure(figsize=(40, 25))
    for i in range(len(cols)):
        plt.subplot(2,len(cols),i+1)
        K = pd.concat([data_v1_hvc[cols[i]],data_v1_hvc['churn_probability']], axis=1)
        K = pd.melt(K,id_vars="churn_probability",var_name="features",value_name='value')
        sb.boxplot(x="features", y="value", hue="churn_probability",data = K)
        plt.xticks()    
        plt.suptitle('Incoming Calls Usage')
        plt.subplot(2,3,3+i+1)
        sb.distplot(data[cols[i]])


# In[65]:


# Analysis Outgoing Minutes of Usage 
cols = [['total_og_mou_6'],
        ['total_og_mou_7'],
        ['total_og_mou_8']]
colboxplt(cols)


# #### Insights:
# #####  If the amount of outgoing is increasing can see for june and july month the amount of churn is relatively more

# In[66]:


# Analysis Incoming Minutes of Usage 
cols =["total_ic_mou_6","total_ic_mou_7","total_ic_mou_8"]
colboxplt(cols)


# #### Insights:
# ###### Total Minutes of usage for Incoming calls are skewed to left side 
# ###### If the total MOU is more, the probability of getting churned is less

# In[67]:


#Analysis total recharge
cols = ['total_rech_num_6','total_rech_num_7','total_rech_num_8']
colboxplt(cols)


# #### Insights:
# 
# #####  For June month, increase in the total rechage, can observe for more churn 

# In[68]:


#define custom function to draw boxplot for bivariate analysis
def Bivariate_boxplt(cols):
    plt.figure(figsize=(60, 45))
    for i in range(0,11):
        plt.subplot(4,3,i+1)
        BV = pd.concat([data_v1_hvc[cols[i]],data_v1_hvc['churn_probability']], axis=1)
        BV = pd.melt(BV,id_vars="churn_probability",var_name="features",value_name='value')
        sb.boxplot(x="features", y="value", hue="churn_probability",data = BV)
        plt.xticks()    
        plt.suptitle('2G-3G Volume')


# In[69]:


# Drawing box plot for below continuous variables and compare with chrun across different months
cols = [
        ['vol_2g_mb_6','vol_2g_mb_7','vol_2g_mb_8'],
        ['vol_3g_mb_6','vol_3g_mb_7','vol_3g_mb_8'],
        ['roam_ic_mou_6','roam_ic_mou_7','roam_ic_mou_8'],
        ['roam_og_mou_6','roam_og_mou_7','roam_og_mou_8'],
        ['monthly_2g_6','monthly_2g_7','monthly_2g_8'],
        ['monthly_3g_6','monthly_3g_7','monthly_3g_8'],
        ['sachet_2g_6','sachet_2g_7','sachet_2g_8'],
        ['sachet_3g_6','sachet_3g_7','sachet_3g_8'],
        ['jun_vbc_3g','jul_vbc_3g','aug_vbc_3g'],
        ['std_og_mou_6','std_og_mou_7','std_og_mou_8'],
        ['isd_og_mou_6','isd_og_mou_7','isd_og_mou_8']
       ]
Bivariate_boxplt(cols)


# #### Insights:
# ##### Increase in roaming, churn is increasing
# ##### Increase in outgoing std, churn is high

# In[70]:


#Ploting Heat map for Correlation analyis

plt.figure(figsize=(25,25))
sb.heatmap(data_v1.corr(),cmap="Reds")


# #### Insights:
# ###### Most of the features seems highly correlated. So, we need to use PCA to handle multicollinearity and dimensionality reductions

# In[71]:


#Spliting testing and training data 

X = data_v1_hvc.drop(["churn_probability"],axis=1)
Y = data_v1_hvc.churn_probability


# In[72]:


xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.3, random_state = 42, stratify = Y)


# In[73]:


# Aggregating the Categorical Columns

train = pd.concat([xtrain, ytrain], axis=1)

# aggregate the categorical variables
display(train.groupby('night_pck_user_6').churn_probability.mean())
display(train.groupby('night_pck_user_7').churn_probability.mean())
display(train.groupby('night_pck_user_8').churn_probability.mean())
display(train.groupby('fb_user_6').churn_probability.mean())
display(train.groupby('fb_user_7').churn_probability.mean())
display(train.groupby('fb_user_8').churn_probability.mean())

# replace categories with aggregated values in each categorical column
mapping = {'night_pck_user_6' : {-1: 0.107529, 0: 0.084044, 1: 0.127596},
           'night_pck_user_7' : {-1: 0.114231, 0: 0.065649, 1: 0.068750},
           'night_pck_user_8' : {-1: 0.126636, 0: 0.032644, 1: 0.034602},
           'fb_user_6'        : {-1: 0.107529, 0: 0.105496, 1: 0.083258},
           'fb_user_7'        : {-1: 0.114231, 0: 0.087029, 1: 0.063630},
           'fb_user_8'        : {-1: 0.126636, 0: 0.062458, 1: 0.029049}
          }
xtrain.replace(mapping, inplace = True)
xtest.replace(mapping, inplace = True)


# In[74]:


data_v1_hvc.shape


# <h1><a id='Principal_Component_Analysis'>Principal Component Analysis</a><br></h1>

# ### Case1 : By Splitting train data into train and test

# In[75]:


#find value count % of churn probability
round(100*data_v1_hvc['churn_probability'].value_counts()/len(data_v1_hvc.index),2)


# In[76]:


#Churn Distribution
pie_chart = data_v1_hvc['churn_probability'].value_counts()*100.0 /len(data_v1_hvc)
ax = pie_chart.plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'],figsize =(8,6), fontsize = 14 )                                                                           
ax.set_ylabel('Churn',fontsize = 12)
ax.set_title('Churn Distribution', fontsize = 12)
plt.show()


# ### Data Scaling

# In[77]:


# Scaling the data using Standard Scaler
col = list(xtrain.columns)
# Data Scaling
scaler = StandardScaler()
xtrain_scaled = scaler.fit_transform(xtrain)
xtest_scaled = scaler.transform(xtest)
test_v1_scaled = scaler.fit_transform(test_v1)

# Applying Principal Component Analysis
pca = PCA()
pca.fit(xtrain)
xtrain_pca = pca.fit_transform(xtrain_scaled)


# ### PCA

# In[78]:


#  feature variance Graph
var_cumu = np.cumsum(pca.explained_variance_ratio_)
fig = plt.figure(figsize=[12,8])
plt.vlines(x=15, ymax=1, ymin=0, colors="r", linestyles="--")
plt.hlines(y=0.95, xmax=30, xmin=0, colors="g", linestyles="--")
plt.plot(var_cumu)
plt.xlabel("PCA Components")
plt.ylabel("Cumulative variance explained")
plt.show()


# In[79]:


df_pca_c1 = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'PC3':pca.components_[2],'Feature':col})
df_pca_c1.head(10)


# In[80]:


np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


# #### 75 variables will be enough to explain 95% variance in the dataset and hence selecting 75 variables for our modelling

# ### Handling Class Impbalance using SMOTE

# In[81]:


from imblearn.over_sampling import SMOTE
from sklearn.decomposition import IncrementalPCA
from IPython.display import display

# Display message
display("Applying SMOTE to normalize imbalance")

# Initialize SMOTE with a sampling strategy
smote = SMOTE(sampling_strategy=0.8)

# Fit and resample the data
x_smote, y_smote = smote.fit_resample(xtrain_scaled, ytrain)

# Display the shape of the resampled training dataset
display("Shape of train dataset after SMOTE: " + str(x_smote.shape))

# Applying Incremental PCA
pca = IncrementalPCA(n_components=75)

# Fit and transform the training data
x_train_smote_pca = pca.fit_transform(x_smote)

# Transform the test data
x_test_smote_pca = pca.transform(xtest_scaled)

# Transform the validation data
test_v1_scaled_pca = pca.fit_transform(test_v1_scaled)


# In[82]:


display("Shape of train datatset after SMOTE and PCA : "+str(x_train_smote_pca.shape))


# In[83]:


x_test_smote_pca.shape


# In[84]:


from collections import Counter

display(Counter(ytrain))
display(Counter(y_smote))


# ### Function to Get Metrics for Model Evaluation

# In[85]:


def evaluate_model_metric(dt_classifier,ytrain,ytest,xtrain,xtest):
    print("Train Accuracy :", accuracy_score(ytrain, dt_classifier.predict(xtrain)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(ytrain, dt_classifier.predict(xtrain)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(ytest, dt_classifier.predict(xtest)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(ytest, dt_classifier.predict(xtest)))
    sensitivity, specificity, _ = sensitivity_specificity_support(ytest, dt_classifier.predict(xtest), average='binary')
    print("Sensitivity:", round(sensitivity, 2))
    print("Specificity:", round(specificity, 2))
    print("Recall Score:",round(metrics.recall_score(ytest,dt_classifier.predict(xtest)),2))
    print("Precision Score:",round(metrics.precision_score(ytest,dt_classifier.predict(xtest)),2))
    print("ROC AUC:",round(metrics.roc_auc_score(ytest,dt_classifier.predict(xtest)),2))
    print("F1 Score:",round(metrics.f1_score(ytest,dt_classifier.predict(xtest)),2))
    


# ## Model Creation 

# ### Model 1 : Logistic Regression Without Hyperparameter Tuning

# In[86]:


# Logistic Regression without Hyper Parameter Turning

#Training the model on the train data

lr = LogisticRegression()
model = lr.fit(x_train_smote_pca,y_smote)
#Making prediction on the test data
pred_probs_test = model.predict_proba(x_test_smote_pca)[:,1]
display("Logistic Regression Accurancy : "+"{:2.2}".format(metrics.roc_auc_score(ytest, pred_probs_test)))
evaluate_model_metric(lr,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)


# ### Model 2 : Logistic Regression With Hyperparameter Tuning

# In[87]:


# Logistic Regression with Hyper Parameter Turning

logistic = LogisticRegression()

# hyperparameter space
params = {'C': [0.1, 0.5, 1, 2, 3, 4, 5, 10], 'penalty': ['l1', 'l2']}

# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)


# create gridsearch object
lr_hp = GridSearchCV(estimator=logistic, cv=folds, param_grid=params, scoring='roc_auc', n_jobs=-1, verbose=1)

# fit model
lr_hp.fit(x_train_smote_pca, y_smote)

# display best hyperparameters
display("Best AUC: ", lr_hp.best_score_)
display("Best hyperparameters: ", lr_hp.best_params_)

evaluate_model_metric(lr_hp,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)


# ####  Insights:
# ##### Logistic Regression before tuning its giving 88% accurancy and after tuning it is giving 92.5% 
# 

# ### Model 3 : Random Forest without Hyperparameter Tuning

# In[88]:


# Random Forest

rf = RandomForestClassifier()
rf.fit(x_train_smote_pca,y_smote)
evaluate_model_metric(rf,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)


# In[89]:


# Random Forest Classifier Tuning 1

# number of variables to consider to split each node
max_features = int(round(np.sqrt(xtrain.shape[1])))  
display(max_features)

rf_model = RandomForestClassifier(n_estimators=100, max_features=max_features, class_weight={0:0.1, 1: 0.9}, oob_score=True, random_state=4, verbose=1)

rf_model.fit(x_train_smote_pca, y_smote)

display("OOB Score",rf_model.oob_score_)

evaluate_model_metric(rf_model,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)


# ###  Model 4: Random Forest with Hyperparameter Tuning

# In[90]:


# Random Forest Classifier Tuning 2: Max Depth

rf = RandomForestClassifier(random_state=42, n_jobs=-1,oob_score=True)
params = {
    'max_depth': range(2, 40, 5)
}

grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy",return_train_score=True)
grid_search.fit(x_train_smote_pca, y_smote)


# In[91]:


evaluate_model_metric(grid_search,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)
scores = pd.DataFrame(grid_search.cv_results_)


# In[92]:


#Plot the graph for train and test accuracy
for key in params.keys():
    hyperparameters = key
    break
plt.figure(figsize=(16,5))
plt.plot(scores["param_"+hyperparameters], scores["mean_train_score"], label="training accuracy")
plt.plot(scores["param_"+hyperparameters], scores["mean_test_score"], label="test accuracy")
plt.xlabel(hyperparameters)
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# #### Insight:
# ##### From the above plot max depth can be either 12 or 17. 
# ##### After 17 graph become constant

# In[93]:


# Random Forest Classifier Tuning 3: parameters = {'n_estimators': range(100, 800, 200)}

rf = RandomForestClassifier(random_state=42, n_jobs=-1,oob_score=True)
params = {
    'n_estimators': range(100, 800, 200)
}

grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy",return_train_score=True)
grid_search.fit(x_train_smote_pca, y_smote)

#Evaluate Metrics
evaluate_model_metric(grid_search,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)
scores = pd.DataFrame(grid_search.cv_results_)

#Plot the graph for train and test accuracy
for key in params.keys():
    hyperparameters = key
    break
plt.figure(figsize=(16,5))
plt.plot(scores["param_"+hyperparameters], scores["mean_train_score"], label="training accuracy")
plt.plot(scores["param_"+hyperparameters], scores["mean_test_score"], label="test accuracy")
plt.xlabel(hyperparameters)
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# #### Insights:
# ##### n_estimators seems to be almost constant lets take 200

# In[ ]:


#Random Forest Classifier Tuning 4:- parameters = {'max_features': [20,30,40,50,60]}

rf = RandomForestClassifier(random_state=42, n_jobs=-1,oob_score=True)
params = {
    'max_features': [20,30,40,50,60]
}

grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy",return_train_score=True)
grid_search.fit(x_train_smote_pca, y_smote)
#Evaluate Metrics
evaluate_model_metric(grid_search,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)
scores = pd.DataFrame(grid_search.cv_results_)
#Plot the graph for train and test accuracy
for key in params.keys():
    hyperparameters = key
    break
plt.figure(figsize=(16,5))
plt.plot(scores["param_"+hyperparameters], scores["mean_train_score"], label="training accuracy")
plt.plot(scores["param_"+hyperparameters], scores["mean_test_score"], label="test accuracy")
plt.xlabel(hyperparameters)
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# #### Insights:
# ##### As after 40, the graph started declining more, we will take max features as 40 

# In[ ]:


# Random Forest Classifier Tuning 5: parameters = {'min_samples_leaf': range(1, 100, 10)}

rf = RandomForestClassifier(random_state=42, n_jobs=-1,oob_score=True)
params = {
    'min_samples_leaf': range(1, 100, 10)
}

grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy",return_train_score=True)
grid_search.fit(x_train_smote_pca, y_smote)

#Evaluate Metrics
evaluate_model_metric(grid_search,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)
scores = pd.DataFrame(grid_search.cv_results_)

#Plot the graph for train and test accuracy
for key in params.keys():
    hyperparameters = key
    break
plt.figure(figsize=(16,5))
plt.plot(scores["param_"+hyperparameters], scores["mean_train_score"], label="training accuracy")
plt.plot(scores["param_"+hyperparameters], scores["mean_test_score"], label="test accuracy")
plt.xlabel(hyperparameters)
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# #### Insights:
# ##### Model may start to overfit as accuracy is decreasing with min sample leaf. Lets consider min sample leaf as 10 or 20

# In[ ]:


# Random Forest Classifier Tuning 6: - parameters = {'min_samples_split': range(10, 100, 10)}

rf = RandomForestClassifier(random_state=42, n_jobs=-1,oob_score=True)
params = {
    'min_samples_split': range(10, 100, 10)
}

grid_search = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy",return_train_score=True)
grid_search.fit(x_train_smote_pca, y_smote)

evaluate_model_metric(grid_search,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)
scores = pd.DataFrame(grid_search.cv_results_)

for key in params.keys():
    hyperparameters = key
    break
plt.figure(figsize=(16,5))
plt.plot(scores["param_"+hyperparameters], scores["mean_train_score"], label="training accuracy")
plt.plot(scores["param_"+hyperparameters], scores["mean_test_score"], label="test accuracy")
plt.xlabel(hyperparameters)
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# #### Insights:
# ##### As plot started decreasing after 40, let consider min_samples_split as 40 

# In[ ]:


# Random Forest Classifier Tuning 7: parameters = # Final Model after all the tuning

rf = RandomForestClassifier(random_state=42, n_jobs=-1,oob_score=True)
params = {
    'max_depth': [12],
    'n_estimators' : [200],
    'max_features' : [40],
    'min_samples_leaf' : [10,20],
    'min_samples_split' : [40]
}

rf_final_model = GridSearchCV(estimator=rf,
                           param_grid=params,
                           cv = 4,
                           n_jobs=-1, verbose=1, scoring="accuracy",return_train_score=True)
rf_final_model.fit(x_train_smote_pca, y_smote)

evaluate_model_metric(rf_final_model,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)
scores = pd.DataFrame(rf_final_model.cv_results_)


# #### Insights of Random Forest Models:
# ##### Consider below metrics, this model seems to be good with best hyperparameters
# ###### > Train Accuracy: 94.29;  Test Accuracy: 89.5;  Recall Score: 0.65; and Precision Score: 0.41

# ### MODEL 5: ADABOOST

# In[ ]:


#Using adaBoosting to predict 'Churn Prob' 
adaboost =  AdaBoostClassifier(n_estimators=200, random_state=1)
adaboost.fit(x_train_smote_pca, y_smote)


# In[ ]:


display('Accuracy of the Train model is:  ',accuracy_score(y_smote, adaboost.predict(x_train_smote_pca)))
display('Accuracy of the Test model is:  ',accuracy_score(ytest, adaboost.predict(x_test_smote_pca)))
evaluate_model_metric(adaboost,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)


# ### Model 6 : ADABOOST without using PCA

# In[ ]:


#Using xtrain and ytrain without PCA, build ADABOOST
adaboost =  AdaBoostClassifier(n_estimators=200, random_state=1)
adaboost.fit(xtrain, ytrain)


# In[ ]:


display('Accuracy of the Train model is:  ',accuracy_score(ytrain, adaboost.predict(xtrain)))
display('Accuracy of the Test model is:  ',accuracy_score(ytest, adaboost.predict(xtest)))
evaluate_model_metric(adaboost,ytrain,ytest,xtrain,xtest)


# ### Model 7 : ADABOOST without using PCA and with Hyperparameter Tuning

# In[ ]:


##Using xtrain and ytrain without PCA, build ADABOOST with hyperparameter tuning
params = {
        'n_estimators' : [50,100, 200], # no of trees   # eta
        'algorithm': ['SAMME', 'SAMME.R'],
        }

folds = 5

param_comb = 800

random_search_ada = RandomizedSearchCV(adaboost, param_distributions=params, n_iter=param_comb, scoring='accuracy', n_jobs=-1, cv=5, verbose=3, random_state=42)
random_search_ada.fit(xtrain, ytrain)


# In[ ]:


display('Accuracy of the Train model is:  ',accuracy_score(ytrain, random_search_ada.predict(xtrain)))
display('Accuracy of the Test model is:  ',accuracy_score(ytest, random_search_ada.predict(xtest)))
evaluate_model_metric(random_search_ada,ytrain,ytest,xtrain,xtest)


# In[ ]:


display('Best estimator:',random_search_ada.best_estimator_)
display('Best accuracy for %d-fold search with %d parameter combinations:' % (folds, param_comb),random_search_ada.best_score_)
display('Best hyperparameters:',random_search_ada.best_params_)


# #### Insights:
# ##### Best Accuracy that we got from adaboost (without PCA) is 94.28% on train and 93.8% percent on test data. Both are very closer
# ##### Recall score of 47% and precision score of 68%

# ### Model 8: XGBOOST

# In[ ]:


### XG Boost - Model 1 
# fit model on training data with default hyperparameters

XGB_model = XGBClassifier()
XGB_model.fit(x_train_smote_pca, y_smote)
evaluate_model_metric(XGB_model,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)


# ### Model 9: XGBOOST with Hyperparameter Tuning

# In[ ]:


# hyperparameter tuning with XGBoost

# creating a KFold object with 5 folds
folds = 5

# specify range of hyperparameters
param_grid = {'learning_rate': [0.1,0.2,0.3], 
             'subsample': [0.3,0.4,0.5]
             }          


# specify model
xgb_model_tune = XGBClassifier(max_depth=2, n_estimators=200,n_jobs=-1)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model_tune, 
                        param_grid = param_grid, 
                        scoring= 'accuracy', 
                        cv = folds, 
                        n_jobs = -1,
                        verbose = 1,
                        return_train_score=True)

model_cv.fit(x_train_smote_pca, y_smote)
evaluate_model_metric(model_cv,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)


# In[ ]:


# displaying the best accuracy score and hyperparameters
print('We  get best score of '+str(round(model_cv.best_score_,2)) +' using parameters '+str(model_cv.best_params_))


# In[ ]:


# tune model with best hyperparameter values
param = {'learning_rate': 0.3,
          'max_depth': 3, 
          'n_estimators':200,
          'subsample':0.4,
          'gamma': 1,
         'objective':'binary:logistic'}

# fit model on training data
xgb_model_tune_2 = XGBClassifier(params = param,max_depth=2, n_estimators=200,min_child_weight=1,scale_pos_weight = 1)
xgb_model_tune_2.fit(x_train_smote_pca, y_smote)
evaluate_model_metric(xgb_model_tune_2,y_smote,ytest,x_train_smote_pca,x_test_smote_pca)


# #### Insights:
# ##### XGBBOOST (with PCA and hyperparameter tuning) is giving good accuracy and recall score as compared to others but instead of using one model we will use combination of Multiple models

# ## Derive output of all models and predict test data based on the combination

# ### XGBOOST

# In[ ]:


test_v1.shape


# In[ ]:


# XGBOOOST
churn_probability_XGB = xgb_model_tune_2.predict(test_v1_scaled_pca)
csvdata = {'id':test['id'],'churn_probability_xgboost':churn_probability_XGB}
df = pd.DataFrame(csvdata)
display(df.shape)


# In[ ]:


#find the churn prob value count
df['churn_probability_xgboost'].value_counts()


# ### Logistic Regression

# In[ ]:


# Logistic Regression
churn_probability_logistic = lr_hp.predict(test_v1_scaled_pca)
df['churn_probability_logistic'] = pd.DataFrame(churn_probability_logistic,columns=['churn_probability_logistic'])
display(df.shape)


# In[ ]:


df['churn_probability_logistic'].value_counts()


# # Random Forest

# In[ ]:


# Random Forest Final model with hyperparater tuning
churn_probability_randomforest = rf_final_model.predict(test_v1_scaled_pca)
df['churn_probability_randomforest'] = pd.DataFrame(churn_probability_randomforest,columns=['churn_probability_randomforest'])
display(df.shape)


# In[ ]:


df['churn_probability_randomforest'].value_counts()


# In[ ]:


test_v1.shape


# In[ ]:


#Print churn probability of different model for respective id
df.head()


# # Adaboost

# In[ ]:


# Adaboost without PCA
churn_probability_adaboost = random_search_ada.predict(test_v1_scaled)
df['churn_probability_adaboost'] = pd.DataFrame(churn_probability_adaboost,columns=['churn_probability_adaboost'])
display(df.shape)


# In[ ]:


df.head()


# In[ ]:


df['churn_probability_adaboost'].value_counts()


# In[ ]:


#Sum up all churn prob
df['total_churn'] = df['churn_probability_xgboost'] + df['churn_probability_adaboost'] + df['churn_probability_randomforest']


# In[ ]:


#Derive final churn prob: if value is more than 1 then 1 else 0; Conisdering if two or more model results 1 then 1
df['Total_Case1'] = df['total_churn'].apply(lambda x : 1 if x>1 else 0)


# In[ ]:


df['Total_Case1'].value_counts()


# In[ ]:


df.head()


# In[ ]:


#Execute command to export the results of test data
df[['id','Total_Case1']].to_csv('final_model.csv',index=False)


# ##### In order to improve further accuracy lets build another set of models on train data directly instead of spliting 
# - we will again use these models in linear combination

# ### Apply Algorithms on X and Y train directly

# In[ ]:


trainX = X.copy()
trainY = Y.copy()

display(trainX.shape,trainY.shape)


# In[ ]:


# Aggregating the Categorical Columns

Dtrain = pd.concat([trainX, trainY], axis=1)

# aggregate the categorical variables
display(Dtrain.groupby('night_pck_user_6').churn_probability.mean())
display(Dtrain.groupby('night_pck_user_7').churn_probability.mean())
display(Dtrain.groupby('night_pck_user_8').churn_probability.mean())
display(Dtrain.groupby('fb_user_6').churn_probability.mean())
display(Dtrain.groupby('fb_user_7').churn_probability.mean())
display(Dtrain.groupby('fb_user_8').churn_probability.mean())



# In[ ]:


# replace categories with aggregated values in each categorical column
mapping = {'night_pck_user_6' : {-1: 0.099925, 0: 0.067941, 1: 0.109091},
           'night_pck_user_7' : {-1: 0.116903, 0: 0.056424, 1: 0.062500},
           'night_pck_user_8' : {-1: 0.142189, 0: 0.030350, 1: 0.029412},
           'fb_user_6'        : {-1: 0.099925, 0: 0.080432, 1: 0.068025},
           'fb_user_7'        : {-1: 0.116903, 0: 0.070099, 1: 0.055439},
           'fb_user_8'        : {-1: 0.142189, 0: 0.076923, 1: 0.024883}
          }
trainX.replace(mapping, inplace = True)


# In[ ]:


#Scaling on entire set 

# Scaling the data - Using Standard Scaler
col = list(trainX.columns)
# Data Scaling
scaler1 = StandardScaler()
trainX_scaled = scaler1.fit_transform(trainX)
testX_v1_scaled = scaler1.transform(test_v1)

# Applying Principal Component Analysis
pca = PCA()
pca.fit(trainX)
trainx_pca = pca.fit_transform(trainX_scaled)


# ### PCA

# In[ ]:


#  Plot variance Graph
var_cumu = np.cumsum(pca.explained_variance_ratio_)
fig = plt.figure(figsize=[12,8])
plt.vlines(x=15, ymax=1, ymin=0, colors="r", linestyles="--")
plt.hlines(y=0.95, xmax=30, xmin=0, colors="g", linestyles="--")
plt.plot(var_cumu)
plt.ylabel("Cumulative Variance Explained")
plt.xlabel("Number of pca components")
plt.show()


# In[ ]:


df_pca = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'PC3':pca.components_[2],'Feature':col})
df_pca.head(10)


# In[ ]:


np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)


# ### Handling Class Impbalance using SMOTE

# In[ ]:


smote = SMOTE(.8)
trainX_smote,trainY_smote = smote.fit_sample(trainX_scaled,trainY)
display("Shape of train datatset after SMOTE : "+ str(trainX_smote.shape))

# Applying Incremental PCA
pca = IncrementalPCA(n_components=80)    
trainX_smote_pca = pca.fit_transform(trainX_smote)
testX_v1_scaled_pca = pca.transform(testX_v1_scaled)


# In[ ]:


display("Shape of train datatset after PCA : "+str(trainX_smote_pca.shape))
from collections import Counter

display(Counter(trainY))
display(Counter(trainY_smote))


# ### RandomForest

# In[ ]:


# Random Forest Classifier - user same hyperparameters that we derived before on tain set

cv = KFold(n_splits=4, shuffle=True, random_state=1)

rf_test = RandomForestClassifier(random_state=42, n_jobs=-1,oob_score=True)
params = {
    'max_depth': [12],
    'min_samples_split' : [40],
    'min_samples_leaf' : [10,20],
    'max_features' : [40],
    'n_estimators' : [200]
}

rf_final_model = GridSearchCV(estimator=rf_test,
                           param_grid=params,
                           cv = cv,
                           n_jobs=-1, verbose=1, scoring="accuracy",return_train_score=True)
rf_final_model.fit(trainX_smote_pca, trainY_smote)

scores = pd.DataFrame(rf_final_model.cv_results_)



# In[ ]:


display("Train Accuracy:", accuracy_score(trainY_smote, rf_final_model.predict(trainX_smote_pca)))
display("Train Confusion Matrix:",confusion_matrix(trainY_smote, rf_final_model.predict(trainX_smote_pca)))


# ### AdaBoost

# In[ ]:


#Using adaBoosting
adaboost_1 =  AdaBoostClassifier(n_estimators=200, random_state=1)
adaboost_1.fit(trainX, trainY)


# In[ ]:


display('Accuracy of the Train model is:  ',accuracy_score(trainY, adaboost_1.predict(trainX)))


# In[ ]:


# Adaboost with Hyperparameter Tuning
param = {
        'n_estimators' : [50,100, 200], 
        'algorithm': ['SAMME', 'SAMME.R'],
        }

cv = KFold(n_splits=5, shuffle=True, random_state=1)

param_comb = 800

random_search_ada_1 = RandomizedSearchCV(adaboost_1, param_distributions=param, n_iter=param_comb, scoring='accuracy', n_jobs=-1, cv=cv, verbose=3, random_state=42)
random_search_ada_1.fit(trainX, trainY)


# In[ ]:


display('Accuracy of the Train model is:  ',accuracy_score(trainY, random_search_ada_1.predict(trainX)))


# ### XGBOOST

# In[ ]:


### XG Boost Model-1: fit model on training data with default hyperparameters

xgb_model = XGBClassifier(max_depth=2, n_estimators=200,n_jobs=-1,learning_rate=.3,subsample=.5)
xgb_model.fit(trainX_scaled, trainY)


# In[ ]:


# Model 2: hyperparameter tuning with XGBoost

# creating a KFold object 
cv = KFold(n_splits=5, shuffle=True, random_state=1)

# 'min_child_weight': [1, 5, 7, 10],
# 'gamma': [0.1, 0.5, 1, 1.5, 5]
# specify range of hyperparameters
param_grid = {'learning_rate': [0.1,0.2,0.3], 
             'subsample': [0.3,0.4,0.5]
             }          


# specify model
xgb_model_1 = XGBClassifier(max_depth=2, n_estimators=200,n_jobs=-1)

# set up GridSearchCV()
model_cv_1 = GridSearchCV(estimator = xgb_model_1, 
                        param_grid = param_grid, 
                        scoring= 'accuracy',
                        cv = folds, 
                        n_jobs = -1,
                        verbose = 1,
                        return_train_score=True)

# fit the model
model_cv_1.fit(trainX_smote_pca, trainY_smote)



# In[ ]:


# displaying the best accuracy score and hyperparameters
display('We  get best score of '+str(round(model_cv_1.best_score_,2)) +' using parameters: '+str(model_cv_1.best_params_))


# In[ ]:


# Model 3: with best score related hyperparameters
param = {'learning_rate': 0.3,
          'subsample':0.4,
          'max_depth': 3, 
          'n_estimators':200,          
          'gamma': 1,
         'objective':'binary:logistic'}

# fit model on training data
XGBmodel_2 = XGBClassifier(params = param,max_depth=2, n_estimators=200,min_child_weight=1,scale_pos_weight = 1)
XGBmodel_2.fit(trainX_smote_pca, trainY_smote)

display('Accuracy of the Train model is:  ',accuracy_score(trainY_smote, XGBmodel_2.predict(trainX_smote_pca)))


# ### Predict test results based on multiple model outputs

# In[ ]:


# Random Forest 
churn_probability_randomforest_1 = rf_final_model.predict(testX_v1_scaled_pca)
df['churn_probability_randomforest_1'] = pd.DataFrame(churn_probability_randomforest_1,columns=['churn_probability_randomforest_1'])
display(df.shape)


# In[ ]:


# XGBOOST 
churn_XGBOOST_1 = XGBmodel_2.predict(testX_v1_scaled_pca)
df['churn_XGBOOST_1'] = pd.DataFrame(churn_XGBOOST_1,columns=['churn_XGBOOST_1'])
display(df.shape)


# In[ ]:


#ADABOOST
churn_probability_adaboost_1 = random_search_ada_1.predict(test_v1)
df['churn_probability_adaboost_1'] = pd.DataFrame(churn_probability_adaboost_1,columns=['churn_probability_adaboost_1'])
display(df.shape)


# In[ ]:


df.head()


# ### Follow same method to compute final chrun probability as in previous method

# In[ ]:


df['total_churn_1'] = df['churn_XGBOOST_1'] + df['churn_probability_adaboost_1'] + df['churn_probability_randomforest_1']


# In[ ]:


df['Total_Case2'] = df['total_churn_1'].apply(lambda x : 1 if x>1 else 0)


# ### Final total sum and further derive the churn probability

# In[ ]:


df['Total_sum'] = df['Total_Case2'] + df['Total_Case1']


# In[ ]:


df['churn_probability'] = df['Total_sum'].apply(lambda x : 1 if x>=1 else 0)


# In[ ]:


df['churn_probability'].value_counts()


# In[ ]:


#Export to excel with final churn probability
df[['id','churn_probability']].to_csv('Submission.csv',index=False)


# ### Feature Importance using the Xgboost Model

# In[ ]:


xgb_model.feature_importances_


# In[ ]:


py.init_notebook_mode(connected=True) 


# In[ ]:


# Scatter plot
trace = go.Scatter(
y = xgb_model.feature_importances_, x = xtrain.columns.values, mode='markers',
marker=dict(
sizemode = 'diameter',
sizeref = 1.3,
size = 12,
color = xgb_model.feature_importances_, colorscale='Portland', showscale=True
),
text = xtrain.columns.values )
data = [trace]
layout= go.Layout( autosize= True,
title= 'XGBOOST Model Feature Importance', hovermode= 'closest',
xaxis= dict( ticklen= 5,
showgrid=False, zeroline=False, showline=False
), yaxis=dict(
title= 'Feature Importance', showgrid=False, zeroline=False,
ticklen= 5,
gridwidth= 2 ),
showlegend= False )
fig = go.Figure(data=data, layout=layout) 
py.iplot(fig,filename='scatter')


# In[ ]:


# Top 20 Features based on Feature Selection
#y = model_1.feature_importances_, x = xtrain.columns.values,
results = pd.DataFrame()
results['columns'] = xtrain.columns.values
results['importances'] = xgb_model.feature_importances_ *100
results.sort_values(by = 'importances', ascending = False, inplace=True)
display(results[:20])
results.to_csv("munirathinam_sreedhark.csv",index=False)


# # Conclusion
# - Telecom company should provide good offers to the customers who are using services from a roaming zone.
# - Telecom company should provide some kind of STD and ISD packages to reduce churn rate.

# In[ ]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

