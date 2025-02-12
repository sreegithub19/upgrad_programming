#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn - ML Group Case Study
# ##### By: Akshay Rohankar

# ###### Business Problem Overview
# In the telecom industry, customers are able to choose from multiple service providers and actively switch from one operator to another. In this highly competitive market, the telecommunications industry experiences an average of 15-25% annual churn rate. Given the fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.
# 
# 
# 
# For many incumbent operators, retaining high profitable customers is the number one business goal.
# 
# 
# 
# To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.
# 
# 
# 
# In this project, you will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.
# 
# ###### Definitions of Churn
# There are various ways to define churn, such as: 1. Revenue-based churn   2.Usage-based churn
# 
# For this project, you will use the **usage-based** definition to define churn.
# 
# **Usage-based churn:** Customers who have not done any usage, either incoming or outgoing - in terms of calls, internet etc. over a period of time.
# A potential shortcoming of this definition is that when the customer has stopped using the services for a while, it may be too late to take any corrective actions to retain them. For e.g., if you define churn based on a ‘two-months zero usage’ period, predicting churn could be useless since by that time the customer would have already switched to another operator.
# 
# ###### business objective:
# The business objective is to predict the churn in the last (i.e. the ninth) month using the data (features) from the first three months. To do this task well, understanding the typical customer behaviour during churn will be helpful.

# ###### Understanding Customer Behaviour During Churn
# Customers usually do not decide to switch to another competitor instantly, but rather over a period of time (this is especially applicable to high-value customers). In churn prediction, we assume that there are three phases of customer lifecycle :
# 
# The ‘good’ phase: In this phase, the customer is happy with the service and behaves as usual.
# 
# The ‘action’ phase: The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a  competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than the ‘good’ months. Also, it is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)
# 
# The ‘churn’ phase: In this phase, the customer is said to have churned. You define churn based on this phase. Also, it is important to note that at the time of prediction (i.e. the action months), this data is not available to you for prediction. Thus, after tagging churn as 1/0 based on this phase, you discard all data corresponding to this phase.
# 
# 
# 
# In this case, since you are working over a four-month window, the first two months are the ‘good’ phase, the third month is the ‘action’ phase, while the fourth month is the ‘churn’ phase.

# #### Data
# The dataset contains customer-level information for a span of four consecutive months - June, July, August and September. The months are encoded as 6, 7, 8 and 9, respectively.
# 
# **Filename:** telecom_churn_data.csv
# 
# 

# ---------

# In[ ]:


# Ignoring warning messages
import warnings
warnings.filterwarnings('ignore')

# Import the required library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns',230)


# In[ ]:


# reading the input data and preview
churn= pd.read_csv('https://github.com/akshayr89/Telecom_Churn_Model/raw/refs/heads/master/telecom_churn_data.csv')
churn.head()


# In[ ]:


print (churn.shape)
print (churn.info())
churn.describe()


# In[ ]:


print ("The cutomer-level information for each customer is represented by %d features"% (churn.shape[1]))
# getting the unique number of custormers from the data
print ("Unique customers/MSISDN in the data: %d"%len(churn.mobile_number.unique()))


# In[ ]:


#list of columns
pd.DataFrame(churn.columns)


# ---
# ##  Data Cleaning
# 
# ---

# Custome function Defination for data cleaning

# In[ ]:


def getMissingValues(missingCutoff):
    # Function to retun the columns with more than missingCutoff% missing values.
    # argument: missingCutoff, % values threshold for missing values
    missing = round(100*(churn.isnull().sum()/churn.shape[0]))
    print("There are {} features having more than {}% missing values/entries".format(len(missing.loc[missing > missingCutoff]),missingCutoff))
    return missing.loc[missing > missingCutoff]


# In[ ]:


def imputeNan(data,imputeColList=False,missingColList=False):
    # Function impute the nan with 0
    # argument: colList, list of columns for which nan is to be replaced with 0
    if imputeColList:
        for col in [y + s for s in ['_6','_7','_8','_9'] for y in imputeColList]:
            data[col].fillna(0, inplace=True)
    else:
        for col in missingColList:
            data[col].fillna(0, inplace=True)


# ##### Handling missing data
# Let's check for missing values in the data.

# In[ ]:


# Missing values per column expressed as % of total number of values
getMissingValues(50)


# Out the these 40 features, many are required and are essential for analysis. The missing values for these features seems to suggest that these customers KPI's did not have any value at that month. We can choose to impute these values with 0 to make enable these features to give value to analysis.

# In[ ]:


# Since av_rech_amt_data_* features are important for getting the high-value customers,
#lets impute the missing av_rech_amt_data_* with 0
imputeCol = ['av_rech_amt_data', 'arpu_2g', 'arpu_3g', 'count_rech_2g', 'count_rech_3g',
             'max_rech_data', 'total_rech_data','fb_user','night_pck_user']
imputeNan(churn,imputeCol)


# In[ ]:


getMissingValues(50)


# In[ ]:


# dropping the columns having more than 50% missing values
missingcol = list(getMissingValues(50).index)
churn.drop(missingcol,axis=1,inplace=True)
churn.shape


# In[ ]:


# Missing values per column expressed as % of total number of values > 5%
getMissingValues(5)


# Looks like all these features for the month sep(9) are missing together. Let's check.

# In[ ]:


# checking if all these above features go missing together since they have the same 8% missing values in each feature.
missingcol = list(getMissingValues(5).index)
print ("There are %d customers/MSISDN's having missing values for %s together"%(len(churn[churn[missingcol].isnull().all(axis=1)]),missingcol))
churn[churn[missingcol].isnull().all(axis=1)][missingcol].head()


# Yes, It looks like for **7745 Customers** all these features are empty together without any value. We can choose to impute these values with 0 also.

# In[ ]:


imputeNan(churn,missingColList=missingcol)


# In[ ]:


churn=churn[~churn[missingcol].isnull().all(axis=1)]
churn.shape


# In[ ]:


# Missing values per column expressed as % of total number of values
getMissingValues(2)


# In[ ]:


missingcol = list(getMissingValues(2).index)
print ("There are %d customers/MSISDN's having missing values for %s together"%(len(churn[churn[missingcol].isnull().all(axis=1)]),missingcol))
churn[churn[missingcol].isnull().all(axis=1)][missingcol].head()


# Yes, It looks like there are **381 Customers** for whom **all** these features are without any value.
# Let's drop these customers from the data.

# In[ ]:


churn=churn[~churn[missingcol].isnull().all(axis=1)]
churn.shape


# In[ ]:


# For other customers where these missing values are spread out, let's impute them with zero.

missingcol.remove('date_of_last_rech_8')
missingcol.remove('date_of_last_rech_9')
imputeNan(churn,missingColList=missingcol)


# In[ ]:


# Missing values per column expressed as % of total number of values
getMissingValues(0)


# In[ ]:


col = ['loc_og_t2o_mou','std_og_t2o_mou','loc_ic_t2o_mou','last_date_of_month_7','last_date_of_month_8','last_date_of_month_9', 'date_of_last_rech_7', 'date_of_last_rech_8', 'date_of_last_rech_9']
for c in col:
    print("Unique values in column %s are %s" % (c,churn[c].unique()))


# In[ ]:


#Some of these features take only one value. Lets impute their missing values in these features with the mode
col = ['loc_og_t2o_mou','std_og_t2o_mou','loc_ic_t2o_mou','last_date_of_month_7','last_date_of_month_8','last_date_of_month_9']
for c in col:
    print(churn[c].value_counts())
    churn[c].fillna(churn[c].mode()[0], inplace=True)
print("All the above features take only one value. Lets impute the missing values in these features with the mode")


# In[ ]:


# Missing values per column expressed as % of total number of values
getMissingValues(0)


# In[ ]:


# All these features are missing together
missingcol = list(getMissingValues(0).index)
print ("There are %d rows in total having missing values for these variables."%(len(churn[churn[missingcol].isnull().all(axis=1)])))


# In[ ]:


churn[churn['date_of_last_rech_6'].isnull()]['date_of_last_rech_6'] = '6/30/2014'
churn[churn['date_of_last_rech_7'].isnull()]['date_of_last_rech_7'] = '7/31/2014'
churn[churn['date_of_last_rech_8'].isnull()]['date_of_last_rech_8'] = '8/31/2014'
churn[churn['date_of_last_rech_9'].isnull()]['date_of_last_rech_9'] = '9/30/2014'


# <br><br>Let's look for columns having all values as 0.

# In[ ]:


zero_columns=churn.columns[(churn == 0).all()]
print ("There are {} features which has only 0 as values. These features are \n{}".format(len(zero_columns),zero_columns))


# In[ ]:


# Let's remove these columns as well. All take a single value '0'.
churn.drop(zero_columns,axis=1,inplace=True)


# In[ ]:


# Percentage of data left after removing the missing values.
print("Percentage of data remaining after treating missing values: {}%".format(round(churn.shape[0]/99999 *100,2)))
print ("Number of customers: {}".format(churn.shape[0]))
print ("Number of features: {}".format(churn.shape[1]))


# ##### Fixing data types and columns names
# 
# Let's check for data types of the different columns.

# In[ ]:


churn.reset_index(inplace=True,drop=True)
# list of all columns which store date
date_columns = list(churn.filter(regex='date').columns)
date_columns


# In[ ]:


# Converting dtype of date columns to datetime
for col in date_columns:
    churn[col] = pd.to_datetime(churn[col], format='%m/%d/%Y')


# In[ ]:


churn.info()


# <br><br> There are some monthly features which are not in the standard naming (\_6,\_7,\_8,\_9)

# In[ ]:


# renaming columns,
#'jun_vbc_3g' : 'vbc_3g_6'
#'jul_vbc_3g' : 'vbc_3g_7'
#'aug_vbc_3g' : 'vbc_3g_8'
#'sep_vbc_3g' : 'vbc_3g_9'
churn.rename(columns={'jun_vbc_3g' : 'vbc_3g_6', 'jul_vbc_3g' : 'vbc_3g_7', 'aug_vbc_3g' : 'vbc_3g_8',
                      'sep_vbc_3g' : 'vbc_3g_9'}, inplace=True)


# **Creating new feature:** 'vol_data_mb_6', 'vol_data_mb_7', 'vol_data_mb_8', 'vol_data_mb_9'
# 
# These will store the total data volume (= vol_2g_mb_* + vol_3g_mb_*) used by user.

# In[ ]:


#Creating new feature: 'vol_data_mb_6', 'vol_data_mb_7', 'vol_data_mb_8', 'vol_data_mb_9',
for i in range(6,10):
    churn['vol_data_mb_'+str(i)] = (churn['vol_2g_mb_'+str(i)]+churn['vol_3g_mb_'+str(i)]).astype(int)


# ###### Filter high-value customers
# Defining high-value customers as follows:
# 
# Those who have recharged with an amount more than or equal to X, where X is the 70th percentile of the average recharge amount in the first two months (the good phase).

# In[ ]:


rechcol = churn.filter(regex=('count')).columns
churn[rechcol].head()


# **Creating new feature:** avg_rech_amt_6,avg_rech_amt_7,avg_rech_amt_8,avg_rech_amt_9
# 
# These will store the average recharge value for each customer for every month

# In[ ]:


# Creating new feature: avg_rech_amt_6,avg_rech_amt_7,avg_rech_amt_8,avg_rech_amt_9
for i in range(6,10):
    churn['avg_rech_amt_'+str(i)] = round(churn['total_rech_amt_'+str(i)]/churn['total_rech_num_'+str(i)]+1,2)


# In[ ]:


imputeNan(churn,missingColList=['avg_rech_amt_6','avg_rech_amt_7','avg_rech_amt_8','avg_rech_amt_9'])


# **Creating new feature:** total_rech_num_data_6,total_rech_num_data_7,total_rech_num_data_8,total_rech_num_data_9
# 
# These will store the total number of data recharge (=count_rech_2g + count_rech_3g ) for each month.

# In[ ]:


#Creating new feature: total_rech_num_data_6,total_rech_num_data_7,total_rech_num_data_8,total_rech_num_data_9
for i in range(6,10):
    churn['total_rech_num_data_'+str(i)] = (churn['count_rech_2g_'+str(i)]+churn['count_rech_3g_'+str(i)]).astype(int)


# **Creating new feature:** total_rech_amt_data_6,total_rech_amt_data_7,total_rech_amt_data_8,total_rech_amt_data_9
# 
# These will store the total amount of data recharge (=total_rech_num_data * av_rech_amt_data ) for each month.

# In[ ]:


#Creating new feature: total_rech_amt_data_6,total_rech_amt_data_7,total_rech_amt_data_8,total_rech_amt_data_9
for i in range(6,10):
    churn['total_rech_amt_data_'+str(i)] = churn['total_rech_num_data_'+str(i)]*churn['av_rech_amt_data_'+str(i)]


# **Creating new feature:** total_month_rech_6,total_month_rech_7,total_month_rech_8,total_month_rech_9
# 
# These will store the total recharge amount (= total_rech_amt + total_rech_amt_data ) for each customer, for each month.

# In[ ]:


#Creating new feature: total_mon_rech_6,total_mon_rech_7,total_mon_rech_8,total_mon_rech_9
for i in range(6,10):
    churn['total_month_rech_'+str(i)] = churn['total_rech_amt_'+str(i)]+churn['total_rech_amt_data_'+str(i)]
churn.filter(regex=('total_month_rech')).head()


# In[ ]:


# calculating the avegare of first two months (good phase) total monthly recharge amount
avg_goodPhase =(churn.total_month_rech_6 + churn.total_month_rech_7)/2
# finding the cutoff which is the 70th percentile of the good phase average recharge amounts
hv_cutoff= np.percentile(avg_goodPhase,70)
# Filtering the users whose good phase avg. recharge amount >= to the cutoff of 70th percentile.
hv_users = churn[avg_goodPhase >=  hv_cutoff]
hv_users.reset_index(inplace=True,drop=True)

print("Number of High-Value Customers in the Dataset: %d\n"% len(hv_users))
print("Percentage High-value users in data : {}%".format(round(len(hv_users)/churn.shape[0]*100),2))


# ###### Tagging Churners
# Now tag the churned customers (churn=1, else 0) based on the fourth month as follows:
# 
# Those who have not made any calls (either incoming or outgoing) AND have not used mobile internet even once in the churn phase. The attributes we need to use to tag churners are:
# - total_ic_mou_9
# - total_og_mou_9
# - vol_2g_mb_9
# - vol_3g_mb_9

# In[ ]:


def getChurnStatus(data,churnPhaseMonth=9):
    # Function to tag customers as churners (churn=1, else 0) based on 'vol_2g_mb_','vol_3g_mb_','total_ic_mou_','total_og_mou_'
    #argument: churnPhaseMonth, indicating the month number to be used to define churn (default= 9)
    churn_features= ['vol_2g_mb_','vol_3g_mb_','total_ic_mou_','total_og_mou_']
    flag = ~data[[s + str(churnPhaseMonth) for s in churn_features ]].any(axis=1)
    flag = flag.map({True:1, False:0})
    return flag


# In[ ]:


hv_users['churn'] = getChurnStatus(hv_users,9)
print("There are {} users tagged as churners out of {} High-Value Customers.".format(len(hv_users[hv_users.churn == 1]),hv_users.shape[0]))
print("High-value Churn Percentage : {}%".format(round(len(hv_users[hv_users.churn == 1])/hv_users.shape[0] *100,2)))


# <br>There are just **8.09% churn** cases.
# <br>This indicated an **highly imbalanced** data set where the churn cases are the minority(8.14%) as opposed to the non-churners who are the majority(91.91)

# ---
# ##  Data Analysis
# 
# ---

# Define few methods to aid in plotting graphs

# In[ ]:


# Function to plot the histogram with labels
# https://stackoverflow.com/questions/6352740/matplotlib-label-each-bin
def plot_hist(dataset,col,binsize):
    fig, ax = plt.subplots(figsize=(20,4))
    counts, bins, patches = ax.hist(dataset[col],bins=range(0,dataset[col].max(),round(binsize)), facecolor='lightgreen', edgecolor='gray')

    # Set the ticks to be at the edges of the bins.
    ax.set_xticks(bins)
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        # Label the percentages
        percent = '%0.0f%%' % (100 * float(count) / counts.sum())
        ax.annotate(percent, xy=(x,0.2), xycoords=('data', 'axes fraction'),
        xytext=(0, -32), textcoords='offset points', va='top', ha='center')

    ax.set_xlabel(col.upper())
    ax.set_ylabel('Count')
    # Give ourselves some more room at the bottom of the plot
    #plt.subplots_adjust(bottom=0.15)
    plt.show()


# In[ ]:


def plot_avgMonthlyCalls(pltType,data,calltype,colList):
    # style
    plt.style.use('ggplot')
    # create a color palette
    palette = plt.get_cmap('Set1')

    if pltType == 'multi':
        #Create dataframe after grouping on AON with colList features
        total_call_mou = pd.DataFrame(data.groupby('aon_bin',as_index=False)[colList].mean())
        total_call_mou['aon_bin']=pd.to_numeric(total_call_mou['aon_bin'])
        total_call_mou
        # multiple line plot
        num=0
        fig, ax = plt.subplots(figsize=(15,8))
        for column in total_call_mou.drop('aon_bin', axis=1):
            num+=1
            ax.plot(total_call_mou['aon_bin'] , total_call_mou[column], marker='', color=palette(num), linewidth=2, alpha=0.9, label=column)

        ## Add legend
        plt.legend(loc=2, ncol=2)
        ax.set_xticks(total_call_mou['aon_bin'])

        # Add titles
        plt.title("Avg.Monthly "+calltype+" MOU  V/S AON", loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("Aon (years)")
        plt.ylabel("Avg. Monthly "+calltype+" MOU")
    elif pltType == 'single':
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(data[colList].mean())
        ax.set_xticklabels(['Jun','Jul','Aug','Sep'])

        # Add titles
        plt.title("Avg. "+calltype+" MOU  V/S Month", loc='left', fontsize=12, fontweight=0, color='orange')
        plt.xlabel("Month")
        plt.ylabel("Avg. "+calltype+" MOU")

    plt.show()


# In[ ]:


def plot_byChurnMou(colList,calltype):
    fig, ax = plt.subplots(figsize=(7,4))
    df=hv_users.groupby(['churn'])[colList].mean().T
    plt.plot(df)
    ax.set_xticklabels(['Jun','Jul','Aug','Sep'])
    ## Add legend
    plt.legend(['Non-Churn', 'Churn'])
    # Add titles
    plt.title("Avg. "+calltype+" MOU  V/S Month", loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel("Month")
    plt.ylabel("Avg. "+calltype+" MOU")


# In[ ]:


def plot_byChurn(data,col):
    # per month churn vs Non-Churn
    fig, ax = plt.subplots(figsize=(7,4))
    colList=list(data.filter(regex=(col)).columns)
    colList = colList[:3]
    plt.plot(hv_users.groupby('churn')[colList].mean().T)
    ax.set_xticklabels(['Jun','Jul','Aug','Sep'])
    ## Add legend
    plt.legend(['Non-Churn', 'Churn'])
    # Add titles
    plt.title( str(col) +" V/S Month", loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel("Month")
    plt.ylabel(col)
    plt.show()
    # Numeric stats for per month churn vs Non-Churn
    return hv_users.groupby('churn')[colList].mean()


# In[ ]:


# Filtering the common monthly columns for each month
comcol = hv_users.filter(regex ='_6').columns
monthlycol = [item.strip('_6') for item in comcol]
monthlycol


# In[ ]:


# getting the number of monthly columns and profile columns
print ("Total number of columns in data :", hv_users.shape[1] )
print ("Number of columns for each month : ",len(monthlycol))
print ("Total monthly columns among the orignal columns (%d*4): %d"%(len(monthlycol), len(monthlycol) * 4))
print ("Columns other than monthly columns :", hv_users.shape[1] - (len(monthlycol) * 4))


# In[ ]:


# Lets remove all the attributes corresponding to the churn phase (all attributes having ‘ _9’, etc. in their names).
col_9List = hv_users.filter(regex=('_9')).columns
hv_users.drop(col_9List,axis=1,inplace=True)


# In[ ]:


# list of all the monthly columns 6,7,8,9
allmonthlycol = [x + s for s in ['_6','_7','_8'] for x in monthlycol]
allmonthlycol


# In[ ]:


# list of column which are not monthly columns
nonmonthlycol = [col for col in hv_users.columns if col not in allmonthlycol]
nonmonthlycol


# ###### Feature: circle_id

# In[ ]:


# Getting the distinct circle_id's in the data
hv_users.circle_id.value_counts()


# Looks like the data at hand is only for a single **circle_id 109.** <br>We can remove this feature going forward as it is not contributing to analysis and model building.

# In[ ]:


hv_users.drop('circle_id',axis=1,inplace=True)


# ###### Feature: aon

# In[ ]:


# Customers distribution of the age on network
print(hv_users.aon.describe())
plot_hist(hv_users,'aon',365)


# - **Minimun Age** on network is 180 days.
# - **Average age** on network for customers is 1200 days (3.2 years).
# - 27% of the **HV users are in their 2nd year** with the network.
# - Almost 71% users have Age on network **less than 4 years.**
# - 15% users are with the network from **over 7 years.**

# In[ ]:


#Create Derived categorical variable
hv_users['aon_bin'] = pd.cut(churn['aon'], range(0,churn['aon'].max(),365), labels=range(0,int(round(churn['aon'].max()/365))-1))


# ###### Incoming VS month VS AON

# In[ ]:


# Plotting Avg. total monthly incoming MOU vs AON
ic_col = hv_users.filter(regex ='total_ic_mou').columns
plot_avgMonthlyCalls('single',hv_users,calltype='incoming',colList=ic_col)
plot_avgMonthlyCalls('multi',hv_users,calltype='incoming',colList=ic_col)


# It is evident from the plot that,
# - The more a customer stays on with the operator(AON), more are the total monthly incoming MOU.
# - Total Incoming MOU avg. for Jul(_7) are more than the previous Jun(_6) for customers in all AON bands.
# - Total Incoming MOU avg. for Aug(_8) cease to increace, infact it shows a decline compared to Jul(_7).
# - Total Incoming MOU avg. for Sep(_9) is well below the first months(jun _6) avg.
# - Althought the Total incoming mou avg inceases from jun to july, it drop little from aug and reduces lower than that for jun.

# ###### Outgoing VS month VS AON

# In[ ]:


# Plotting Avg. total monthly outgoing MOU vs AON
og_col = hv_users.filter(regex ='total_og_mou').columns
plot_avgMonthlyCalls('single',hv_users,calltype='outgoing',colList=og_col)
plot_avgMonthlyCalls('multi',hv_users,calltype='outgoing',colList=og_col)


# What is the above plot saying ?
# - Overall, the Avg. total outgoing usage reduces with the increasing age on network.
# - Total Outgoing MOU avg. for Jul(_7) are more than the previous Jun(_6) for customers in all AON bands, except in the AON band between 7 - 8 years where it is almost simillar.
# - Total outgoing MOU avg. for Aug(_8) cease to increace, infact it shows a significant decline compared to Jul(_7).
# - Total outgoing MOU avg. for Sep(_9) is the lowest of all 4 months.
# - The Avg. outgoing usage reduces drastically for customers in the AON band between 7 - 8  years.

# ###### Incoming/Outgoing MOU VS Churn

# In[ ]:


ic_col = ['total_ic_mou_6','total_ic_mou_7','total_ic_mou_8']
og_col = ['total_og_mou_6','total_og_mou_7','total_og_mou_8']
plot_byChurnMou(ic_col,'Incoming')
plot_byChurnMou(og_col,'Outgoing')


# It can be observed,
# - Churners Avg. Incoming/Outgoing MOU's **drops drastically after the 2nd month,Jul.**
# - While the non-churners Avg. MOU's remains consistant and stable with each month.
# - Therefore, users MOU is a key feature to predict churn.

# Let's also see this trend in terms of actual numbers.

# In[ ]:


# Avg.Incoming MOU per month churn vs Non-Churn
hv_users.groupby(['churn'])[['total_ic_mou_6','total_ic_mou_7','total_ic_mou_8']].mean()


# In[ ]:


# Avg. Outgoing MOU per month churn vs Non-Churn
hv_users.groupby(['churn'])[['total_og_mou_6','total_og_mou_7','total_og_mou_8']].mean()


# **Create new feature:** og_to_ic_mou_6, og_to_ic_mou_7, og_to_ic_mou_8
# These features will hold the **ratio** (=total_og_mou_* / total_ic_mou_*) for each month. These features will combine both incoming and outgoing informations and should be a **better predictor of churn.**

# In[ ]:


#Creating new feature: og_to_ic_mou_6, og_to_ic_mou_7, og_to_ic_mou_8
# adding 1 to denominator to avoid dividing by 0 and getting nan values.
for i in range(6,9):
    hv_users['og_to_ic_mou_'+str(i)] = (hv_users['total_og_mou_'+str(i)])/(hv_users['total_ic_mou_'+str(i)]+1)


# In[ ]:


plot_byChurn(hv_users,'og_to_ic_mou')


# - Outgoing to incoming mou remains drops significantly for churners from month Jul(6) to Aug(7).
# - While it remains almost consistent for the non-churners.

# **Create new feature:** loc_og_to_ic_mou_6, loc_og_to_ic_mou_7, loc_og_to_ic_mou_8
# These features will hold the **ratio** (=loc_og_mou_* / loc_ic_mou_*) for each month. These features will combine the local calls, both incoming and outgoing informations and should be a **better predictor of churn.**

# In[ ]:


#Create new feature: loc_og_to_ic_mou_6, loc_og_to_ic_mou_7, loc_og_to_ic_mou_8
# adding 1 to denominator to avoid dividing by 0 and getting nan values.
for i in range(6,9):
    hv_users['loc_og_to_ic_mou_'+str(i)] = (hv_users['loc_og_mou_'+str(i)])/(hv_users['loc_ic_mou_'+str(i)]+1)


# In[ ]:


plot_byChurn(hv_users,'loc_og_to_ic_mou')


# It can be observed that,
# - The local outgoing to incoming call mou ratio is genrally low for churners right from the begining of the good phase.
# - local mou pattern for the non-churners remains almost constant through out the 3 months.
# - The churners genrally show a low loc mou ratio but it drops dramatically after the 2nd month.
# - This might suggest that people who are not making/reciving much local calls during their tenure are more likely to churn.

# ###### Total data volume VS Churn

# In[ ]:


plot_byChurn(hv_users,'vol_data_mb')


# - The volume of data mb used drops significantly for churners from month Jul(6) to Aug(7).
# - While it remains almost consistent for the non-churners.

# ###### Total monthly rech VS Churn

# In[ ]:


plot_byChurn(hv_users,'total_month_rech')


# - total monthly rech amount also drops significantly for churners from month Jul(6) to Aug(7).
# - While it remains almost consistent for the non-churners.

# ###### max_rech_amt VS Churn

# In[ ]:


plot_byChurn(hv_users,'max_rech_amt')


# - maximum recharge amount also drops significantly for churners from month Jul(6) to Aug(7).
# - While it remains almost consistent for the non-churners.

# ###### arpu VS Churn

# In[ ]:


plot_byChurn(hv_users,'arpu')


# - Average revenue per user,arpu also drops significantly for churners from month Jul(6) to Aug(7).
# - While it remains almost consistent for the non-churners.

# **Create new feature:** Total_loc_mou_6, Total_loc_mou_7, Total_loc_mou_8<br>
# These features will hold the **Total MOU** (=loc_og_mou+loc_ic_mou) for each month.<br>
# Using this we will find if the loc MOU (both incoming and outgoing) drops or increaces as the months goes by.<br>
# This informations should be a **better predictor of churn.**

# In[ ]:


#Create new feature: Total_loc_mou_6,Total_loc_mou_7,lTotal_loc_mou_8
for i in range(6,9):
    hv_users['Total_loc_mou_'+str(i)] = (hv_users['loc_og_mou_'+str(i)])+(hv_users['loc_ic_mou_'+str(i)])


# In[ ]:


plot_byChurn(hv_users,'Total_loc_mou_')


# It can be observed that,
# - The Total local call mou is genrally low for churners right from the begining of the good phase.
# - local mou pattern for the non-churners remains almost constant through out the 3 months.
# - The churners genrally show a low total loc mou but it drops dramatically after the 2nd month.
# - This might suggest that people who are not making/reciving much local calls during their tenure are more likely to churn.

# **Create new feature:** Total_roam_mou_6,Total_roam_mou_7,Total_roam_mou_8<br>
# These features will hold the **Total roaming MOU** (=roam_ic_mou+roam_og_mou) for each month.<br>
# Using this we will find if the roam MOU (both incoming and outgoing) drops or increaces as the months goes by.<br>
# This informations should be a **better predictor of churn.**

# In[ ]:


#Create new feature: Total_roam_mou_6,Total_roam_mou_7,Total_roam_mou_8
for i in range(6,9):
    hv_users['Total_roam_mou_'+str(i)] = (hv_users['roam_ic_mou_'+str(i)])+(hv_users['roam_og_mou_'+str(i)])


# In[ ]:


plot_byChurn(hv_users,'Total_roam_mou')


# It can be observed that,
# - Surprisingly, the roaming usage of churners is way higher than those of non-churners across all months
# - People who are making/reciving more roaming calls during their tenure are more likely to churn.
# - This might suggest that the operators roaming tariffs are higher than what are offered by its competitor, thus forming one of the reasons of churn.

# ###### last_day_rch_amt VS Churn

# In[ ]:


plot_byChurn(hv_users,'last_day_rch_amt')


# - The avg. last recharge amount for churners is less than half the amount of that of the non-churners.
# - Suggesting, as the recharge amount reduces for a customer its chances to churn increases.

# ## Modeling

# In[ ]:


import sklearn.preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[ ]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 6))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds


# In[ ]:


def getModelMetrics(actual_churn=False,pred_churn=False):

    confusion = metrics.confusion_matrix(actual_churn, pred_churn)

    TP = confusion[1,1] # true positive
    TN = confusion[0,0] # true negatives
    FP = confusion[0,1] # false positives
    FN = confusion[1,0] # false negatives

    print("Roc_auc_score : {}".format(metrics.roc_auc_score(actual_churn,pred_churn)))
    # Let's see the sensitivity of our logistic regression model
    print('Sensitivity/Recall : {}'.format(TP / float(TP+FN)))
    # Let us calculate specificity
    print('Specificity: {}'.format(TN / float(TN+FP)))
    # Calculate false postive rate - predicting churn when customer does not have churned
    print('False Positive Rate: {}'.format(FP/ float(TN+FP)))
    # positive predictive value
    print('Positive predictive value: {}'.format(TP / float(TP+FP)))
    # Negative predictive value
    print('Negative Predictive value: {}'.format(TN / float(TN+ FN)))
    # sklearn precision score value
    print('sklearn precision score value: {}'.format(metrics.precision_score(actual_churn, pred_churn )))



# In[ ]:


def predictChurnWithProb(model,X,y,prob):
    # Funtion to predict the churn using the input probability cut-off
    # Input arguments: model instance, x and y to predict using model and cut-off probability

    # predict
    pred_probs = model.predict_proba(X)[:,1]

    y_df= pd.DataFrame({'churn':y, 'churn_Prob':pred_probs})
    # Creating new column 'predicted' with 1 if Churn_Prob>0.5 else 0
    y_df['final_predicted'] = y_df.churn_Prob.map( lambda x: 1 if x > prob else 0)
    # Let's see the head
    getModelMetrics(y_df.churn,y_df.final_predicted)
    return y_df


# In[ ]:


def findOptimalCutoff(df):
    #Function to find the optimal cutoff for classifing as churn/non-churn
    # Let's create columns with different probability cutoffs
    numbers = [float(x)/10 for x in range(10)]
    for i in numbers:
        df[i] = df.churn_Prob.map( lambda x: 1 if x > i else 0)
    #print(df.head())

    # Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
    cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
    from sklearn.metrics import confusion_matrix

    # TP = confusion[1,1] # true positive
    # TN = confusion[0,0] # true negatives
    # FP = confusion[0,1] # false positives
    # FN = confusion[1,0] # false negatives

    num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for i in num:
        cm1 = metrics.confusion_matrix(df.churn, df[i] )
        total1=sum(sum(cm1))
        accuracy = (cm1[0,0]+cm1[1,1])/total1

        speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
        sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
        cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
    print(cutoff_df)
    # Let's plot accuracy sensitivity and specificity for various probabilities.
    cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
    plt.show()


# In[ ]:


def modelfit(alg, X_train, y_train, performCV=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(X_train, y_train)

    #Predict training set:
    dtrain_predictions = alg.predict(X_train)
    dtrain_predprob = alg.predict_proba(X_train)[:,1]

    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, X_train, y_train, cv=cv_folds, scoring='roc_auc')

    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.roc_auc_score(y_train, dtrain_predictions))
    print ("Recall/Sensitivity : %.4g" % metrics.recall_score(y_train, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, dtrain_predprob))

    if performCV:
        print ("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))


# In[ ]:


# creating copy of the final hv_user dataframe
hv_users_PCA = hv_users.copy()
# removing the columns not required for modeling
hv_users_PCA.drop(['mobile_number', 'aon_bin'], axis=1, inplace=True)


# In[ ]:


# removing the datatime columns before PCA
dateTimeCols = list(hv_users_PCA.select_dtypes(include=['datetime64']).columns)
print(dateTimeCols)
hv_users_PCA.drop(dateTimeCols, axis=1, inplace=True)


# In[ ]:


from sklearn.model_selection import train_test_split

#putting features variables in X
X = hv_users_PCA.drop(['churn'], axis=1)

#putting response variables in Y
y = hv_users_PCA['churn']

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)


# In[ ]:


#Rescaling the features before PCA as it is sensitive to the scales of the features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[ ]:


# fitting and transforming the scaler on train
X_train = scaler.fit_transform(X_train)
# transforming the train using the already fit scaler
X_test = scaler.transform(X_test)


# ### Handling class imbalance.

# Standard classifier algorithms like Decision Tree and Logistic Regression have a bias towards classes which have number of instances. They tend to only predict the majority class data. The features of the minority class are treated as noise and are often ignored. Thus, there is a high probability of misclassification of the minority class as compared to the majority class.

# **Informed Over Sampling: Synthetic Minority Over-sampling Technique**
# 
# This technique is followed to avoid overfitting which occurs when exact replicas of minority instances are added to the main dataset. A subset of data is taken from the minority class as an example and then new synthetic similar instances are created. These synthetic instances are then added to the original dataset. The new dataset is used as a sample to train the classification models.
# 
# **Advantages**
# - Mitigates the problem of overfitting caused by random oversampling as synthetic examples are generated rather than replication of instances
# - No loss of useful information

# In[ ]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
print("Before OverSampling, churn event rate : {}% \n".format(round(sum(y_train==1)/len(y_train)*100,2)))


# In[ ]:


from imblearn.over_sampling import SMOTE

# Initialize SMOTE with a sampling strategy
sm = SMOTE(random_state=12, sampling_strategy=1.0)

# Fit and resample the data
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Resampling completed successfully!")


# In[ ]:


print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
print("After OverSampling, churn event rate : {}% \n".format(round(sum(y_train_res==1)/len(y_train_res)*100,2)))


# In[ ]:


#Improting the PCA module
from sklearn.decomposition import PCA
pca = PCA(svd_solver='randomized', random_state=42)


# In[ ]:


#Doing the PCA on the train data
pca.fit(X_train_res)


# we'll let PCA select the number of components basen on a variance cutoff we provide

# In[ ]:


# let PCA select the number of components basen on a variance cutoff
#pca_again = PCA(0.9)


# In[ ]:


#df_train_pca2 = pca_again.fit_transform(X_train_res)
#df_train_pca2.shape
# we see that PCA selected 12 components


# In[ ]:


#X_train_pca = pca_again.fit_transform(X_train_res)
#X_train_pca.shape


# In[ ]:


#Applying selected components to the test data - 50 components
#X_test_pca = pca_again.transform(X_test)
#X_test_pca.shape


#  **Looking at the screeplot to assess the number of needed principal components**

# In[ ]:


pca.explained_variance_ratio_[:50]


# In[ ]:


#Making the screeplot - plotting the cumulative variance against the number of components
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize = (12,8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# ##### **Looks like 50 components are enough to describe 95% of the variance in the dataset**
# - We'll choose 50 components for our modeling

# In[ ]:


#Using incremental PCA for efficiency - saves a lot of time on larger datasets
from sklearn.decomposition import IncrementalPCA
pca_final = IncrementalPCA(n_components=35)


# In[ ]:


X_train_pca = pca_final.fit_transform(X_train_res)
X_train_pca.shape


# In[ ]:


#creating correlation matrix for the principal components
corrmat = np.corrcoef(X_train_pca.transpose())
# 1s -> 0s in diagonals
corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)
# we see that correlations are indeed very close to 0


# Indeed - there is no correlation between any two components! We effectively have removed multicollinearity from our situation, and our models will be much more stable

# In[ ]:


#Applying selected components to the test data - 50 components
X_test_pca = pca_final.transform(X_test)
X_test_pca.shape


# For the prediction of churn customers we will be fitting variety of models and select one which is the best predictor of churn. Models trained are,
#     1. Logistic Regression
#     2. Decision Tree
#     3. Random Forest
#     4. Boosting models - Gradient Boosting Classifier and XGBoost Classifier
#     5. SVM

# ### 1. Logistic Regression

# ##### Applying Logistic Regression on our principal components

# In[ ]:


#Training the model on the train data
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

lr0 = LogisticRegression(class_weight='balanced')


# In[ ]:


modelfit(lr0, X_train_pca, y_train_res)


# In[ ]:


# predictions on Test data
pred_probs_test = lr0.predict(X_test_pca)
getModelMetrics(y_test,pred_probs_test)


# In[ ]:


print("Accuracy : {}".format(metrics.accuracy_score(y_test,pred_probs_test)))
print("Recall : {}".format(metrics.recall_score(y_test,pred_probs_test)))
print("Precision : {}".format(metrics.precision_score(y_test,pred_probs_test)))


# In[ ]:


#Making prediction on the test data
pred_probs_train = lr0.predict_proba(X_train_pca)[:,1]
print("roc_auc_score(Train) {:2.2}".format(metrics.roc_auc_score(y_train_res, pred_probs_train)))


# In[ ]:


cut_off_prob=0.5
y_train_df = predictChurnWithProb(lr0,X_train_pca,y_train_res,cut_off_prob)
y_train_df.head()


# **Plotting the ROC Curve :**
# An ROC curve demonstrates several things:
# 
# - It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in specificity).
# - The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
# - The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

# In[ ]:


draw_roc(y_train_df.churn, y_train_df.final_predicted)


# The roc curve is lying in the top left corner which is a sign of a good fit.

# In[ ]:


#draw_roc(y_pred_final.Churn, y_pred_final.predicted)
print("roc_auc_score : {:2.2f}".format(metrics.roc_auc_score(y_train_df.churn, y_train_df.final_predicted)))


# **Finding Optimal Cutoff Point**<br>
# Since recall or sensitivity is a much more important metrics for churn prediction. A trade off between sensitivity(or recall) and specificity is to be considered in doing so. We will try adjusting the probability threshold which shall lead to higher sensitivity or recall rate.

# In[ ]:


# finding cut-off with the right balance of the metrices
# sensitivity vs specificity trade-off
findOptimalCutoff(y_train_df)


# #### **From the curve above, 0.45 is the optimum point .**
# Although, other cutoff between 0.4 and 0.6 can also be taken but to keep the test sensitivity/recall significant we choose 0.45. At this point there is a balance of sensitivity, specificity and accuracy.

# In[ ]:


# predicting with the choosen cut-off on train
cut_off_prob = 0.45
predictChurnWithProb(lr0,X_train_pca,y_train_res,cut_off_prob)


# **Making prediction on test**

# In[ ]:


# predicting with the choosen cut-off on test
predictChurnWithProb(lr0,X_test_pca,y_test,cut_off_prob)


# The resulting model, after PCA and logistic regression (with optimal cutoff setting) has a right balance of different metrics score for sensitivity, specificity and Roc Accuracy on the train and test set.
# - **train sensitivity  :** 86.47%, **train roc auc score  :** 82.1%
# - **test sensitivity   :** 84.40%, **test roc auc score  :** 81.21%

# ### 2. Decision Tree

# ##### Applying Decision Tree Classifier on our principal components with Hyperparameter tuning

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

# Initialize the DecisionTreeClassifier with valid parameters
dt0 = DecisionTreeClassifier(class_weight='balanced',
                             max_features='sqrt',  # Use 'sqrt' instead of 'auto'
                             min_samples_split=100,
                             min_samples_leaf=100,
                             max_depth=6,
                             random_state=10)

# Fit the model (assuming modelfit is a custom function you have)
modelfit(dt0, X_train_pca, y_train_res)


# In[ ]:


# make predictions
pred_probs_test = dt0.predict(X_test_pca)
#Let's check the model metrices.
getModelMetrics(actual_churn=y_test,pred_churn=pred_probs_test)


# In[ ]:


# Create the parameter grid based on the results of random search
param_grid = {
    'max_depth': range(5,15,3),
    'min_samples_leaf': range(100, 400, 50),
    'min_samples_split': range(100, 400, 100),
    'max_features': [8,10,15]
}
# Create a based model
dt = DecisionTreeClassifier(class_weight='balanced',random_state=10)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = dt, param_grid = param_grid,
                          cv = 3, n_jobs = 4,verbose = 1,scoring="f1_weighted")


# In[ ]:


# Fit the grid search to the data
grid_search.fit(X_train_pca, y_train_res)


# In[ ]:


# printing the optimal accuracy score and hyperparameters
print('We can get recall of',grid_search.best_score_,'using',grid_search.best_params_)


# In[ ]:


# model with the best hyperparameters
dt_final = DecisionTreeClassifier(class_weight='balanced',
                             max_depth=14,
                             min_samples_leaf=100,
                             min_samples_split=100,
                             max_features=15,
                             random_state=10)


# In[ ]:


modelfit(dt_final,X_train_pca,y_train_res)


# In[ ]:


# make predictions
pred_probs_test = dt_final.predict(X_test_pca)
#Let's check the model metrices.
getModelMetrics(actual_churn=y_test,pred_churn=pred_probs_test)


# In[ ]:


# classification report
print(classification_report(y_test,pred_probs_test))


# Even after hyperparameter tuning for the Decision Tree. The recall rate is 67.54% which is not very significant to predict the churn.

# Let's see if we can achive a better Recall rate by deciding an optimal cut-off for the model to predict churn.

# In[ ]:


# predicting churn with default cut-off 0.5
cut_off_prob = 0.5
y_train_df = predictChurnWithProb(dt_final,X_train_pca,y_train_res,cut_off_prob)
y_train_df.head()


# In[ ]:


# finding cut-off with the right balance of the metrices
findOptimalCutoff(y_train_df)


# **From the curve above, let'choose 0.4 as the optimum point to make a high enough sensitivity.**

# In[ ]:


# predicting churn with cut-off 0.4
cut_off_prob=0.4
y_train_df = predictChurnWithProb(dt_final,X_train_pca,y_train_res,cut_off_prob)
y_train_df.head()


# - At 0.58 cut-off prob. there is a balance of sensitivity , specificity and accuracy.
# <br>Lets see how it performs on test data.

# In[ ]:


#Lets see how it performs on test data.
y_test_df= predictChurnWithProb(dt_final,X_test_pca,y_test,cut_off_prob)
y_test_df.head()


# - Decision tree after selecting optimal cut-off also is resulting in a model with
# <br>**Train Recall : 89.78%**  and  **Train Roc_auc_score : 82.40**
# <br>**Test Recall : 78.13%**  and  **Test Roc_auc_score : 76.56**
# 
# Random Forest still seems overfitted to the data.

# ### 3. Random Forest

# ##### Applying Random Forest Classifier on our principal components with Hyperparameter tuning

# In[ ]:


def plot_traintestAcc(score,param):
    scores = score
    # plotting accuracies with max_depth
    plt.figure()
    plt.plot(scores["param_"+param],
    scores["mean_train_score"],
    label="training accuracy")
    plt.plot(scores["param_"+param],
    scores["mean_test_score"],
    label="test accuracy")
    plt.xlabel(param)
    plt.ylabel("f1")
    plt.legend()
    plt.show()


# #### Tuning max_depth

# In[ ]:


parameters = {'max_depth': range(10, 30, 5)}
rf0 = RandomForestClassifier()
rfgs = GridSearchCV(rf0, parameters,
                    cv=5,
                   scoring="f1")
rfgs.fit(X_train_pca,y_train_res)


# In[ ]:


# scores = rfgs.cv_results_
# # plotting accuracies with max_depth
# plt.figure()
# plt.plot(scores["param_max_depth"],
#          scores["mean_train_score"],
#          label="training accuracy")
# plt.plot(scores["param_max_depth"],
#          scores["mean_test_score"],
#          label="test accuracy")
# plt.xlabel("max_depth")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()


rfgs = GridSearchCV(estimator=rf0, param_grid=param_grid, scoring='accuracy', cv=5, return_train_score=True)
rfgs.fit(X_train, y_train)

scores = rfgs.cv_results_

plt.figure()
plt.plot(scores["param_max_depth"],
         scores["mean_train_score"],
         label="training accuracy")
plt.plot(scores["param_max_depth"],
         scores["mean_test_score"],
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# Test f1-score almost becomes constant after max_depth=20

# #### Tuning n_estimators

# In[ ]:


parameters = {'n_estimators': range(50, 150, 25)}
rf1 = RandomForestClassifier(max_depth=20,random_state=10)
rfgs = GridSearchCV(rf1, parameters,
                    cv=3,
                   scoring="recall")


# In[ ]:


rfgs.fit(X_train_pca,y_train_res)


# In[ ]:


#plot_traintestAcc(rfgs.cv_results_,'n_estimators')

parameters = {'n_estimators': range(50, 150, 25)}
rf1 = RandomForestClassifier(max_depth=20, random_state=10)

rfgs = GridSearchCV(rf1, parameters, cv=3, scoring="recall", return_train_score=True)  # Update
rfgs.fit(X_train_pca, y_train_res)

plot_traintestAcc(rfgs.cv_results_, 'n_estimators') # Call the plotting function



# Selecting n_estimators = 80

# #### Tuning max_features

# In[ ]:


parameters = {'max_features': [4, 8, 14, 20, 24]}
rf3 = RandomForestClassifier(max_depth=20,n_estimators=80,random_state=10)
rfgs = GridSearchCV(rf3, parameters,
                    cv=5,
                   scoring="f1",return_train_score=True)


# In[ ]:


rfgs.fit(X_train_pca,y_train_res)
plot_traintestAcc(rfgs.cv_results_,'max_features')


# Selecting max_features = 5

# #### Tuning min_sample_leaf

# In[ ]:


parameters = {'min_samples_leaf': range(100, 400, 50)}
rf4 = RandomForestClassifier(max_depth=20,n_estimators=80,max_features=5,random_state=10)
rfgs = GridSearchCV(rf4, parameters,
                    cv=3,
                   scoring="f1", return_train_score=True)


# In[ ]:


rfgs.fit(X_train_pca,y_train_res)
plot_traintestAcc(rfgs.cv_results_,'min_samples_leaf')


# Selecting min_sample_leaf = 100

# #### Tuning min_sample_split

# In[ ]:


parameters = {'min_samples_split': range(50, 300, 50)}
rf5 = RandomForestClassifier(max_depth=20,n_estimators=80,max_features=5,min_samples_leaf=100,random_state=10)
rfgs = GridSearchCV(rf5, parameters,
                    cv=3,
                   scoring="f1", return_train_score=True)


# In[ ]:


rfgs.fit(X_train_pca,y_train_res)
plot_traintestAcc(rfgs.cv_results_,'min_samples_split')


# Selecting min_sample_split = 150

# #### Tunned Random Forest

# In[ ]:


rf_final = RandomForestClassifier(max_depth=20,
                                  n_estimators=80,
                                  max_features=5,
                                  min_samples_leaf=100,
                                  min_samples_split=50,
                                  random_state=10)


# In[ ]:


print("Model performance on Train data:")
modelfit(rf_final,X_train_pca,y_train_res)


# In[ ]:


# predict on test data
predictions = rf_final.predict(X_test_pca)


# In[ ]:


print("Model performance on Test data:")
getModelMetrics(y_test,predictions)


# After hyperparameter tuning for the random forest. The Recall rate(Test) is 73.39%.

# Let's see if we can achive a better Recall rate by deciding an optimal cut-off for the model to predict churn.

# In[ ]:


# predicting churn with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictChurnWithProb(rf_final,X_train_pca,y_train_res,cut_off_prob)
y_train_df.head()


# In[ ]:


# finding cut-off with the right balance of the metrices
findOptimalCutoff(y_train_df)


# **From the curve above, 0.45 is the optimal point with high enough sensitivity.**

# In[ ]:


cut_off_prob=0.45
predictChurnWithProb(rf_final,X_train_pca,y_train_res,cut_off_prob)


# **Making prediction on test**

# In[ ]:


y_test_df= predictChurnWithProb(rf_final,X_test_pca,y_test,cut_off_prob)
y_test_df.head()


# - Random Forest after selecting optimal cut-off also is resulting in a model with
# <br>**Train Recall : 88.70%**  and  **Train Roc_auc_score : 85.60**
# <br>**Test Recall : 77.57%**  and  **Test Roc_auc_score : 79.65**

# ### 4. Boosting models

# ###### 4.1 Gradiant boosting Classifier

# ###### Applying Gradiant boosting Classifier on our principal components with Hyperparameter tuning

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
# Fitting the default GradientBoostingClassifier
gbm0 = GradientBoostingClassifier(random_state=10)
modelfit(gbm0, X_train_pca, y_train_res)


# In[ ]:


# Hyperparameter tuning for n_estimators
param_test1 = {'n_estimators':range(20,150,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10),
param_grid = param_test1, scoring='f1',n_jobs=4,iid=False, cv=3)
gsearch1.fit(X_train_pca, y_train_res)


# In[ ]:


gsearch1.best_params_, gsearch1.best_score_


# In[ ]:


# Hyperparameter tuning for max_depth and min_sample_split
param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}
gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=140, max_features='sqrt', subsample=0.8, random_state=10),
param_grid = param_test2, scoring='f1',n_jobs=4,iid=False, cv=3)
gsearch2.fit(X_train_pca, y_train_res)


# In[ ]:


gsearch2.best_params_, gsearch2.best_score_


# In[ ]:


# Hyperparameter tuning for min_sample_leaf
param_test3 = {'min_samples_leaf':range(30,71,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=140,max_depth=15,min_samples_split=200, max_features='sqrt', subsample=0.8, random_state=10),
param_grid = param_test3, scoring='f1',n_jobs=4,iid=False, cv=3)
gsearch3.fit(X_train_pca, y_train_res)


# In[ ]:


gsearch3.best_params_, gsearch3.best_score_


# In[ ]:


# Hyperparameter tuning for max_features
param_test4 = {'max_features':range(7,20,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=140,max_depth=15, min_samples_split=200, min_samples_leaf=30, subsample=0.8, random_state=10),
param_grid = param_test4, scoring='f1',n_jobs=4,iid=False, cv=3)
gsearch4.fit(X_train_pca, y_train_res)


# In[ ]:


gsearch4.best_params_, gsearch4.best_score_


# Tunned GradientBoostingClassifier

# In[ ]:


# Tunned GradientBoostingClassifier
gbm_final = GradientBoostingClassifier(learning_rate=0.1, n_estimators=140,max_features=15,max_depth=15, min_samples_split=200, min_samples_leaf=40, subsample=0.8, random_state=10)
modelfit(gbm_final, X_train_pca, y_train_res)


# In[ ]:


# predictions on Test data
dtest_predictions = gbm_final.predict(X_test_pca)


# In[ ]:


# model Performance on test data
getModelMetrics(y_test,dtest_predictions)


# Let's see if we can achive a better Recall rate by deciding an optimal cut-off for the model to predict churn.

# In[ ]:


# predicting churn with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictChurnWithProb(gbm_final,X_train_pca,y_train_res,cut_off_prob)
y_train_df.head()


# In[ ]:


findOptimalCutoff(y_train_df)


# In[ ]:


cut_off_prob=0.1
predictChurnWithProb(gbm_final,X_train_pca,y_train_res,cut_off_prob)


# **Making prediction on test**

# In[ ]:


y_test_df= predictChurnWithProb(gbm_final,X_test_pca,y_test,cut_off_prob)
y_test_df.head()


# This model is litrally over-fitting the Training data with a lower performance on the Test data.

# ###### 4.2 XGBoost Classifier

# ##### Applying XGBoost Classifier on our principal components with Hyperparameter tuning

# In[ ]:


import xgboost as xgb
from xgboost.sklearn import XGBClassifier
# Fitting the XGBClassifier
xgb1 = XGBClassifier(learning_rate =0.1,
                    n_estimators=1000,
                    max_depth=5,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective= 'binary:logistic',
                    nthread=4,
                    scale_pos_weight=1,
                    seed=27)


# In[ ]:


# Model fit and performance on Train data
modelfit(xgb1, X_train_pca, y_train_res)


# In[ ]:


# Hyperparameter tunning for the XGBClassifer
param_test1 = {'max_depth':range(3,10,2),'min_child_weight':range(1,6,2)}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
 param_grid = param_test1, scoring='f1',n_jobs=4,iid=False, cv=3)
gsearch1.fit(X_train_pca, y_train_res)


# In[ ]:


gsearch1.best_params_, gsearch1.best_score_


# In[ ]:


# Some more hyperparameter tunning for the XGBClassifer
param_test2 = param_test3 = {'gamma':[i/10.0 for i in range(0,5)]}
gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=9,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
 param_grid = param_test2, scoring='f1',n_jobs=4,iid=False, cv=3)
gsearch2.fit(X_train_pca, y_train_res)


# In[ ]:


gsearch2.best_params_, gsearch2.best_score_


# In[ ]:


# Final XGBClassifier
xgb2 = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=9,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27)


# In[ ]:


# Fit Train data
modelfit(xgb2, X_train_pca, y_train_res)


# In[ ]:


# Prediction on Test data
dtest_predictions = xgb2.predict(X_test_pca)


# In[ ]:


# Model evaluation on Test data
getModelMetrics(y_test,dtest_predictions)


# Let's see if we can achive a better Recall rate by deciding an optimal cut-off for the model to predict churn.

# In[ ]:


# predicting churn with default cut-off 0.5
cut_off_prob=0.5
y_train_df = predictChurnWithProb(xgb2,X_train_pca,y_train_res,cut_off_prob)
y_train_df.head()


# In[ ]:


# Finding optimal cut-off probability
findOptimalCutoff(y_train_df)


# In[ ]:


# Selecting 0.2 as cut-off in an attempt to improve recall rate
cut_off_prob=0.2
predictChurnWithProb(xgb2,X_train_pca,y_train_res,cut_off_prob)


# **Making prediction on test**

# In[ ]:


y_test_df= predictChurnWithProb(xgb2,X_test_pca,y_test,cut_off_prob)
y_test_df.head()


# ### 5. SVM

# ##### Using linear kernal

# In[ ]:


# instantiate an object of class SVC()
# note that we are using cost C=1
svm0 = SVC(C = 1)


# In[ ]:


# fit
svm0.fit(X_train_pca, y_train_res)

# predict on train
y_pred = svm0.predict(X_train_pca)
getModelMetrics(y_train_res,y_pred)


# In[ ]:


# Predict on test
y_pred = svm0.predict(X_test_pca)
getModelMetrics(y_test,y_pred)


# ###### Hyperparameter tuning for linear kernal

# Let's see if we can tune the hyperparameters of SVM and get a better Sensitivity score.

# In[ ]:


# specify range of parameters (C) as a list
params = {"C": [0.1, 1, 10, 100, 1000]}

svm1 = SVC()

# set up grid search scheme
# note that we are still using the 5 fold CV scheme
model_cv = GridSearchCV(estimator = svm1, param_grid = params,
                        scoring= 'f1',
                        cv = 5,
                        verbose = 1,
                        n_jobs=4,
                       return_train_score=True)
model_cv.fit(X_train_pca, y_train_res)


# In[ ]:


plot_traintestAcc(model_cv.cv_results_,'C')


# In[ ]:


model_cv.best_params_


# In[ ]:


svm_final = SVC(C = 1000)
# fit
svm_final.fit(X_train_pca, y_train_res)


# In[ ]:


# predict
y_pred = svm_final.predict(X_test_pca)


# In[ ]:


getModelMetrics(y_test,y_pred)


# ##### Using non-linear kernal

# In[ ]:


svm_k = SVC(C = 1000, kernel='rbf')
svm_k.fit(X_train_pca, y_train_res)


# In[ ]:


y_pred = svm_k.predict(X_test_pca)


# In[ ]:


getModelMetrics(y_test,y_pred)


# **Recall Score: 78%**

# Now that we have a variety of models used to predict the churn for the telecom. Let's caompare and decide a model of choice for this problem of churn prediction.

# ---------------

# ## Final Choice of Model
# 
# Recall is the most important business metric for the telecom churn problem. The company would like to identify most customers at risk of churning, even if there are many customers that are misclassified as churn. The cost to the company of churning is much higher than having a few false positives.

# | Model/Metrics                         | Train   | Test   |
# |---------------------------------------|---------|--------|
# | Logistic Regression ( cut-off = 0.45) |         |        |
# | Roc_auc_score                         | 82.11%  | 81.21% |
# | Sensitivity/Recall                    | 86.48%  | 84.40% |
# | Specificity                           | 77.75%  | 78.02% |
# | precision                             | 79.54%  | 25.04% |
# | DecisionTree ( cut-off = 0.4)         |         |        |
# | Roc_auc_score                         | 82.41%  | 76.57% |
# | Sensitivity/Recall                    | 89.79%  | 78.13% |
# | Specificity                           | 75.03%  | 75%    |
# | precision                             | 78.24%  | 21.38% |
# | Random Forest (cut-off = 0.45)        |         |        |
# | Roc_auc_score                         | 85.60%  | 96.53% |
# | Sensitivity/Recall                    | 88.70%  | 77.57% |
# | Specificity                           | 82.50%  | 81.73% |
# | precision                             | 83.52%  | 26.97% |
# | GBC                                   |         |        |
# | Roc_auc_score                         | 96.11%  | 80.84% |
# | Sensitivity/Recall                    | 100.00% | 79.87% |
# | Specificity                           | 92.21%  | 81.81% |
# | precision                             | 92.78%  | 28.52% |
# | XGB (cut-off = 0.2)                   |         |        |
# | Roc_auc_score                         | 97.24%  | 80.76% |
# | Sensitivity/Recall                    | 99.99%  | 76.13% |
# | Specificity                           | 94.49%  | 85.38% |
# | precision                             | 94.78%  | 32.13% |
# | SVM (linear   C = 1000 )              |         |        |
# | Roc_auc_score                         | 81.33%  | 82.62% |
# | Sensitivity/Recall                    | 79.91%  | 78.40% |
# | Specificity                           | 82.75%  | 86.85% |
# | precision                             | 82.25%  | 35.14% |

# Overall, the **Logistic Regression** model with probability cut-off = 0.45, performs best. It achieved the **best recall accuracy of 84.4%** for test data. Also the overall accuracy and specificity is consistent for Test and train data, thus avoiding overfitting. The precision is compromised in this effort but the business objective to predict Churn customers is most accuratety captured by it.
# 
# Next, Linear SVM which achives a recall rate of 78.40%, a slightly better precision of 35.14% and a balanced overall accuracy on train and test.
# 
# From the Tree Family, the Decision Tree overfitted the data slightly while obtaining 78.13% recall accuracy on test data.
# The Random Forest avoided overfitting but obtained only 77.57% recall accuracy on test data.
# 
# Among the Bossting Methods, Gradient Boosting Classifer (GBC) achived 81.81% recall rate and XGBoost Classifier achived 76.13% but both tend to overfit the training data.
# 
# 

# ## Identifying relevant churn features.
# 
# We will use an instance of Random Forest classifier to identify the features most relevant to churn.

# ### Random Forest for churn driver features

# In[ ]:


# Create the parameter grid based on the results of random search
param_grid = {
    'max_depth': [8,10,12],
    'min_samples_leaf': range(100, 400, 200),
    'min_samples_split': range(200, 500, 200),
    'n_estimators': [100,200, 300],
    'max_features': [12, 15, 20]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = 4,verbose = 1)


# In[ ]:


# Fit the grid search to the data
grid_search.fit(X_train_res, y_train_res)


# In[ ]:


# printing the optimal accuracy score and hyperparameters
print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)


# In[ ]:


rf = RandomForestClassifier(max_depth=12,
                            max_features=20,
                            min_samples_leaf=100,
                            min_samples_split=200,
                            n_estimators=300,
                            random_state=10)


# In[ ]:


rf.fit(X_train_res, y_train_res)


# In[ ]:


plt.figure(figsize=(15,40))
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(len(X.columns)).sort_values().plot(kind='barh', align='center')


# Some of the top main predictiors of churn are the monthly KPI features for the action phase (3rd month August).

# In[ ]:


the graph above suggest that the top 25 features ranked in order of importance as produced by our RandomForest implementation are the features that belong to month 8 i.e., the action month. Hence, it is clear that what happens in the action phase has a direct impact on the customer churn of high value customers. Specifically, these features are as follows:




1.	**total_ic_mou_8**		-- *Total incoming minutes of usage in month 8*
2.	**loc_ic_mou_8**		-- *local incoming minutes of usage in month 8*
3.	**total_month_rech_8**	-- *Total month recharge amount in month 8*
4.	**total_roam_mou_8**	-- *Total incoming+outgoing roaming minutes of usage in month 8*
5.	**loc_ic_t2m_mou_8**	-- *local incoming calls to another operator minutes of usage in month 8*
6.	**roam_og_mou_8**		-- *outgoing roaming calls minutes of usage in month 8*
7.	**Total_loc_mou_8**		-- *Total local minutes of usage in month 8*
8.	**roam_ic_mou_8**		-- *incoming roaming calls minutes of usage in month 8*
9.	**total_rech_amt_8**	-- *total recharge amount in month 8*
10.	**loc_ic_t2t_mou_8**	-- *local incoming calls from same operator minutes of usage in month 8*
11.	**max_rech_amt_8**		-- *maximum recharge amount in month 8*
12.	**last_day_rch_amt_8**	-- *last (most recent) recharge amount in month 8*
13.	**arpu_8**				-- *average revenue per user in month 8*
14.	**loc_og_mou_8**		-- *local outgoing calls minutes of usage in month 8*
15.	**loc_og_t2n_mou_8**	-- *local outgoing calls minutes of usage to other operator mobile in month 8*
16.	**av_rech_amt_data_8**	-- *average recharge amount for mobile data in month 8*
17.	**total_rech_data_8**	-- *total data recharge (MB) in month 8*
18.	**total_og_t2t_mou_8**	-- *total outgoing calls from same operator minutes of usage in month 8*
19.	**total_rech_num_8**	-- *total number of recharges done in the month 8*
20.	**total_rech_amt_data_8**	-- *total recharge amount for data in month 8*
21.	**max_rech_data_8**		-- *maximum data recharge (MB) in month 8*
22.	**avg_rech_amt_8**		-- *average recharge amount in month 8*
23.	**fb_user_8**			-- *services of Facebook and similar social networking sites for month 8*
24.	**vol_data_mb_8**		-- *volume of data (MB) consumed for month 8*
25.	**count_rech_2g_8**		-- *Number of 2g data recharge in month 8*
26.	**loc_og_to_ic_mou_8**	-- *local outgoing to incoming mou ratio for month of 8*
27.	**spl_og_mou_7**		-- *Special outgoing call for the month of 7*


# Local calls Mou's be it incoming or outgoing have a very important role for churn predictions. Reduction in these KPI's forms a clear indicator of churn.
# 
# Overall, drop in any of these indicator KPI is a signal that the customer is not actively engaging in the services offered by the Network operator and thus may choose to churn in the near future.
# 
# Next, we will look at some of the stratergic steps which can be taken to retain these predicted churners.

# ## Strategies to manage customer churn
# 
# It is a fact that it costs 5-10 times more to acquire a new customer than to retain an existing one, customer retention has now become even more important than customer acquisition.
# 
# For many incumbent operators, retaining high profitable customers is the number one business goal.

# #### Monitoring Drop in usage

# Customer churn seems to be well predicted by drop in usage.
# 
# Aside from using the Machine Learning model for predicting churn, the telecom company should pay close attention to drop in MoU, ARPU and data usage (2g and 3g) month over month. If feasible, the company should track these numbers week over week. Since billing cycles are typically monthly, a drop in usage numbers will give the company time to react when tracked at weekly level.
# 
# Contact these customers proactively to find out what's affecting their experience. Perhaps, offer them coupons or other incentives to continue to use the services, while the company fixes the issues reported.
# 
# Marketing team must come up with campaigns which targets these high-value to-be churner.

# ###### Improving Outgoing services

# In[ ]:


# Outgoing Mou
plot_byChurnMou(og_col,'Outgoing')


# -  Initially, churner's outgoing usage was more than that of non-churners. Gradually they dropped there outgoing usage. May be these customers din't like the outgoing services offered to them or may be the call tariffs seemed expensive to them or may be the overall call quality, network coverage was not liked my them. This could be further investigated by the network service provider.

# Stratergy suggestions,
# - The Network operators must futher investigate their outgoing tariffs, plans and campaigns.
# - Might be that the outgoing tariffs offered to it's customer are less competitive to the outgoing tariffs of their competitor.
# - New campaigns which targets the customers with high outgoing usage be rolled out.Like,
#     - Discounted outgoing rates during particular hours of the day for these customers.
#     - For every X mou, grant customer with some % of X free mou.
#     - Investigate and if need be revise the outgoing tarrifs to make it competitive.
#     - Free monthly outgoing mou's depending on the users past roaming mou usage.

# ###### Improving Roaming services

# In[ ]:


plot_byChurn(hv_users,'Total_roam_mou')


# Stratergy suggestions,
# - Churners show higher roaming usage than non-churners.
# - The Network operators must futher investigate their roaming tariffs, and quality of service.
# - Might be that the roaming tariffs offered are less competitive than their competitor.
# - It might be that the customer is not getting good quality of service while roaming. In this case, quality of service guarantees with roaming partners and network quality need to be investigated.
# - New campaigns which targets the roaming customers can be rolled out. Like,
#     - Discounted roaming rates during particular hours of the day.
#     - Free monthly roaming mou's depending on the users past roaming mou usage.

# In[ ]:


import datetime, pytz;
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

