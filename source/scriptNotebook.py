# %% [markdown]
# <h2 style="font-weight:bold; font-family:sans-serif"><b>Goal of creating this Notebook</h2>
# 
# 1. Perform Clustering / Segmentation on the dataset and identify popular customer groups along with their definitions/rules
# 2. Perform Location-wise analysis to identify regional trends in India
# 3. Perform transaction-related analysis to identify interesting trends that can be used by a bank to improve / optimi their user experiences
# 4. Customer Recency, Frequency, Monetary analysis
# 5. Network analysis or Graph analysis of customer data.

# %% [markdown]
# **Table of contents of this notebook:**
# 
# **1.** Importing Necessary Libraries<br>
# **2.** Data Collection<br>
# **3.** Data Cleaning<br>
# **4.** Exploratory Data Analysis

# %% [markdown]
# <h2  style="text-align: center; padding: 20px; font-weight:bold">1. Importing Libraries</h2>

# %%
import re
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import seaborn as sns

# %% [markdown]
# <h2  style="text-align: center; padding: 20px; font-weight:bold">2. Data Collection</h2><a id = "2"></a>

# %% [markdown]
# We import our relative path Dataset and in addition to that, we rename the TransactionAmount (INR) column in TransactionAmount, this in order to avoid that in the future the special characters of the original name give us problems.

# %%
def import_data():
    """This function import the data and rename a column.

    This is a more detailed description of the function,
    which can span multiple lines.

    Args:
        

    Returns:
        DataFrame: A DataFrame with the DataSet.

    Raises:
        ValueError: If something goes wrong.
    """
    dfGet = pd.read_csv(r"D:\Gabriel Suarez\Desktop\University\Semestre 7\Inteligencia Artificial\Final Project - AI\Bank-Customer-Segmentation\data\bank_transactions.csv")
    initialRows = len(dfGet)
    dfGet = dfGet.rename(columns={'TransactionAmount (INR)':'TransactionAmount'})
    return dfGet

df = import_data()
df.head()

"""
# %% [markdown]
# We obtain the initial information of the Dataset, the number of records, number of variables, non-null objects and data type.

# %%
display(df.info())

# %% [markdown]
# <h2  style="text-align: center; padding: 20px; font-weight:bold">3. Data Cleaning</h2><a id = "3"></a>

# %% [markdown]
# The amount of null data and unique is calculated

# %%
def check(df):
    l=[]
    columns=df.columns
    for col in columns:
        dtypes=df[col].dtypes
        nunique=df[col].nunique()
        sum_null=df[col].isnull().sum()
        l.append([col,dtypes,nunique,sum_null])
    df_check=pd.DataFrame(l)
    df_check.columns=['Column','Types','Unique','Nulls']
    return df_check 
check(df)


# %% [markdown]
# Eliminamos los valores nulos

# %%
shapeInitial = df.shape[0]
df.dropna(inplace=True)
shapeFinal = shapeInitial-df.shape[0]
print("Amount to remove " + str(shapeFinal))

# %% [markdown]
# The amount of null values to eliminate is equal to 6953 data, we eliminate these values because they do not represent more than 0.7% of the total data. <br> We check if there are repeated elements in our DataSet

# %%
df.duplicated().sum()

# %% [markdown]
# The CustomerDOB column is analyzed because it may contain atypical data.
# <br>
# We analyze the number of records for each client's date of birth.

# %%
# Getting distinct values from CustomerDOB variable
df['CustomerDOB'].value_counts()

# %% [markdown]
# Dates 1/1/1800 are deleted because it is not possible to define whether they are children, adults or persons without date of birth. This is an important variable for the business, for this reason we cannot make assumptions that bias the project, for this reason, it is better to eliminate these outliers or erroneously measured data.

# %%
# Removing CustomerDOB == '1/1/1800'
df = df.loc[~(df['CustomerDOB'] == '1/1/1800')]
# Cheking distinct values from dataframe
df['CustomerDOB'].value_counts()

# %% [markdown]
# We print the minimum and maximum date of the CustomerDOB column in order to see in which range the values in this column oscillate.

# %%
# Range of CustomerDOB object type as string
print("min: " + df['CustomerDOB'].min() + " max: " + df['CustomerDOB'].max())

# %% [markdown]
# It can be seen that the person with the oldest birth date has a date of January 1, 1900 and the youngest person has a date of September 9, 1997.

# %% [markdown]
# Convert type of columns TransactionDate, CustomerDOB from string to datetime, this convertation will be in the format of dayfirst, so the date will be DD/MM/YY

# %%
# Using pandas convert to datetime tool for CustomerDOB variable with specific format
df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'], dayfirst=True)

# %% [markdown]
# Now we will check if the conversion was as expected and in the required format.

# %%
# Checking converting problem of to_datetime pandas function
print(df['CustomerDOB'].min(), df['CustomerDOB'].max())

# %% [markdown]
# We can see that the most "recent" date is December 31, 2072, but it is illogical because this is a future date, so we subtract 100 from all values greater than 1999 to get the real value. (This is a problem Pandas has when converting a date).

# %%
# Fixing the problem base on analysis above
df.loc[df['CustomerDOB'].dt.year > 1999, 'CustomerDOB'] -= pd.DateOffset(years=100)
print(df['CustomerDOB'].min(), df['CustomerDOB'].max())

# %% [markdown]
# In the same way that we converted the CustomerDOB column, we convert the TransactionDate column with the same expected format as the first day.

# %%
# Using pandas convert to datetime tool for TransactionDate variable
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], dayfirst=True)
# Checking range of TransactionDate variable
print(df['TransactionDate'].min(), df['TransactionDate'].max())

# %% [markdown]
# All the trnasactions took place in a roughly two month period from August to October, this could account for the low transaction frequency

# %% [markdown]
# Once all the unnecessary data for the study has been eliminated, we can see the following summary, which shows us how much data we lost and what is the percentage of loss obtained

# %%
print(" Number of initial rows: ", initialRows, "\n",
"Number of rows after: ", df.shape[0], "\n",
"Number of rows deleted: ", initialRows - df.shape[0], "\n",
"Percentage of rows deleted: ", (initialRows - df.shape[0]) / initialRows * 100, "%")

# %% [markdown]
# We can see that we lost <b>6.03%</b> of the data, although what is expected <b>by theory is to lose less than 5%</b> in this case we must ignore this metric because there are null values and measurement failure errors that <b>force us to eliminate them</b>, because we cannot speculate about them.

# %% [markdown]
# <h2  style="text-align: center;padding: 20px;font-weight:bold">4. Exploratory Data Analysis</h2><a id = "4"></a>

# %% [markdown]
# Determine minority group of people aged <b> over 100 years</b>

# %%
# We filter our dataframe specifically on the DOB column to make a decision regarding date ambiguity.
df_filtered = df["CustomerDOB"].apply(lambda x: x if x.year < 1917 else 0)
# Amortizing and removing values ​​greater than 1917 represented as 0
counts = df_filtered.value_counts().drop(0)
print(counts)
# Amount of customer in this age range
print(len(counts))
# Plot the amortized
counts.plot()
plt.show()
del df_filtered

# %% [markdown]
# <h3><b>Calculate customer age :</b></h3>
# <b>CustomerDOB:</b> is the birth date of the customer 
# <br>
# <b>TransactionDate:</b> is the date of transaction that customer is done
# <br>
# The age calculation is done by <b>subtracting</b> the TransactionDate from the CustomerDOB.
# 

# %%
# Getting the customer age at transaction moment and adding a new column in our dataframe
df['CustomerAge'] = (df['TransactionDate'] - df['CustomerDOB'])/np.timedelta64(1, 'Y')
df['CustomerAge'] = df['CustomerAge'].astype(int)
# Checking range of CustomerAge variable
print("min: " + str(df['CustomerAge'].min()) + " max: " + str(df['CustomerAge'].max()))

# %% [markdown]
# Once this is obtained, we have that the minimum age is equal to 16 years and the maximum age is equal to 116 years, it should be noted that the ages over 100 are a minimum percentage.

# %% [markdown]
# We obtain the percentage between customers who are women and men.

# %%
# Getting distinct values from CustGender variable
df.CustGender.value_counts()

# %% [markdown]
# <h5> Visualize the distribution of the numeric data and detect posible outliers. Boxplots show the median, quartiles, and extreme values ​​of the data, and points that are above or below the extreme values ​​are considered outliers.</h5>

# %%
num_col = df.select_dtypes(include=np.number)
cat_col = df.select_dtypes(exclude=np.number)

for i in num_col.columns:
    sns.boxplot(x=num_col[i])
    plt.title("Boxplot " + i)
    plt.show()

# %% [markdown]
# <h3 style="font-family:Sans-Serif; font-weight:bold">RFM Metrics:</h3>
# <ul>
# <li><b>Recency: </b>The freshness of customer activity e.g. time since last activity</li>
# <li><b>Frequency: </b>The requency of customer transactions e.g. the totla number of recorded transactions</li>
# <li><b>Monetary: </b>The willingness to spend e.g. the thoal transaction value</li>
# </ul>

# %% [markdown]
# <p>Those two articles will help you to understand this topic:</p>
# <a href="https://connectif.ai/en/what-are-rfm-scores-and-how-to-calculate-them/">What Are RFM Scores and How To Calculate Them</a>
# <br>
# <a href="https://www.datacamp.com/tutorial/introduction-customer-segmentation-python">Introduction to Customer Segmentation in Python</a>

# %% [markdown]
# We prepare some columns to make the RFM table

# %%
df['TransactionDate1']=df['TransactionDate'] # ==> to calculate the minimum (first transaction)
df['TransactionDate2']=df['TransactionDate'] # ==> to calculate the maximum (last transaction)

# %% [markdown]
# Este código es utilizado para crear una tabla RFM (Recency, Frequency, Monetary)
# Se agrupan los datos por CustomerID utilizando el método groupby y luego se utiliza la función agg para calcular distintas métricas para cada cliente.
# 
# Las métricas que se calculan son las siguientes:
# <ul>
# <li><b>TransactionID:</b> cantidad de transacciones realizadas por el cliente.</li>
# <li><b>CustGender:</b> género del cliente (tomado de la primera transacción registrada para el cliente).</li>
# <li><b>CustLocation:</b> ubicación del cliente (también tomada de la primera transacción registrada para el cliente).</li>
# <li><b>CustAccountBalance:</b> saldo promedio de la cuenta del cliente.</li>
# <li><b>TransactionTime:</b> hora promedio de las transacciones realizadas por el cliente.</li>
# <li><b>TransactionAmount:</b> monto promedio de las transacciones realizadas por el cliente.</li>
# <li><b>CustomerAge:</b> edad mediana del cliente.</li>
# <li><b>TransactionDate2:</b> fecha más reciente en la que el cliente realizó una transacción.</li>
# <li><b>TransactionDate1:</b> fecha más antigua en la que el cliente realizó una transacción.</li>
# <li><b>TransactionDate:</b> fecha mediana en la que el cliente realizó una transacción.</li>
# </ul>

# %%
#Creating MRF Table Strategy
RFM_df = df.groupby("CustomerID").agg({
                                        "TransactionID" : "count",
                                        "CustGender" : "first",
                                        "CustLocation":"first",
                                        "CustAccountBalance"  : "mean",
                                        "TransactionTime": "mean",
                                        "TransactionAmount" : "mean",
                                        "CustomerAge" : "median",
                                        "TransactionDate2":"max",
                                        "TransactionDate1":"min",
                                        "TransactionDate":"median"
                        })

RFM_df = RFM_df.reset_index()
RFM_df.head()

# %% [markdown]
# Now we calculate the number of records we have left after grouping by CustomerID

# %%
# After Grouping by CustomerID
RFM_df.shape

# %% [markdown]
# The ID of the customer is irrelevant to solve our problem, so we decided to remove it

# %%
# The id of the customer is irrelevant
RFM_df.drop(columns=["CustomerID"],inplace=True)

# %% [markdown]
# <h4><b>Frequency</b></h4>
# <p>As we count the TransactionID column, we can replace the name of this column by Frequency, because this is the number of times a customer has made a transaction.</p>

# %%
# Renaming specific column adapting to problem goal and replacing with inplace property of function
RFM_df.rename(columns={"TransactionID":"Frequency"},inplace=True)

# %% [markdown]
# <h4><b>Recency</b></h4>
# <p>The recency is the number of days since the last purchase or order so we will create a new column of TransactionDate to subtract the last transaction from the first transaction</p>

# %%
# Getting Recency that is by definition: number of days since the last purchase or order
RFM_df['Recency']=RFM_df['TransactionDate2']-RFM_df['TransactionDate1']
# Conversion from timedelta64[ns] to string representtion in days of weeks of Recency variable
RFM_df['Recency']=RFM_df['Recency'].astype(str)

# %% [markdown]
# We apply a lambda function to adjust the format of our output in the Recency variable

# %%
# Using re library for apply an regular expresion in each value of Recency column for extract the number of days in this string representation. 
RFM_df['Recency']=RFM_df['Recency'].apply(lambda x :re.search('\d+',x).group())
# Conversion from string '18' to int representtion for folloeing operations
RFM_df['Recency']=RFM_df['Recency'].astype(int)

# %% [markdown]
# <p> <b>Appreciation:</b> Days mean that a customer has done transaction recently one time by logic so I will convert 0 to 1 </p>

# %%
# Transformation of 0 days base on business meaning
RFM_df['Recency'] = RFM_df['Recency'].apply(lambda x: 1 if x == 0 else x)

# %% [markdown]
# The TransactionDate1 and TransactionDate2 columns have already fulfilled their objectives, which is to calculate the Recency, we can eliminate these columns.

# %%
# Columns that were only needed for the calculation we eliminated
RFM_df.drop(columns=["TransactionDate1","TransactionDate2"],inplace=True)

# %% [markdown]
# Now, let's see if our DataSet once cleaned contains atypical data

# %%
# To calculate the otliers for each feature
lower_list=[]
upper_list=[]
num_list=[]
perc_list=[]
cols=['Frequency', 'CustAccountBalance','TransactionAmount', 'CustomerAge', 'Recency']
for i in cols:
    Q1 = RFM_df[i].quantile(0.25)
    Q3 = RFM_df[i].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # Calculate number of outliers
    num=RFM_df[(RFM_df[i] < lower) | (RFM_df[i] > upper)].shape[0]
    # Calculate percentage of outliers
    perc = (num / RFM_df.shape[0]) * 100
    lower_list.append(lower)
    upper_list.append(upper)
    num_list.append(num)
    perc_list.append(round(perc,2))

    
dic={'lower': lower_list, 'upper': upper_list, 'outliers': num_list, 'Perc%':perc_list }
outliers_df=pd.DataFrame(dic,index=['Frequency', 'CustAccountBalance','TransactionAmount', 'CustomerAge', 'Recency'])
outliers_df

# %% [markdown]
# <h3 style="font-family:Sans-Serif; font-weight:bold">Observations:</h3>
# <p>We will not remove outliers for the following two reasons: First, in boxplots those values ​​can be outliers because they represent points that are above or below extreme values. However, these were not measurement errors and are both true and significant, given that while customers 100+ do not represent a key demographic for most banks. Secoind it is important that banks are aware of the specific needs and challenges that these clients may face, and that they adapt their strategies accordingly.</p>

# %% [markdown]
# Now, let's go to see our RFM Table

# %%
RFM_df.head()

# %% [markdown]
# We describe each of the columns with different factors

# %%
RFM_df.describe()

# %% [markdown]
# It creates a correlation matrix between the different features in the RFM table (RFM_df), and then plots this matrix as a heat map using it. The correlation matrix is a square matrix that shows how the different features are related to each other.

# %%
# correlation between features
plt.figure(figsize=(7,5))
correlation=RFM_df.corr(numeric_only=True)
sns.heatmap(correlation,vmin=None,
    vmax=0.8,
    cmap='rocket_r',
    annot=True,
    fmt='.1f',
    linecolor='white',
    cbar=True);

# %% [markdown]
# We obtain the frequency bar chart, this chart shows the distribution of the variable Frequency, it is worth noting that the frequency is the number of times a customer has made transactions in the period from August to October.

# %%
plt.style.use("fivethirtyeight")
chart=sns.countplot(x='Frequency',data=RFM_df,palette='rocket', order = RFM_df['Frequency'].value_counts().index)
plt.title("Frequency",
          fontsize='20',
          backgroundcolor='AliceBlue',
          color='magenta');

# %% [markdown]
# We obtain the age distribution of the clients and also the percentage of women and men in the records we have.

# %%
plt.style.use("fivethirtyeight")
fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(15,5))
palette_color = sns.color_palette('rocket')
ax[0].hist(x=RFM_df['CustomerAge'],color='purple')
ax[0].set_title("Distribution of Customer Age")
ax[1].pie(RFM_df['CustGender'].value_counts(),autopct='%1.f%%',colors=palette_color,labels=['Male','Female'])
ax[1].set_title("Customer Gender")
plt.tight_layout();

# %% [markdown]
# In this graph we obtain the number of times a transaction was made in different areas of the country, only the top 20 locations with the most transactions made will be shown.

# %%
plt.style.use("fivethirtyeight")
plt.figure(figsize=(15,7))
chart=sns.countplot(y='CustLocation',data=RFM_df,palette='rocket', order = RFM_df['CustLocation'].value_counts()[:20].index)
plt.title("Most 20 Location of Customer ",
          fontsize='15',
          backgroundcolor='AliceBlue',
          color='magenta');

# %% [markdown]
# We generate the scatter plot of the data referring to the variable Frequency.

# %%
plt.style.use("fivethirtyeight")
sns.pairplot(RFM_df,hue='Frequency')

# %% [markdown]
# This code generates a scatter plot. The data used comes from the RFM_df dataframe and is represented on the X axis (horizontal) the transaction amounts (TransactionAmount) and on the Y axis (vertical) the customer's account balance (CustAccountBalance). We add a third dimension to the graph which is Frequency and a fourth one with Recency.

# %%
plt.style.use("fivethirtyeight")
sns.scatterplot(x='TransactionAmount',y='CustAccountBalance',data=RFM_df,palette='rocket',hue='Frequency',size='Recency' )
plt.legend(loc = "upper right")
plt.title("TransactionAmount (INR) and CustAccountBalance",
          fontsize='15',
          backgroundcolor='AliceBlue',
          color='magenta');

# %% [markdown]
# We calculate the farthest distance between two completed transactions

# %%
# difference between maximum and minimum date
RFM_df['TransactionDate'].max()-RFM_df['TransactionDate'].min()

# %% [markdown]
# We group the transactions according to the month in which they were made and obtain the average for each table.

# %%
RFM_df=RFM_df.sort_values(by='TransactionDate')
groupbby_month = RFM_df.groupby([pd.Grouper(key='TransactionDate', freq='M')])[['Frequency', 'TransactionAmount', 'CustAccountBalance', 'TransactionTime', 'CustomerAge', 'Recency']].mean()
print(groupbby_month.shape)
groupbby_month

# %% [markdown]
# We made line graphs of the information we obtained previously.

# %%
plt.figure(figsize=(13.4,7))
plt.title("Average of account balance per month")
plt.plot(groupbby_month.index,groupbby_month['CustAccountBalance'],color='purple',marker='o',label='Customer Account Balance', linestyle='dashed', linewidth=2, markersize=10)
plt.legend();

plt.figure(figsize=(13.8,7))
plt.title("Average of transaction amount per month")
plt.plot(groupbby_month.index,groupbby_month['TransactionAmount'],color='green',marker='o',label='Transaction Amount', linestyle='dashed', linewidth=2, markersize=10)


"""