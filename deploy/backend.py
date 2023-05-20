import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(data):
    df = pd.read_csv(data)
    return df

def dateConvertion(dataframe, column):
    """
    Converts dataframe column to datetime format using pandas tool with specific format 'dayfirst'

    Args:
        dataframe (DataFrame): Source dataset.
        column (string): Variable or column in dataframe

    Returns:
        DataFrame: A pandas DataFrame already modified.

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """
    return pd.to_datetime(dataframe[column], dayfirst=True)
df['CustomerDOB'] = dateConvertion(df,'CustomerDOB')

def refactorDates(dataframe):
    """
    Refactors date dob column substrating 100 from values greater than 1999
    Note: Fixing the problem base on analysis above

    Args:
        dataframe (DataFrame): Source dataset.

    Returns:
        DataFrame: A pandas DataFrame already modified.

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """
    dataframe.loc[df['CustomerDOB'].dt.year > 1999, 'CustomerDOB'] -= pd.DateOffset(years=100)
    return dataframe
df = refactorDates(df)
minAndMax(df,'CustomerDOB')

# Getting the customer age at transaction moment and adding a new column in our dataframe
def getCustomerAge(dataframe):
    """
    Gets the customer age at transaction moment

    Args:
        dataframe (DataFrame): Source dataset.

    Returns:
        DataFrame: A pandas DataFrame already modified.

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """
    df['CustomerAge'] = (df['TransactionDate'] - df['CustomerDOB'])/np.timedelta64(1, 'Y')
    df['CustomerAge'] = df['CustomerAge'].astype(int)
    # Checking range of CustomerAge variable
    print("min: " + str(df['CustomerAge'].min()) + " max: " + str(df['CustomerAge'].max()))

getCustomerAge(df)

df['TransactionDate1']=df['TransactionDate'] # ==> to calculate the minimum (first transaction)
df['TransactionDate2']=df['TransactionDate'] # ==> to calculate the maximum (last transaction)

#Creating MRF Table Strategy

def MRFTable(df):   
    """
    Creates a MRF Table from a dataframe

    Args:
        dataframe (DataFrame): Source dataset.

    Returns:
        DataFrame: A pandas DataFrame already modified.

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """
    
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

    RFM_df.reset_index()
    return RFM_df
    

RFM_df = MRFTable(df)
RFM_df.head()

# Renaming specific column adapting to problem goal and replacing with inplace property of function
RFM_df.rename(columns={"TransactionID":"Frequency"},inplace=True)

# Getting Recency that is by definition: number of days since the last purchase or order
RFM_df['Recency']=RFM_df['TransactionDate2']-RFM_df['TransactionDate1']
# Conversion from timedelta64[ns] to string representtion in days of weeks of Recency variable
RFM_df['Recency']=RFM_df['Recency'].astype(str)

def formatOutputInRecency(RFM_df):
    
    """
    Formats the output of Recency column

    Args:
        dataframe (DataFrame): Source dataset.

    Returns:
        DataFrame: A pandas DataFrame already modified.

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """
    # Using re library for apply an regular expresion in each value of Recency column for extract the number of days in this string representation. 
    RFM_df['Recency']=RFM_df['Recency'].apply(lambda x :re.search('\d+',x).group())
    # Conversion from string '18' to int representtion for folloeing operations
    RFM_df['Recency']=RFM_df['Recency'].astype(int)

formatOutputInRecency(RFM_df)

# Transformation of 0 days base on business meaning
RFM_df['Recency'] = RFM_df['Recency'].apply(lambda x: 1 if x == 0 else x)

# Columns that were only needed for the calculation we eliminated
RFM_df.drop(columns=["TransactionDate1","TransactionDate2"],inplace=True)

RFM_df['TransactionMonth'] = RFM_df["TransactionDate"].dt.month
RFM_df['TransactionMonthName'] = RFM_df["TransactionDate"].dt.month_name()
RFM_df['TransactionDay'] = RFM_df["TransactionDate"].dt.day
RFM_df['TransactionDayName'] = RFM_df["TransactionDate"].dt.day_name()
RFM_df.head

def groupTransaccionsByMonth(RFM_df):
    
    """
    Groups the transactions by month and calculates the mean of each feature

    Args:
        none

    Returns:
        groupbby_month: A dataframe with the mean of each feature base on month

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """
    RFM_df = RFM_df.sort_values(by='TransactionDate')
    groupbby_month = RFM_df.groupby([pd.Grouper(key='TransactionDate', freq='M')])[['Frequency', 'TransactionAmount', 'CustAccountBalance', 'TransactionTime', 'CustomerAge']].mean()
    print(groupbby_month.shape)
    return groupbby_month

groupbby_month = groupTransaccionsByMonth(RFM_df)
groupbby_month.head()

def replaceGenderforInt():
    
    """
    Replace de gender data to -1 if is women and 1 if is men

    Args:
        none

    Returns:
        none

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """
    RFM_df.CustGender.replace(['F','M'],[-1,1],inplace=True)
    RFM_df.CustGender = RFM_df.CustGender.astype(np.int64)

replaceGenderforInt()

RFM_df.drop(['TransactionDate'], axis=1, inplace=True) # Porque creamos 3 variable basadas en la fecha, Dia, Mes, Nombre dia
RFM_df.drop(['Recency'], axis=1, inplace=True) # Correlación con variable frecuuencia

def dataToEncoder(RFM_df):
      
    """
      Apply the label encoder to the data, transform variable type objects or string to int

      Args:
        The Dataset

      Returns:
        none

      Raises:
        TypeError: If the dataframe is not a DataFrame.
    """
    encoder = LabelEncoder()
    RFM_df.CustLocation = encoder.fit_transform(RFM_df.CustLocation)
    RFM_df.TransactionMonthName = encoder.fit_transform(RFM_df.TransactionMonthName)
    RFM_df.TransactionDayName = encoder.fit_transform(RFM_df.TransactionDayName)
    # Custom Cast
    RFM_df.CustLocation = RFM_df.CustLocation.astype(np.int64)
    RFM_df.TransactionMonthName = RFM_df.TransactionMonthName.astype(np.int64)
    RFM_df.TransactionDayName = RFM_df.TransactionDayName.astype(np.int64)
    return RFM_df
    
RFM_df = dataToEncoder(RFM_df)

RFM_df.drop(['TransactionMonth', 'TransactionMonthName', 'TransactionDay', 'TransactionDayName'], axis=1, inplace=True)

def scale_data(df):
    """
    Scale the data using StandardScaler.

    Args:
        df (pd.DataFrame): The input data to be scaled.

    Returns:
        pd.DataFrame: The scaled data.

    """
    column_names = df.columns
    scaler = StandardScaler()
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), columns=column_names)
    return df

pipeline = Pipeline([
    ('scaler', FunctionTransformer(scale_data)),
])

RFM_df = pipeline.fit_transform(RFM_df)
RFM_df.head()

def encode_units(x):
    """
    Encode units to binary values.

    Args:
      x: input unit.

    Returns:
      int: 1 if x >= 1, 0 otherwise.
    """
    if x <= 0:
      return 0
    if x >= 1:
      return 1