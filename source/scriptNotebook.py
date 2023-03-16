import re
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import seaborn as sns


def import_data():
    """
    Imports our relative path Dataset and in addition to that, renames the TransactionAmount (INR) column in TransactionAmount, 
    this in order to avoid that in the future the special characters of the original name give us problems.
    
    Returns:
        DataFrame: dfGet source data
    """

def dfInformation(dataframe):
    """
    Gets the initial information of the Dataset, the number of records, number of variables, non-null objects and data type.

    Args:
        dataframe (DataFrame): Source dataset.
    
    Returns:
        void: A range indes conforma by float64(2), int64(1), object(6)

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """

def check(dataframe):
    """
    Gets the amount of null data and unique is calculated

    Args:
        dataframe (DataFrame): Source dataset.

    Returns:
        DataFrame: A new Dataframe tha represents de amortized values of null and unique values for each column.

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """
    
def removeNullValues(dataframe):
    """
    Removes null values from data source and calculates the amount eliminated

    Args:
        dataframe (DataFrame): Source dataset.

    Returns:
        int: The total of null values already deleted

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """

def checkDuplicates(dataframe):
    """
    Checks duplicated values for each column and amortized this count.

    Args:
        dataframe (DataFrame): Source dataset.

    Returns:
        int: The total of duplicated values in an specifica dataframe

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """

def uniqueRows(dataframe, column):
    """
    Getting distinct values from column or specific variable

    Args:
        dataframe (DataFrame): Source dataset.
        column (string): Variable or column in dataframe

    Returns:
        Series: A series containing counts of unique rows in the DataFrame.

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """

def removeValues(dataframe,column, value):
    """
    Removes an specific value from a source column in a dataframe

    Args:
        dataframe (DataFrame): Source dataset.
        column (string): Variable or column in dataframe
        value (any): Value with column type

    Returns:
        DataFrame: A pandas DataFrame already modified.

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """

def minAndMax(dataframe, column):
    """
    Gets the minimum and maximum values of any column in order to see in which range the values in this column oscillate.

    Args:
        dataframe (DataFrame): Source dataset.
        column (string): Variable or column in dataframe

    Returns:
        void: Shows the minimun and maximun values from this column

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """

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

def filterDOB(dataframe):
    """
    Filters dataframe by DOB column

    Args:
        dataframe (DataFrame): Source dataset.

    Returns:
        DataFrame: A pandas DataFrame already modified.

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """


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

def outliers(dataframe):
    """
    Calculates the outliers for each numeric column in dataframe

    Args:
        dataframe (DataFrame): Source dataset.

    Returns:
        DataFrame: A pandas DataFrame already modified.

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """

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
 
def outliersWhenCleaned():
    """
    Calculates the outliers for each feature once the data is cleaned

    Args:
        none
    
    Returns:
       print: A print with the outliers for each feature

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """

def correlation():
    """
    Calculates the correlation between features

    Args:
        none

    Returns:
        heatmap: A heatmap with the correlation between features

    Raises:
        TypeError: If the dataframe is not a DataFrame. 
    """

def distributionFrequency():
    """
    Plots the distribution of Frequency variable

    Args:
        none

    Returns:
        chart: A chart with the distribution of Frequency variable

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """
    
def graphAgeAndGender():
    """
    Shows a graph with the distribution of the age and a pie graph of the gender in the dataset

    Args:
        none

    Returns:
        histogram: A histogram with the distribution of the age
        pie: percentage of women and men

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """

def graphLocation():
    """
    Shows a graph with the distribution of the location in the dataset

    Args:
        none

    Returns:
        chart: A chart with the distribution of the location

    Raises:
        TypeError: If the dataframe is not a DataFrame.

    """

def scatterOFData():
    """
    Shows a scatter plot of the data

    Args:
        none

    Returns:
        chart: A chart with the scatter plot of the data

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """
    
def graphTransactionAmountAndCustAccountBalance():
    
    """
    Shows a scatter plot of the TransactionAmount and CustAccountBalance

    Args:
        none

    Returns:
        chart: A chart with the scatter plot of the TransactionAmount and CustAccountBalance

    Raises:
        TypeError: If the dataframe is not a DataFrame.
        
    """

def groupTransaccionsByMonth():
    
    """
    Groups the transactions by month and calculates the mean of each feature

    Args:
        none

    Returns:
        groupbby_month: A dataframe with the mean of each feature

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """

def grphLineBalanceAndTransactionAmount():
    
    """
    Shows a line graph of the average of account balance and transaction amount per month

    Args:
        none

    Returns:
        chart: A chart with the line graph of the average of account balance and transaction amount per month

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """