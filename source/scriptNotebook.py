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

def replaceGenderforInt():
    """
    Replace de gender data to -1 if is women and 1 if is men

    Args:
        none

    Returns:
        none
    """

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

def pca_analysis(df):
    """
    We are going to do the method of PCA to find the best features

    Args:
        The Dataset

    Returns:
        none

    Raises:
        TypeError: If the dataframe is not a DataFrame.
    """

def scale_data(df):
    """
    Scale the data using StandardScaler.

    Args:
        df (pd.DataFrame): The input data to be scaled.

    Returns:
        pd.DataFrame: The scaled data.

    """

def plot_boxplot(df, columns):
    """
    Crea un gráfico de boxplot para las columnas especificadas de un DataFrame normalizado.

    Args:
        df (pandas.DataFrame): El DataFrame a graficar.
        columns (list): Lista de nombres de columnas para graficar.

    Returns:
        None
    """

def encode_units(x):
    """
    Encode units to binary values.

    Args:
      x: input unit.

    Returns:
      int: 1 if x >= 1, 0 otherwise.
    """

def get_frequent_itemsets(df):
    """
    Apply apriori algorithm on input DataFrame to obtain frequent itemsets.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Dataframe with frequent itemsets and their support.

    """

def calculate_sse(data):
    """
    Calculates the sum of squared errors (SSE) for a range of k values (number of clusters)
    using the KMeans algorithm and plots the resulting SSE values.

    Args:
      data: A pandas DataFrame with the input data for KMeans clustering.

    Returns:
      A dictionary with the SSE values for each k value.
    """

def elbowFindValue(RFM_df_importance):
    """
    This function fits a KMeans model and generates an elbow plot to help identify the optimal number of clusters.
    The function returns the optimal number of clusters identified by the elbow plot.  
  
    Args:
      df (pandas.DataFrame): DataFrame with the data to fit the model to.
    
    Returns:
      none
    """

def groupByCluster3(RFM_df_importance, model1):
    """
    This function generates a summary of the clusters generated by a KMeans model.
    
    Args:
      df (pandas.DataFrame): DataFrame with the data used to generate the clusters.
      model1: model of the kmeans fit
    Returns:
      RFM_df_kmeans3: the new Dataset
    """

def viewTheClustersKmeans3(RFM_df_kmeans3):
    """
    This function generate the plot with the clusters
    
    Args:
      df (pandas.DataFrame): DataFrame with the data used to generate the clusters.
    
    Returns:
      none
    """

def groupByCluster5(RFM_df_importance, model2):
    """
    This function generates a summary of the clusters generated by a KMeans model.
    
    Args:
      df (pandas.DataFrame): DataFrame with the data used to generate the clusters.
      model2: model of the kmeans fit
    
    Returns:
      RFM_df_kmeans5: the new Dataset
    """

def viewTheClustersKmeans5(RFM_df_kmeans5):
    """
    This function generate the plot with the clusters
    
    Args:
      df (pandas.DataFrame): DataFrame with the data used to generate the clusters.
  
    Returns:
      none
    """

def calculateSample(RFM_df_importance):
    """
    This function calculates the sample size needed based on the number of observations in the `RFM_df_importance` dataset.

    Args:
      none

    Returns:
       none
    """

def hierarchicalClusteringAge(RFM_df_importance):
    """
      Perform hierarchical clustering on the age and account balance variables in the input DataFrame using Ward's method.
    
    Args:
      RFM_df_importance : pandas DataFrame
        A DataFrame containing customer information, including the 'CustomerAge' and 'CustAccountBalance' columns.

    Returns:
      dendogram_age : dict
        A dictionary containing information about the dendrogram visualization of the hierarchical clustering. The keys are 'icoord', 'dcoord', 'leaves', and 'color_list', which correspond to the output of scipy's dendrogram function.
    """

def getBestClustersAge(dendogram_age):
    """
    Determines the optimal number of clusters based on a dendrogram obtained through hierarchical clustering.

    Args:
      dendogram_age (dict): A dictionary containing the dendrogram information.

    Returns:
      num_clusters_age (int): The optimal number of clusters.
    """

def hierarchicalClusteringLocation(RFM_df_importance):
    """
    Perform hierarchical clustering on the age and account balance variables in the input DataFrame using Ward's method.
    
    Args:
      RFM_df_importance : pandas DataFrame
        A DataFrame containing customer information, including the 'CustomerAge' and 'CustAccountBalance' columns.

    Returns:
      dendogram_location : dict
        A dictionary containing information about the dendrogram visualization of the hierarchical clustering. The keys are 'icoord', 'dcoord', 'leaves', and 'color_list', which correspond to the output of scipy's dendrogram function.
    """

def getBestClustersLocation(dendogram_location):
    """
    Determines the optimal number of clusters based on a dendrogram obtained through hierarchical clustering.

    Args:
      dendogram_location (dict): A dictionary containing the dendrogram information.

    Returns:
      num_clusters_location (int): The optimal number of clusters.
    """

def dbscanInitial(temp):
    """
    Performs DBSCAN clustering on a sample of data using the default settings.

    Args:
      temp: pandas DataFrame
        The DataFrame containing the data to be clustered.

    Returns:
      None
    """

def get_kdist_plot(X=None, k=None, radius_nbrs=1.0):
    """
    Plots the k-distance graph for a given dataset and returns the distance value at the knee point.

    Args:
        X (numpy array or pandas DataFrame): The input dataset.
        k (int): The number of neighbors to consider.
        radius_nbrs (float): The radius of neighbors to consider.

    Returns:
        float: The distance value at the knee point.

    Raises:
        ValueError: If the input dataset is empty or the number of neighbors is negative.
    """

def calculateMinimumSamples(x):
    """
    Calculates the minimum number of samples required for DBSCAN clustering by evaluating the silhouette score
    for different values of the minimum number of samples parameter.
    
    Args:
      x: float
        The distance threshold value for DBSCAN clustering.

    Returns:
      None
    """

def optimizedDbscan(x):
    """
    Apply DBSCAN algorithm to cluster data using the input value for epsilon (eps) and a minimum of 6 samples for each cluster.

    Args:
        x (float): The value of epsilon (eps) to use for DBSCAN clustering.

    Returns:
        optimized_dbscan (DBSCAN object): An instance of the DBSCAN algorithm with the specified parameters fit to the input data.
    """

def optDbscanClusteval(temp):
    """
    Evaluates optimal number of clusters for DBSCAN clustering algorithm using clusteval library on input data.
    
    Args:
        temp : numpy array or pandas DataFrame
            A dataset of input features to be clustered using DBSCAN algorithm.

    Returns:
        ce : clusteval object
            An object of the clusteval library containing clustering evaluation results.
    """

def enhancedDbscan(temp):
    """
    Perform density-based clustering using DBSCAN with an enhanced hyperparameter setting.

    Args:
        temp (pandas.DataFrame): A DataFrame containing the data to be clustered.

    Returns:
        enenhanced_dbscan (sklearn.cluster.DBSCAN): A DBSCAN object that has been fit to the data.
    """