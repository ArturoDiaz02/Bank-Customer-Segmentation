import streamlit as st
import joblib
import scipy.cluster.hierarchy as sch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler, StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram


def scatterplot(df, prediction, model):
    """
    Generate a scatter plot based on data from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        prediction (str): The column in the DataFrame that contains the predictions/clusters.
        model: Model used for making predictions (not used in the method).

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    colores = ['red', 'green', 'blue']
    asignar = []
    

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    sns.scatterplot(data=df ,x='CustAccountBalance', y='TransactionAmount', hue=prediction,s=300,alpha=0.6,palette='summer')
    ax.set_xlabel('CustAccountBalance')
    ax.set_ylabel('TransactionAmount')

    plt.title('Gráfico de los clusters')

    return fig

def radarchar(df, predict):
    """
    Generate a radar chart based on data from a DataFrame and predictions/clusters.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        predict: The predictions/clusters (not used in the method).

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """

    predictions_df = pd.DataFrame(predict, columns=['Cluster'])
    merged_df = pd.concat([df, predictions_df], axis=1)
    df_aux = merged_df.drop(columns=['CustLocation', 'CustGender', 'Frequency'])

    categories = ['CustAccountBalance', 'TransactionAmount', "CustomerAge"]

    scaler = MinMaxScaler()
    df_aux_norm = pd.DataFrame(scaler.fit_transform(df_aux), columns=df_aux.columns)

    avg_values = df_aux_norm.groupby('Cluster')[categories].mean()

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    count = 0
    for cluster, values in avg_values.iterrows():
        cluster_values = values.values.tolist()
        cluster_values += [cluster_values[0]]
        ax.plot(angles, cluster_values, linewidth=1, label=f'Cluster {count}')
        ax.fill(angles, cluster_values, alpha=0.25)
        count += 1

    ax.legend(loc='upper right')
    ax.set_title('Gráfico de Radar por Cluster')
    return fig

def numclusters(predict):
    """
    Generate a bar chart showing the number of instances per cluster.

    Args:
        predict: The predictions/clusters.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    unique, counts = np.unique(predict, return_counts=True)
    items = list(dict(zip(unique, counts)).items())

    clusters = [cluster[0] for cluster in items]
    counts = [cluster[1] for cluster in items]

    fig = plt.figure(figsize=(10, 7))
    plt.bar(clusters, counts)
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.title('Número de instancias por cluster')
    return fig

def numclustersTable(predict):
    unique, counts = np.unique(predict, return_counts=True)
    items = list(dict(zip(unique, counts)).items())

    return items

def dateConvertion(df):
    """
    Convert date columns in the DataFrame to datetime format.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: The DataFrame with converted date columns.
    """
    df["CustomerDOB"] = pd.to_datetime(df["CustomerDOB"], dayfirst=True)
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], dayfirst=True)
    df['TransactionDate1'] = df['TransactionDate']
    df['TransactionDate2'] = df['TransactionDate']
    return df


def refactorDates(df):
    """
    Refactor the CustomerDOB column in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: The DataFrame with refactored CustomerDOB column.
    """
    df.loc[df['CustomerDOB'].dt.year > 1999, 'CustomerDOB'] -= pd.DateOffset(years=100)
    return df


def getCustomerAge(df):
    """
    Calculate the CustomerAge based on the CustomerDOB and TransactionDate columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: The DataFrame with the calculated CustomerAge column.
    """
    df['CustomerAge'] = (df['TransactionDate'] - df['CustomerDOB']) / np.timedelta64(1, 'Y')
    df['CustomerAge'] = df['CustomerAge'].astype(int)
    return df


def RFMTable(df):
    """
    Generate an RFM table based on the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: The RFM table.
    """
    RFM_df = df.groupby("CustomerID").agg({
        "TransactionID": "count",
        "CustGender": "first",
        "CustLocation": "first",
        "CustAccountBalance": "mean",
        "TransactionTime": "mean",
        "TransactionAmount": "mean",
        "CustomerAge": "median",
        "TransactionDate2": "max",
        "TransactionDate1": "min",
        "TransactionDate": "median"
    })
    RFM_df.reset_index(inplace=True)
    RFM_df.rename(columns={"TransactionID": "Frequency"}, inplace=True)
    RFM_df['Recency'] = RFM_df['TransactionDate2'] - RFM_df['TransactionDate1']
    RFM_df['Recency'] = RFM_df['Recency'].astype(str)

    return RFM_df


def formatOutputInRecency(RFM_df):
    """
    Format the Recency column in the RFM table.

    Args:
        RFM_df (pd.DataFrame): The RFM table.

    Returns:
        pd.DataFrame: The RFM table with formatted Recency column.
    """
    RFM_df['Recency'] = RFM_df['Recency'].apply(lambda x: re.search('\d+', x).group())
    RFM_df['Recency'] = RFM_df['Recency'].astype(int)
    RFM_df['Recency'] = RFM_df['Recency'].apply(lambda x: 1 if x == 0 else x)
    RFM_df.drop(columns=["TransactionDate1", "TransactionDate2"], inplace=True)
    return RFM_df


def groupbby_month_RFM(RFM_df):
    """
    Add month and day-related columns to the RFM table based on the TransactionDate.

    Args:
        RFM_df (pd.DataFrame): The RFM table.

    Returns:
        pd.DataFrame: The RFM table with added month and day-related columns.
    """
    RFM_df['TransactionMonth'] = RFM_df["TransactionDate"].dt.month
    RFM_df['TransactionMonthName'] = RFM_df["TransactionDate"].dt.month_name()
    RFM_df['TransactionDay'] = RFM_df["TransactionDate"].dt.day
    RFM_df['TransactionDayName'] = RFM_df["TransactionDate"].dt.day_name()
    return RFM_df


def replaceGenderforInt(RFM_df):
    """
    Replace gender values in the RFM table with corresponding integer values.

    Args:
        RFM_df (pd.DataFrame): The RFM table.

    Returns:
        pd.DataFrame: The RFM table with replaced gender values.
    """
    RFM_df.CustGender.replace(['F', 'M'], [-1, 1], inplace=True)
    RFM_df.CustGender = RFM_df.CustGender.astype(np.int64)
    return RFM_df


def dataToEncoder(RFM_df):
    """
    Prepare the RFM table for encoding by dropping unnecessary columns and applying label encoding.

    Args:
        RFM_df (pd.DataFrame): The RFM table.

    Returns:
        pd.DataFrame: The preprocessed RFM table.
    """
    RFM_df.drop(['TransactionDate'], axis=1,
                inplace=True)
    RFM_df.drop(['Recency'], axis=1, inplace=True)
    RFM_df.drop(['CustomerID'], axis=1, inplace=True)

    encoder = LabelEncoder()
    RFM_df.CustLocation = encoder.fit_transform(RFM_df.CustLocation)
    RFM_df.TransactionMonthName = encoder.fit_transform(RFM_df.TransactionMonthName)
    RFM_df.TransactionDayName = encoder.fit_transform(RFM_df.TransactionDayName)
    RFM_df.CustLocation = RFM_df.CustLocation.astype(np.int64)
    RFM_df.TransactionMonthName = RFM_df.TransactionMonthName.astype(np.int64)
    RFM_df.TransactionDayName = RFM_df.TransactionDayName.astype(np.int64)

    RFM_df.drop(['TransactionMonth', 'TransactionMonthName', 'TransactionDay', 'TransactionDayName'], axis=1,
                inplace=True)

    return RFM_df

def valueToEncoder(val):
    """
    Apply label encoding to a given value.

    Args:
        val: The value to encode.

    Returns:
        The encoded value.
    """
    encoder = LabelEncoder()
    val = encoder.fit_transform(val)
    return val

def scale_data(RFM_df):
    """
    Scale the numeric columns in the RFM table using robust scaler.

    Args:
        RFM_df (pd.DataFrame): The RFM table.

    Returns:
        pd.DataFrame: The scaled RFM table.
    """
    column_names = RFM_df.columns
    scaler = RobustScaler()
    scaler.fit(RFM_df)
    RFM_DF = pd.DataFrame(scaler.transform(RFM_df), columns=column_names)
    return RFM_df


def importance_columns(RFM_df):
    """
    Select the important columns from the RFM table.

    Args:
        RFM_df (pd.DataFrame): The RFM table.

    Returns:
        pd.DataFrame: The RFM table with selected important
    """
    RFM_df = RFM_df[
        ["Frequency", "CustLocation", "CustGender", "CustAccountBalance", "TransactionAmount", "CustomerAge"]]
    return RFM_df
