import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler

def dateConvertion(df):
    df["CustomerDOB"] = pd.to_datetime(df["CustomerDOB"], dayfirst=True)
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], dayfirst=True)
    df['TransactionDate1'] = df['TransactionDate']
    df['TransactionDate2'] = df['TransactionDate']
    return df


def refactorDates(df):    
    df.loc[df['CustomerDOB'].dt.year > 1999, 'CustomerDOB'] -= pd.DateOffset(years=100)
    return df


def getCustomerAge(df):    
    df['CustomerAge'] = (df['TransactionDate'] - df['CustomerDOB'])/np.timedelta64(1, 'Y')
    df['CustomerAge'] = df['CustomerAge'].astype(int)
    return df


def RFMTable(df):
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
    RFM_df.reset_index(inplace=True)
    RFM_df.rename(columns={"TransactionID":"Frequency"},inplace=True)
    RFM_df['Recency']=RFM_df['TransactionDate2']-RFM_df['TransactionDate1']
    RFM_df['Recency']=RFM_df['Recency'].astype(str)
    
    return RFM_df
    

def formatOutputInRecency(RFM_df):     
    RFM_df['Recency']=RFM_df['Recency'].apply(lambda x :re.search('\d+',x).group())
    RFM_df['Recency']=RFM_df['Recency'].astype(int) 
    RFM_df['Recency']=RFM_df['Recency'].apply(lambda x: 1 if x == 0 else x)
    RFM_df.drop(columns=["TransactionDate1","TransactionDate2"],inplace=True)
    return RFM_df


def groupbby_month_RFM(RFM_df):

    RFM_df['TransactionMonth'] = RFM_df["TransactionDate"].dt.month
    RFM_df['TransactionMonthName'] = RFM_df["TransactionDate"].dt.month_name()
    RFM_df['TransactionDay'] = RFM_df["TransactionDate"].dt.day
    RFM_df['TransactionDayName'] = RFM_df["TransactionDate"].dt.day_name()
    return RFM_df


def replaceGenderforInt(RFM_df):
    RFM_df.CustGender.replace(['F','M'],[-1,1],inplace=True)
    RFM_df.CustGender = RFM_df.CustGender.astype(np.int64)
    return RFM_df


def dataToEncoder(RFM_df):

    RFM_df.drop(['TransactionDate'], axis=1, inplace=True) # Porque creamos 3 variable basadas en la fecha, Dia, Mes, Nombre dia
    RFM_df.drop(['Recency'], axis=1, inplace=True) # Correlación con variable frecuuencia
    RFM_df.drop(['CustomerID'], axis=1, inplace=True) # No aporta valor al modelo
    
    encoder = LabelEncoder()
    RFM_df.CustLocation = encoder.fit_transform(RFM_df.CustLocation)
    RFM_df.TransactionMonthName = encoder.fit_transform(RFM_df.TransactionMonthName)
    RFM_df.TransactionDayName = encoder.fit_transform(RFM_df.TransactionDayName)
    # Custom Cast
    RFM_df.CustLocation = RFM_df.CustLocation.astype(np.int64)
    RFM_df.TransactionMonthName = RFM_df.TransactionMonthName.astype(np.int64)
    RFM_df.TransactionDayName = RFM_df.TransactionDayName.astype(np.int64)

    RFM_df.drop(['TransactionMonth', 'TransactionMonthName', 'TransactionDay', 'TransactionDayName'], axis=1, inplace=True)

    return RFM_df
    

def scale_data(RFM_df):
    
    column_names = RFM_df.columns
    scaler = StandardScaler()
    scaler.fit(RFM_df)
    RFM_DF = pd.DataFrame(scaler.transform(RFM_df), columns=column_names)
    return RFM_df

def importance_columns(RFM_df):
    RFM_df = RFM_df[["Frequency","CustLocation", "CustGender", "CustAccountBalance", "TransactionAmount", "CustomerAge"]]
    return RFM_df