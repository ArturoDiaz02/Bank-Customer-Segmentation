import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from backend import *

st.title("Bank Customer Segmentation - Model KMeans Prediction")


def main():
    tab1, tab2, tab3, tab4 = st.tabs(["Cargar Datos", "Evaluación individual", "Información sobre los clusters", "Información del Datset"])

    with tab3:
        st.header("¿Cuales son los clusters?")
        st.image("./deploy/resources/clusters.png")
        st.image("./deploy/resources/scatterplot.png")
        st.write(
            "El algoritmo de agrupación espectral ha dividido a los clientes en tres grupos distintos. ✨ El **primer** grupo está compuesto por aquellas personas que son dinámicas, es decir, que realizan muchas transacciones 💸 y, por ende, tienen poco saldo en su cuenta. Además, está compuesto en su mayoría por personas jóvenes 🌟 y donde predominan las mujeres 👩‍💼. El segundo **grupo** son aquellas personas que deciden ahorrar más dinero 💰 y, por ende, no realizan muchas transacciones. Está compuesto en su mayoría por hombres de avanzada edad 👴 que tienen una mentalidad ahorradora. Por último, el **tercer** grupo está compuesto por aquellas personas que realizan más transacciones que los del grupo 2 pero menos que los del grupo 1. Está compuesto por hombres y mujeres entre 30 y 50 años 👨‍👩‍👧‍👦. Esta es toda la información que se tiene de los clusters. 📊")
    with tab4:
        st.header("Columnas del Dataset")
        variable_list = [
            {"Variable": "TransactionID", "Descripción": "ID único por cada transacción realizada"},
            {"Variable": "CustomerID", "Descripción": "ID único de un cliente"},
            {"Variable": "CustomerDOB", "Descripción": "Fecha de nacimiento de cada cliente"},
            {"Variable": "CustLocation",
             "Descripción": "Localización de cada cliente, lugar donde se hizo la transacción"},
            {"Variable": "CustAccountBalance", "Descripción": "Cantidad de dinero en la cuenta de cada cliente"},
            {"Variable": "TransactionDate", "Descripción": "Fecha en la que realizo la transacción"},
            {"Variable": "CustGender", "Descripción": "Genero de cada cliente"},
            {"Variable": "TransactionTime", "Descripción": "Tiempo que un cliente tardó en realizar una transacción"},
            {"Variable": "TransactionAmount (INR)", "Descripción": "Cantidad de dinero girado por cada transacción"}
        ]

        df_variables = pd.DataFrame(variable_list)
        st.write("Para predecir los datos de tu dataset, debe de estar compuesto de la siguiente manera")

        st.table(df_variables)

    with tab2:
        st.header("Ingresa los datos para predicir")
        fr = st.number_input("Ingresa la frecuencia")
        lc = st.text_input("Ingresa el nombre del cliente (En mayúsculas)")
        gn = st.selectbox("Ingresa el género", ["Masculino", "Femenino"])
        bl = st.number_input("Ingresa el balance de la cuenta")
        tr = st.number_input("Ingresa la cantidad de transacciones")
        age = st.number_input("Ingresa la edad del cliente")

        if(st.button("Predecir")):

            if fr is None or lc or gn or bl is None or tr is None or age is None:
                st.write("Ingresa todos los datos")
                return         
            if lc:
                lc = np.array([lc])  # Convert lc to a 1-dimensional array
                lc = valueToEncoder(lc)
            model = joblib.load("./deploy/model.joblib")
            if gn == "Masculino":
                gn = 1
            else:
                gn = -1
            df = pd.DataFrame(
                [[fr, lc, gn, bl, tr, age]], columns=['Frequency', 'CustLocation', 'CustGender', 'CustAccountBalance', 'TransactionAmount', 'CustomerAge'])
            column_names = df.columns
            scaler = RobustScaler()
            scaler.fit(df)
            df = pd.DataFrame(scaler.transform(df), columns=column_names)
            predict = model.predict(df)
            st.write("Predicción: ", predict)
            st.write(numclustersTable(predict))


    with tab1:
        uploaded_file = st.file_uploader("Sube tu archivo CSV aquí", type="csv")
        if uploaded_file is not None:
            st.write("Archivo subido correctamente, espere mientras transformamos los datos...")
            df = pd.read_csv(uploaded_file)
            df = dateConvertion(df)
            df = refactorDates(df)
            df = getCustomerAge(df)
            st.write("Creando la tabla RFM...")
            df = RFMTable(df)
            df = formatOutputInRecency(df)
            df = groupbby_month_RFM(df)
            df = replaceGenderforInt(df)
            df = dataToEncoder(df)
            column_names = df.columns
            scaler = RobustScaler()
            scaler.fit(df)
            df = pd.DataFrame(scaler.transform(df), columns=column_names)
            df = importance_columns(df)
            st.write(df)
            st.write("Datos procesados correctamente")
            if st.button("Predecir"):
                model = joblib.load("./deploy/model.joblib")
                predict = model.predict(df)
                st.write("Predicción: ", predict)
                st.write(numclustersTable(predict))

                predictions_df = pd.DataFrame(predict, columns=['Cluster'])
                merged_df = pd.concat([df, predictions_df], axis=1)
                df_aux = merged_df.drop(columns=['CustLocation', 'CustGender', 'Frequency'])

                categories = ['CustAccountBalance', 'TransactionAmount', "CustomerAge"]


                avg_values = df_aux.groupby('Cluster')[categories].mean()
                st.write(avg_values)
                st.write("Gráficos de los clusters:")
                st.pyplot(scatterplot(df, predict, model))
                st.pyplot(radarchar(df, predict))


if __name__ == "__main__":
    main()
