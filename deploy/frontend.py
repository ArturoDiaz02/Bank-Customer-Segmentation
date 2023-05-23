import streamlit as st
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import re

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder, StandardScaler
from backend import *

st.title("Bank Customer Segmentation - Model KMeans Prediction")

def main():
    tab1, tab2 = st.tabs(["Cargar Datos", "Información sobre los clusters"])

    with tab2:
        st.header("¿Cuales son los clusters?")
        st.image("resources/clusters.png")
        st.image("resources/scatterplot.png")
        st.write(
            "El algoritmo de agrupación espectral ha dividido a los clientes en tres grupos distintos. El **primer** grupo está formado por los clientes dinámicos 👥 que tienen un saldo de cuenta más bajo y suelen gastar menos efectivo en las transacciones. El **segundo** son mujeres de entre 20 y 30 años que realizan transacciones de gran valor 👩🏻🛍️. El **tercer** grupo de hombres de entre 30 y 40 años 👨🏻💼 que trabajan y guardan dinero en su cuenta para una posible inversión.")

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
            df = scale_data(df)
            df = importance_columns(df)
            st.write("Datos procesados correctamente")

            if st.button("Predecir"):
                model = joblib.load("model.joblib")
                predict = model.predict(df)
                st.write("Predicción: ", predict)
                st.write("Número de elementos en cada cluster: ", numclusters(predict))
                st.write("Gráficos de los clusters:")
                st.pyplot(scatterplot(df, predict, model))
                st.pyplot(lineplot(df, predict))


if __name__ == "__main__":
    main()
