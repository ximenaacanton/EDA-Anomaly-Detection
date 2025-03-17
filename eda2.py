# -----------------------------------------------------------------Hipotesis1PiSA.ipynb-----------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_TEMP_LAYER_1 = pd.read_csv("CONTIFORM_MMA_CONTIFORM_MMA1_PreformTemperatureLayer_1.csv", index_col=0,parse_dates=True).drop(columns=["message"], errors="ignore").iloc[:57097]
df_TEMP_LAYER_3 = pd.read_csv("CONTIFORM_MMA_CONTIFORM_MMA1_PreformTemperatureLayer_3.csv", index_col=0,parse_dates=True).drop(columns=["message"], errors="ignore").iloc[:57097]
df_TEMP_LAYER_5 = pd.read_csv("CONTIFORM_MMA_CONTIFORM_MMA1_PreformTemperatureLayer_5.csv", index_col=0,parse_dates=True).drop(columns=["message"], errors="ignore").iloc[:57097]
df_TEMP_LAYER_7 = pd.read_csv("CONTIFORM_MMA_CONTIFORM_MMA1_PreformTemperatureLayer_7.csv", index_col=0,parse_dates=True).drop(columns=["message"], errors="ignore").iloc[:57097]
df_TEMP_LAYER_9 = pd.read_csv("CONTIFORM_MMA_CONTIFORM_MMA1_PreformTemperatureLayer_9.csv", index_col=0,parse_dates=True).drop(columns=["message"], errors="ignore").iloc[:57097]
df_RECHAZOS = pd.read_csv("CONTIFORM_MMA_CONTIFORM_MMA1_WS_Tot_Rej_0.csv", index_col=0,parse_dates=True).drop(columns=["message"], errors="ignore",inplace=True)

df_RECHAZOS = pd.read_csv("CONTIFORM_MMA_CONTIFORM_MMA1_WS_Tot_Rej_0.csv", index_col=0,parse_dates=True).drop(columns=["message"], errors="ignore")

# Assign the DataFrame with dropped columns to df_RECHAZOS.
df = df_RECHAZOS.copy()
df["PreformTemperatureLayer_1"] = df_TEMP_LAYER_1["value"] # Select the 'value' column
df["PreformTemperatureLayer_3"] = df_TEMP_LAYER_3["value"] # Select the 'value' column
df["PreformTemperatureLayer_5"] = df_TEMP_LAYER_5["value"] # Select the 'value' column
df["PreformTemperatureLayer_7"] = df_TEMP_LAYER_7["value"] # Select the 'value' column
df["PreformTemperatureLayer_9"] = df_TEMP_LAYER_9["value"] # Select the 'value' column

umbral_rechazo = pd.to_numeric(df[df_RECHAZOS.columns[0]], errors='coerce').quantile(0.75)
df["Grupo Rechazo"] = ["Alto" if x > umbral_rechazo else "Bajo" for x in pd.to_numeric(df[df_RECHAZOS.columns[0]], errors='coerce')]

for var in ["PreformTemperatureLayer_1", "PreformTemperatureLayer_3", "PreformTemperatureLayer_5", "PreformTemperatureLayer_7", "PreformTemperatureLayer_9"]:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="Grupo Rechazo", y=var, data=df)
    plt.title(f"Distribución de {var} en Lotes con Alto y Bajo Rechazo")
    plt.xlabel("Grupo de Rechazo")
    plt.ylabel("Temperatura de la Preforma")
    plt.show()
    
    
# ----------------------------------------------------------------------Layer9PiSA.ipynb---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore

# Configuración de estilo
sns.set_style("whitegrid")

# Ruta del archivo CSV
archivo_csv = "CONTIFORM_MMA_CONTIFORM_MMA1_PreformTemperatureLayer_9.csv"  # MODIFICA ESTO

# Cargar el CSV
data = pd.read_csv(archivo_csv)
print(data)

# ---- 1.1. Análisis Descriptivo Inicial ----
print("Información del dataset:")
print(data.info())
print("\nValores nulos:")
print(data.isnull().sum())
print("\nValores duplicados:")
print(data.duplicated().sum())

data.dropna(subset=['value'], inplace=True)
data.drop(columns=['message'], inplace=True)

data

# Resumen estadístico
print("\nResumen estadístico:")
print(data.describe(percentiles=[0.25, 0.5, 0.75]))

# ---- 1.2. Análisis de Distribución y Detección de Outliers ----
plt.figure(figsize=(10, 5))
for col in data.select_dtypes(include=np.number).columns:
    plt.figure()
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f"Distribución de {col}")
    plt.show()

    plt.figure()
    sns.boxplot(y=data[col])
    plt.title(f"Boxplot de {col}")
    plt.show()

# Z-score para detectar outliers
    data[f"{col}_zscore"] = np.abs(zscore(data[col]))
    print(f"Valores atípicos detectados en {col} (Z-score > 3):")
    print(data[data[f"{col}_zscore"] > 3][[col]])

# IQR (Interquartile Range Method) para detectar outliers
for col in data.select_dtypes(include=np.number).columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))]
    print(f"Valores atípicos detectados en {col} por IQR:")
    print(outliers[[col]])

if "user_ts" in data.columns:
    # El fromato "string" ahora coincide con el formato de los datos 
    data["user_ts"] = pd.to_datetime(data["user_ts"], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')
    data = data.dropna(subset=['user_ts'])
    data = data.sort_values(by="user_ts")
    for col in data.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(10, 4))
        plt.plot(data["user_ts"], data[col], label=col)
        plt.title(f"Serie de tiempo de {col}")
        plt.legend()
        plt.show()

# Rolling Mean & Std
    for col in data.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(10, 4))
        plt.plot(data["user_ts"], data[col].rolling(window=30).mean(), label="Rolling Mean")
        plt.plot(data["user_ts"], data[col].rolling(window=30).std(), label="Rolling Std")
        plt.title(f"Rolling Mean & Std de {col}")
        plt.legend()
        plt.show()

# ---- 1.4. Detección de Anomalías Basada en Modelos ----
X = data.select_dtypes(include=np.number)  # Solo variables numéricas
if not X.empty:
    X = X.dropna()
    # Regresión lineal
    if X.shape[1] > 1:
        reg = LinearRegression()
        reg.fit(X.iloc[:, :-1], X.iloc[:, -1])
        print("Coeficientes de la regresión lineal:")
        print(reg.coef_)

    # DBSCAN para detección de anomalías
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    data["dbscan_anomaly"] = dbscan.fit_predict(X)

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    data["isolation_anomaly"] = iso_forest.fit_predict(X)

    # LOF (Local Outlier Factor)
    lof = LocalOutlierFactor(n_neighbors=20)
    data["lof_anomaly"] = lof.fit_predict(X)

    # ---- Visualizar anomalías ----
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1] if X.shape[1] > 1 else X.iloc[:, 0], hue=data["isolation_anomaly"], palette={1:"blue", -1:"red"})
    plt.title("Anomalías detectadas con Isolation Forest")
    plt.legend(title="Anomalía", labels=["Normal", "Anómalo"])
    plt.show()

    print("\nCantidad de anomalías detectadas por cada método:")
    print("Isolation Forest:")
    print(data["isolation_anomaly"].value_counts())
    print("DBSCAN:")
    print(data["dbscan_anomaly"].value_counts())
    print("Local Outlier Factor (LOF):")
    print(data["lof_anomaly"].value_counts())


# -------------------------------------------------------------------Layer1PiSA.ipynb-------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore

# Configuración de estilo
sns.set_style("whitegrid")

# Ruta del archivo CSV
archivo_csv = "CONTIFORM_MMA_CONTIFORM_MMA1_PreformTemperatureLayer_1.csv"  # MODIFICA ESTO

# Cargar el CSV
data = pd.read_csv(archivo_csv)
print(data)

# ---- 1.1. Análisis Descriptivo Inicial ----
print("Información del dataset:")
print(data.info())
print("\nValores nulos:")
print(data.isnull().sum())
print("\nValores duplicados:")
print(data.duplicated().sum())

"""Eliminar valores nulos de "value" y eliminar la columna "message" pues son puros valores nulos."""

data.dropna(subset=['value'], inplace=True)
data.drop(columns=['message'], inplace=True)

data

# Resumen estadístico
print("\nResumen estadístico:")
print(data.describe(percentiles=[0.25, 0.5, 0.75]))

# ---- 1.2. Análisis de Distribución y Detección de Outliers ----
plt.figure(figsize=(10, 5))
for col in data.select_dtypes(include=np.number).columns:
    plt.figure()
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f"Distribución de {col}")
    plt.show()

    plt.figure()
    sns.boxplot(y=data[col])
    plt.title(f"Boxplot de {col}")
    plt.show()

# Z-score para detectar outliers
    data[f"{col}_zscore"] = np.abs(zscore(data[col]))
    print(f"Valores atípicos detectados en {col} (Z-score > 3):")
    print(data[data[f"{col}_zscore"] > 3][[col]])

# IQR (Interquartile Range Method) para detectar outliers
for col in data.select_dtypes(include=np.number).columns:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))]
    print(f"Valores atípicos detectados en {col} por IQR:")
    print(outliers[[col]])

if "user_ts" in data.columns:
    # The format string now matches the timestamp format in your data
    data["user_ts"] = pd.to_datetime(data["user_ts"], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')
    data = data.dropna(subset=['user_ts'])
    data = data.sort_values(by="user_ts")
    for col in data.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(10, 4))
        plt.plot(data["user_ts"], data[col], label=col)
        plt.title(f"Serie de tiempo de {col}")
        plt.legend()
        plt.show()

# Rolling Mean & Std
    for col in data.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(10, 4))
        plt.plot(data["user_ts"], data[col].rolling(window=30).mean(), label="Rolling Mean")
        plt.plot(data["user_ts"], data[col].rolling(window=30).std(), label="Rolling Std")
        plt.title(f"Rolling Mean & Std de {col}")
        plt.legend()
        plt.show()

# ---- 1.4. Detección de Anomalías Basada en Modelos ----
X = data.select_dtypes(include=np.number)  # Solo variables numéricas
if not X.empty:
    X = X.dropna()
    # Regresión lineal
    if X.shape[1] > 1:
        reg = LinearRegression()
        reg.fit(X.iloc[:, :-1], X.iloc[:, -1])
        print("Coeficientes de la regresión lineal:")
        print(reg.coef_)

    # DBSCAN para detección de anomalías
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    data["dbscan_anomaly"] = dbscan.fit_predict(X)

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    data["isolation_anomaly"] = iso_forest.fit_predict(X)

    # LOF (Local Outlier Factor)
    lof = LocalOutlierFactor(n_neighbors=20)
    data["lof_anomaly"] = lof.fit_predict(X)

    # ---- Visualizar anomalías ----
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1] if X.shape[1] > 1 else X.iloc[:, 0], hue=data["isolation_anomaly"], palette={1:"blue", -1:"red"})
    plt.title("Anomalías detectadas con Isolation Forest")
    plt.legend(title="Anomalía", labels=["Normal", "Anómalo"])
    plt.show()

    print("\nCantidad de anomalías detectadas por cada método:")
    print("Isolation Forest:")
    print(data["isolation_anomaly"].value_counts())
    print("DBSCAN:")
    print(data["dbscan_anomaly"].value_counts())
    print("Local Outlier Factor (LOF):")
    print(data["lof_anomaly"].value_counts())