import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore

layer_5=pd.read_csv("PreformTemperatureLayer_5.csv")
layer_3=pd.read_csv("PreformTemperatureLayer_3.csv")
# ---- 1.1. Análisis Descriptivo Inicial ----
print("Información del dataset layer 3:")
print(layer_3.info())
print("Información del dataset layer 5:")
print(layer_5.info())

print("\nValores nulos layer 3:")
print(layer_3.isnull().sum())
print("\nValores nulos layer 5:")
print(layer_5.isnull().sum())

print("\nValores duplicados layer 3:")
print(layer_3.duplicated().sum())
print("\nValores duplicados layer 5:")
print(layer_5.duplicated().sum())

# Eliminar la columna 'message' ya que es totalmente nula
layer_3.drop(columns=['message'], inplace=True)
layer_5.drop(columns=['message'], inplace=True)
# Manejo de valores nulos en 'value'
if layer_3['value'].isnull().sum() > 0:
    layer_3['value'].fillna(layer_3['value'].median(), inplace=True)  
    
if layer_5['value'].isnull().sum() > 0:
    layer_5['value'].fillna(layer_5['value'].median(), inplace=True) 
    

print("\nResumen estadístico layer 3:")
print(layer_3.describe(percentiles=[0.25, 0.5, 0.75]))
print("\nResumen estadístico layer 5:")
print(layer_5.describe(percentiles=[0.25, 0.5, 0.75]))

# ---- 1.2. Análisis de Distribución y Detección de Outliers ----
plt.figure(figsize=(10, 5))
for col in layer_3.select_dtypes(include=np.number).columns:
    plt.figure()
    sns.histplot(layer_3[col], kde=True, bins=30)
    plt.title(f"Distribución de {col} en layer 3")
    plt.show()
    
    plt.figure()
    sns.boxplot(y=layer_3[col])
    plt.title(f"Boxplot de {col} en layer 3")
    plt.show()
    
    # Z-score para detectar outliers
    layer_3[f"{col}_zscore"] = np.abs(zscore(layer_3[col]))
    print(f"Valores atípicos detectados en {col} (Z-score > 3) en layer 3:")
    print(layer_3[layer_3[f"{col}_zscore"] > 3][[col]])
    
plt.figure(figsize=(10, 5))
for col in layer_5.select_dtypes(include=np.number).columns:
    plt.figure()
    sns.histplot(layer_5[col], kde=True, bins=30)
    plt.title(f"Distribución de {col} en layer 5")
    plt.show()
    
    plt.figure()
    sns.boxplot(y=layer_5[col])
    plt.title(f"Boxplot de {col} en layer 5")
    plt.show()
    
    # Z-score para detectar outliers
    layer_5[f"{col}_zscore"] = np.abs(zscore(layer_5[col]))
    print(f"Valores atípicos detectados en {col} (Z-score > 3) en layer 5:")
    print(layer_5[layer_5[f"{col}_zscore"] > 3][[col]])
    
# ---- 1.3. Análisis Temporal y Tendencias ----
if "user_ts" in layer_3.columns:
    layer_3["user_ts"] = pd.to_datetime(layer_3["user_ts"],format='%Y-%m-%d %H:%M:%S%z', errors='coerce')
    layer_3 = layer_3.dropna(subset=['user_ts']) 
    layer_3 = layer_3.sort_values(by="user_ts")
    for col in layer_3.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(10, 4))
        plt.plot(layer_3["user_ts"], layer_3[col], label=col)
        plt.title(f"Serie de tiempo de {col} en layer 3")
        plt.legend()
        plt.show()

    # Rolling Mean & Std
    for col in layer_3.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(10, 4))
        plt.plot(layer_3["user_ts"], layer_3[col].rolling(window=30).mean(), label="Rolling Mean")
        plt.plot(layer_3["user_ts"], layer_3[col].rolling(window=30).std(), label="Rolling Std")
        plt.title(f"Rolling Mean & Std de {col} en layer 3")
        plt.legend()
        plt.show()
        
if "user_ts" in layer_5.columns:
    layer_5["user_ts"] = pd.to_datetime(layer_5["user_ts"], format="mixed",errors="coerce")
    layer_5 = layer_5.dropna(subset=['user_ts']) 
    layer_5 = layer_5.sort_values(by="user_ts")
    for col in layer_5.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(10, 4))
        plt.plot(layer_5["user_ts"], layer_5[col], label=col)
        plt.title(f"Serie de tiempo de {col} en layer 5")
        plt.legend()
        plt.show()

    # Rolling Mean & Std
    for col in layer_5.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(10, 4))
        plt.plot(layer_5["user_ts"], layer_5[col].rolling(window=30).mean(), label="Rolling Mean")
        plt.plot(layer_5["user_ts"], layer_5[col].rolling(window=30).std(), label="Rolling Std")
        plt.title(f"Rolling Mean & Std de {col} en layer 5")
        plt.legend()
        plt.show()
        
# ---- 1.4. Detección de Anomalías Basada en Modelos ----
X = layer_3.select_dtypes(include=np.number)  # Solo variables numéricas
if not X.empty:
    
    # Regresión lineal
    if X.shape[1] > 1:
        reg = LinearRegression()
        reg.fit(X.iloc[:, :-1], X.iloc[:, -1])
        print("Coeficientes de la regresión lineal layer 3:")
        print(reg.coef_)
    
    # DBSCAN para detección de anomalías
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    layer_3["dbscan_anomaly"] = dbscan.fit_predict(X)
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    layer_3["isolation_anomaly"] = iso_forest.fit_predict(X)
    
    # LOF (Local Outlier Factor)
    lof = LocalOutlierFactor(n_neighbors=20)
    layer_3["lof_anomaly"] = lof.fit_predict(X)
    
    # ---- Visualizar anomalías ----
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1] if X.shape[1] > 1 else X.iloc[:, 0], hue=layer_3["isolation_anomaly"], palette={1:"blue", -1:"red"})
    plt.title("Anomalías detectadas con Isolation Forest de layer 3")
    plt.legend(title="Anomalía Layer 3", labels=["Normal", "Anómalo"])
    plt.show()
    
    print("\nCantidad de anomalías detectadas por cada método en layer 3:")
    print("Isolation Forest:")
    print(layer_3["isolation_anomaly"].value_counts())
    print("DBSCAN:")
    print(layer_3["dbscan_anomaly"].value_counts())
    print("Local Outlier Factor (LOF):")
    print(layer_3["lof_anomaly"].value_counts())
    
       
from sklearn.cluster import MiniBatchKMeans

# Seleccionar solo variables numéricas
X = layer_5.select_dtypes(include=np.number)  # Solo variables numéricas

# Reducir muestra si el dataset es muy grande
sample_size = min(50000, len(X))  # Tomar máximo 50,000 filas
X_sample = X.sample(n=sample_size, random_state=42)

if not X_sample.empty:
    # Regresión lineal
    if X_sample.shape[1] > 1:
        reg = LinearRegression()
        reg.fit(X_sample.iloc[:, :-1], X_sample.iloc[:, -1])
        print("Coeficientes de la regresión lineal layer 5:")
        print(reg.coef_)
    
    # MiniBatchKMeans para detección de clusters
    kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=10000)
    layer_5.loc[X_sample.index, "kmeans_cluster"] = kmeans.fit_predict(X_sample)
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    layer_5.loc[X_sample.index, "isolation_anomaly"] = iso_forest.fit_predict(X_sample)
    
    # LOF (Local Outlier Factor)
    lof = LocalOutlierFactor(n_neighbors=20)
    layer_5.loc[X_sample.index, "lof_anomaly"] = lof.fit_predict(X_sample)
    
    # ---- Visualizar anomalías ----
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X_sample.iloc[:, 0], y=X_sample.iloc[:, 1] if X_sample.shape[1] > 1 else X_sample.iloc[:, 0], hue=layer_5.loc[X_sample.index, "isolation_anomaly"], palette={1:"blue", -1:"red"})
    plt.title("Anomalías detectadas con Isolation Forest de layer 5 (Muestra Reducida)")
    plt.legend(title="Anomalía Layer 5", labels=["Normal", "Anómalo"])
    plt.show()
    
    print("\nCantidad de anomalías detectadas por cada método en layer 5 (Muestra Reducida):")
    print("Isolation Forest:")
    print(layer_5.loc[X_sample.index, "isolation_anomaly"].value_counts())
    print("MiniBatchKMeans Clusters:")
    print(layer_5.loc[X_sample.index, "kmeans_cluster"].value_counts())
    print("Local Outlier Factor (LOF):")
    print(layer_5.loc[X_sample.index, "lof_anomaly"].value_counts())
