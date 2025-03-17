import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Rutas de los archivos CSV 
csv_files = {
    "Layer_1": "ruta_a_layer_1.csv",
    "Layer_3": "ruta_a_layer_3.csv",
    "Layer_5": "ruta_a_layer_5.csv",
    "Layer_7": "ruta_a_layer_7.csv",
    "Layer_9": "ruta_a_layer_9.csv"
}

# Cargar cada CSV y unirlos en un solo DataFrame por `user_ts`
df_layers = []
for layer, file in csv_files.items():
    df_temp = pd.read_csv(file)
    df_temp["user_ts"] = pd.to_datetime(df_temp["user_ts"])  # Convertir timestamps
    df_temp = df_temp.rename(columns={"value": layer})  # Renombrar la columna de temperatura
    df_layers.append(df_temp[["user_ts", layer]])  # Conservar solo `user_ts` y temperatura

# Unir los datasets por `user_ts`
df = df_layers[0]
for i in range(1, len(df_layers)):
    df = pd.merge(df, df_layers[i], on="user_ts", how="inner")

# Boxplots para visualizar el rango de temperaturas en cada capa
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.drop(columns=["user_ts"]))
plt.title("Distribución de Temperaturas en las Diferentes Capas")
plt.ylabel("Temperatura (°C)")
plt.xticks(rotation=45)
plt.show()

# Calcular estadísticas clave
stats = df.drop(columns=["user_ts"]).describe(percentiles=[0.25, 0.5, 0.75])
print("📌 Estadísticas de las temperaturas por capa:")
print(stats)

# Comparar temperaturas antes del reemplazo de la sopladora
if "reemplazo_sopladora" in df.columns:
    plt.figure(figsize=(12, 6))
    for layer in csv_files.keys():
        sns.lineplot(data=df, x="user_ts", y=layer, label=layer, alpha=0.7)
    plt.axvline(df[df["reemplazo_sopladora"] == 1]["user_ts"].min(), color="red", linestyle="dashed", label="Reemplazo")
    plt.title("Temperaturas antes del reemplazo de la sopladora")
    plt.xlabel("Fecha")
    plt.ylabel("Temperatura (°C)")
    plt.legend()
    plt.show()

# Correlación entre temperaturas y frecuencia de mantenimiento
if "mantenimientos" in df.columns:
    corr_matrix = df.drop(columns=["user_ts"]).corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlación entre Temperaturas y Mantenimiento")
    plt.show()
