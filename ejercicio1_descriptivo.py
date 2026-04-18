import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

def cargar_dataset(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga el dataset desde un archivo CSV.
    
    Parámetros:
    ruta_archivo (str): Ruta al archivo.
    
    Retorna:
    pd.DataFrame: El dataset original.
    """
    return pd.read_csv(ruta_archivo)

def limpiar_y_tratar_dataset(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las decisiones de tratamiento de nulos y eliminación de columnas.
    
    Parámetros:
    df_original (pd.DataFrame): El dataset tal cual se cargó.
    
    Retorna:
    pd.DataFrame: El dataset procesado y limpio.
    """
    # Creamos una copia para no modificar el original por referencia
    df = df_original.copy()
    
    # DECISIÓN: Eliminar identificadores y fechas sin valor predictivo [cite: 101]
    columnas_a_eliminar = ['id', 'name', 'host_name', 'host_id', 'last_review']
    df = df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns], errors='ignore')
    
    # DECISIÓN: Imputación de nulos en reseñas [cite: 72]
    # Si no hay reseñas por mes, asumimos que es 0.
    if 'reviews_per_month' in df.columns:
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
        
    # DECISIÓN: Limpieza final de nulos residuales
    df = df.dropna()
    
    return df

def analizar_estructura(df: pd.DataFrame, titulo: str):
    """
    Muestra el resumen de la estructura del DataFrame.
    """
    print(f"\n--- {titulo} ---")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")
    
    uso_memoria = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Uso de memoria: {uso_memoria:.2f} MB")
    
    print("\nTipos de datos (dtypes):") 
    print(df.dtypes)
    
    print("\nPorcentaje de valores nulos:") 
    print((df.isnull().sum() / len(df)) * 100)

    import os

def calcular_estadisticos_descriptivos(df: pd.DataFrame, columna_objetivo: str, carpeta_salida: str):
    """
    Calcula los estadísticos descriptivos de las variables numéricas y los guarda en CSV.
    Calcula IQR, asimetría y curtosis para la variable objetivo.
    """
    print("\n--- B) ESTADÍSTICOS DESCRIPTIVOS ---")
    
    # Filtramos solo las columnas numéricas
    df_numerico = df.select_dtypes(include=[np.number])
    
    # 1. Calculamos los estadísticos básicos con describe()
    # Esto incluye: media, std, min, max y cuartiles (25%, 50%, 75%) 
    estadisticos = df_numerico.describe()
    
    # 2. Añadimos manualmente mediana, varianza y moda (que no vienen en el describe básico) 
    # La mediana es el percentil 50, pero la añadimos como fila extra para claridad
    estadisticos.loc['mediana'] = df_numerico.median()
    estadisticos.loc['varianza'] = df_numerico.var()
    # Para la moda, como puede haber varias, tomamos la primera
    estadisticos.loc['moda'] = df_numerico.mode().iloc[0]
    
    # 3. Guardar la tabla resultante en la carpeta output [cite: 92, 205]
    ruta_csv = os.path.join(carpeta_salida, "ej1_descriptivo.csv")
    estadisticos.to_csv(ruta_csv)
    print(f"Tabla de estadísticos guardada en: {ruta_csv}")
    
    # 4. Cálculos específicos para la Variable Objetivo [cite: 74, 75]
    q1 = df[columna_objetivo].quantile(0.25)
    q3 = df[columna_objetivo].quantile(0.75)
    iqr = q3 - q1
    asimetria = df[columna_objetivo].skew()
    curtosis = df[columna_objetivo].kurt()
    
    print(f"\nEstadísticos específicos para '{columna_objetivo}':")
    print(f"- Rango Intercuartílico (IQR): {iqr:.2f}")
    print(f"- Coeficiente de Asimetría (Skewness): {asimetria:.2f}")
    print(f"- Curtosis: {curtosis:.2f}")

def tratar_outliers_iqr(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """
    Detecta y elimina outliers utilizando el método IQR para la variable objetivo.
    
    Parámetros:
    df (pd.DataFrame): El dataset.
    columna (str): La columna a limpiar (ej. 'price').
    
    Retorna:
    pd.DataFrame: El dataset sin los valores atípicos.
    """
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    # CORRECCIÓN LÓGICA: Un precio no puede ser inferior a 0 (o a un mínimo razonable)
    # Si el cálculo da negativo, lo fijamos en el mínimo real del dataset que sea > 0
    if limite_inferior < 0:
        limite_inferior = df[df[columna] > 0][columna].min()

    # Filtramos los datos
    df_filtrado = df[(df[columna] >= limite_inferior) & (df[columna] <= limite_superior)]
    
    filas_eliminadas = len(df) - len(df_filtrado)
    print(f"\n--- TRATAMIENTO DE OUTLIERS (MÉTODO IQR) ---")
    print(f"Columna: {columna}")
    print(f"Límites: [{limite_inferior}, {limite_superior}]")
    print(f"Valores eliminados: {filas_eliminadas}")
    
    return df_filtrado

def generar_graficos_distribucion(df: pd.DataFrame, columna_target: str, columnas_cat: list, carpeta_salida: str):
    """
    Genera y guarda histogramas y boxplots optimizados para rapidez.
    """
    # 1. Histogramas: Seleccionamos solo variables con valor estadístico real
    # Excluimos latitud y longitud porque no son distribuciones útiles para histograma
    cols_interes = [columna_target, 'minimum_nights', 'number_of_reviews', 'availability_365', 'reviews_per_month']
    columnas_num = [c for c in cols_interes if c in df.columns]
    
    print(f"Generando histogramas para: {columnas_num}...")
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    axes = axes.flatten()

    for i, col in enumerate(columnas_num):
        # Desactivamos kde=True para ganar mucha velocidad
        sns.histplot(df[col], kde=False, ax=axes[i], color='skyblue', bins=30)
        axes[i].set_title(f'Distribución de {col}')
    
    # Limpiamos subplots vacíos si los hay
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "ej1_histogramas.png"))
    plt.close()
    print("Histogramas guardados.")

    # 2. Boxplots de la variable objetivo
    print(f"Generando boxplots segmentados por {columnas_cat}...")
    plt.figure(figsize=(14, 6))
    for i, cat in enumerate(columnas_cat, 1):
        plt.subplot(1, len(columnas_cat), i)
        # showfliers=False hace que el gráfico se genere instantáneamente
        sns.boxplot(x=cat, y=columna_target, data=df, hue=cat, palette='Set2', legend=False, showfliers=False)
        plt.xticks(rotation=45)
        plt.title(f'{columna_target} por {cat} (sin outliers)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(carpeta_salida, "ej1_boxplots.png"))
    plt.close()
    print(f"Boxplots guardados en: {carpeta_salida}")

def analizar_variables_categoricas(df: pd.DataFrame, columnas_cat: list, carpeta_salida: str):
    """
    Calcula frecuencias y genera gráficos de barras para variables categóricas.
    Identifica si existe desbalanceo en las categorías.
    """
    print("\n--- D) ANÁLISIS DE VARIABLES CATEGÓRICAS ---")
    
    for cat in columnas_cat:
        if cat in df.columns:
            # 1. Cálculo de Frecuencias
            abs_freq = df[cat].value_counts()
            rel_freq = df[cat].value_counts(normalize=True) * 100
            
            # Crear una tabla resumen para imprimir
            resumen = pd.DataFrame({
                'Frecuencia Absoluta': abs_freq,
                'Frecuencia Relativa (%)': rel_freq
            })
            
            print(f"\nResumen para la variable: {cat}")
            print(resumen)
            
            # 2. Análisis de desbalanceo (Regla simple: si una categoría supera el 50%)
            dominante = rel_freq.idxmax()
            porcentaje_dom = rel_freq.max()
            if porcentaje_dom > 50:
                print(f"AVISO: La categoría '{dominante}' domina el dataset ({porcentaje_dom:.2f}%).")
            else:
                print(f"La variable '{cat}' está relativamente balanceada.")

            # 3. Gráfico de Barras
            plt.figure(figsize=(10, 6))
            sns.countplot(x=cat, data=df, hue=cat, palette='viridis', legend=False)
            plt.title(f'Distribución de frecuencias: {cat}')
            plt.xticks(rotation=45)
            plt.ylabel('Cantidad de alojamientos')
            
            nombre_grafico = f"ej1_barras_{cat}.png"
            plt.savefig(os.path.join(carpeta_salida, nombre_grafico))
            plt.close()
            print(f"Gráfico guardado como: {nombre_grafico}")

def analizar_correlaciones(df: pd.DataFrame, columna_target: str, carpeta_salida: str):
    """
    Genera un mapa de calor de Pearson y detecta multicolinealidad.
    """
    print("\n--- E) ANÁLISIS DE CORRELACIONES ---")
    
    # 1. Filtrar solo variables numéricas y calcular matriz de Pearson
    df_numerico = df.select_dtypes(include=[np.number])
    matriz_corr = df_numerico.corr(method='pearson')
    
    # 2. Generar el Mapa de Calor (Heatmap)
    plt.figure(figsize=(12, 10))
    # Usamos un mapa de colores divergente (coolwarm) para ver bien positivos y negativos
    sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Matriz de Correlación de Pearson')
    
    ruta_heatmap = os.path.join(carpeta_salida, "ej1_heatmap_correlacion.png")
    plt.savefig(ruta_heatmap)
    plt.close()
    print(f"Mapa de calor guardado en: {ruta_heatmap}")
    
    # 3. Identificar las 3 variables con mayor correlación absoluta con el target
    correlacion_target = matriz_corr[columna_target].drop(columna_target).abs().sort_values(ascending=False)
    top_3 = correlacion_target.head(3)
    
    print(f"\nTop 3 variables con mayor correlación absoluta con '{columna_target}':")
    for var, val in top_3.items():
        # Recuperamos el valor real (con signo) de la matriz original
        valor_real = matriz_corr.loc[var, columna_target]
        print(f"- {var}: {valor_real:.4f}")
        
    # 4. Detección de Multicolinealidad (|r| > 0.9)
    print("\nComprobación de Multicolinealidad (Pares con |r| > 0.9):")
    pares_altos = []
    # Iteramos solo por la mitad de la matriz para no repetir pares
    for i in range(len(matriz_corr.columns)):
        for j in range(i):
            coef = matriz_corr.iloc[i, j]
            if abs(coef) > 0.9:
                v1 = matriz_corr.columns[i]
                v2 = matriz_corr.columns[j]
                pares_altos.append(f"{v1} y {v2} (r = {coef:.2f})")
    
    if pares_altos:
        for p in pares_altos:
            print(f"ALERTA: {p}")
    else:
        print("No se ha detectado multicolinealidad extrema entre las variables predictoras.")



if __name__ == "__main__":
    np.random.seed(42)
    
    path = "data/AB_NYC_2019.csv"
    
    # Fase 1: Carga pura
    df_raw = cargar_dataset(path)
    analizar_estructura(df_raw, "ESTRUCTURA ORIGINAL (DATOS ORIGINALES)")
    
    # Fase 2: Limpieza y Tratamiento
    df_limpio = limpiar_y_tratar_dataset(df_raw)
    analizar_estructura(df_limpio, "ESTRUCTURA FINAL (DATOS PROCESADOS)")

    # Fase 3: Estadísticos Descriptivos
    calcular_estadisticos_descriptivos(df_limpio, columna_objetivo='price', carpeta_salida='output')

    # Fase 4: Tratamiento de Outliers
    df_sin_outliers = tratar_outliers_iqr(df_limpio, columna='price')

    # Fase 5: Gráficos de Distribución
    generar_graficos_distribucion(df_sin_outliers, columna_target='price', columnas_cat=['neighbourhood_group', 'room_type'], carpeta_salida='output')

    # Fase 6: Variables Categóricas
    analizar_variables_categoricas(df_limpio, ["neighbourhood_group", "room_type"], "output")

    # Fase 7: Correlaciones
    analizar_correlaciones(df_limpio, "price", "output")