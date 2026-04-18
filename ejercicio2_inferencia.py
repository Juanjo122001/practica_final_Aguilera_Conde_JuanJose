import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def regresion_ejercicio2(df_limpio: pd.DataFrame, target: str, carpeta_out: str):
    print("\n=== INICIANDO EJERCICIO 2: REGRESIÓN LINEAL ===")
    
    # 1. ELIMINAR COLUMNAS DE TEXTO SOBRANTES
    # 'neighbourhood' tiene cientos de valores, ya usamos 'neighbourhood_group' para simplificar
    cols_a_quitar = ['neighbourhood'] 
    df_pre = df_limpio.drop(columns=[c for c in cols_a_quitar if c in df_limpio.columns], errors='ignore')

    # 2. CODIFICACIÓN (One-Hot Encoding)
    # Convertimos a números las categorías que sí queremos (distrito y tipo de habitación)
    df_pre = pd.get_dummies(df_pre, columns=['neighbourhood_group', 'room_type'], drop_first=True)
    
    # 3. SEPARAR X e Y
    X = df_pre.drop(columns=[target])
    y = df_pre[target]
    
    # Ahora X solo tiene números (bools de los dummies y valores numéricos)
    # El resto del código debería funcionar perfectamente
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- 2.2 MODELO A: REGRESIÓN LINEAL ---
    
    # 1. Entrenamiento
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 2. Predicción
    y_pred = model.predict(X_test_scaled)
    
    # 3. Evaluación
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # 4. Guardar métricas en TXT
    ruta_txt = os.path.join(carpeta_out, "ej2_metricas_regresion.txt")
    with open(ruta_txt, "w") as f:
        f.write("MÉTRICAS DEL MODELO DE REGRESIÓN LINEAL\n")
        f.write("========================================\n")
        f.write(f"MAE (Error Absoluto Medio): {mae:.4f}\n")
        f.write(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f}\n")
        f.write(f"R2 (Coeficiente de Determinación): {r2:.4f}\n")
    
    print(f"Métricas guardadas en: {ruta_txt}")
    
    # 5. Gráfico de Residuos
    residuos = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuos, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Gráfico de Residuos (Valores Predichos vs Residuos)')
    plt.xlabel('Predicciones ($)')
    plt.ylabel('Residuos ($)')
    
    ruta_residuos = os.path.join(carpeta_out, "ej2_residuos.png")
    plt.savefig(ruta_residuos)
    plt.close()
    print(f"Gráfico de residuos guardado en: {ruta_residuos}")

    # Extra para el análisis: Coeficientes más influyentes
    coef_df = pd.DataFrame({'Variable': X.columns, 'Coeficiente': model.coef_})
    coef_df = coef_df.sort_values(by='Coeficiente', ascending=False)
    print("\nVariables más influyentes según el modelo:")
    print(coef_df.head(3)) # Las que más suben el precio
    print(coef_df.tail(3)) # Las que más ban el precio