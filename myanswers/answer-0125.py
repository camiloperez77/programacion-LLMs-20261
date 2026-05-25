import pandas as pd
import numpy as np

def limpiar_dataset_ventas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia un DataFrame de ventas: imputa nulos, elimina duplicados,
    normaliza strings y convierte fechas.
    """
    # Se trabaja sobre una copia para evitar SettingWithCopyWarning
    df_out = df.copy()

    # 1. Eliminar filas duplicadas
    df_out = df_out.drop_duplicates()

    # 2. Imputar valores nulos numéricos con la mediana de cada columna
    cols_num = df_out.select_dtypes(include=[np.number]).columns
    for col in cols_num:
        median = df_out[col].median()
        df_out[col] = df_out[col].fillna(median)

    # 3. Normalizar strings: minúsculas y sin espacios extra (ignorando 'fecha')
    cols_obj = df_out.select_dtypes(include=["object"]).columns
    for col in cols_obj:
        if col != "fecha":
            df_out[col] = df_out[col].str.strip().str.lower()

    # 4. Convertir fecha a datetime
    if "fecha" in df_out.columns:
        df_out["fecha"] = pd.to_datetime(df_out["fecha"])

    return df_out
