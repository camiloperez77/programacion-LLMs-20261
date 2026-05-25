import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

def seleccionar_mejores_caracteristicas(df, target_col, k=3):
    """
    Selecciona las k mejores características numéricas que tienen 
    mayor relación con el target usando f_regression.
    """
    # 1. Preparación: Separar características (X) y objetivo (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Limpieza: Rellenar nulos en X con 0
    X_filled = X.fillna(0)
    
    # 3. Selección: Aplicar SelectKBest
    selector = SelectKBest(score_func=f_regression, k=k)
    
    # 4. Transformación: Obtener la matriz reducida
    X_new = selector.fit_transform(X_filled, y)
    
    # Obtener la lista de nombres de las columnas que fueron elegidas
    cols_seleccionadas = X.columns[selector.get_support()].tolist()
    
    # 5. Resultado
    return X_new, cols_seleccionadas
