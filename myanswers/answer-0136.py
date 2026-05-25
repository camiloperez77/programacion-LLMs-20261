import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

def detectar_outliers_lof(df, features=None, n_neighbors=20, contamination=0.05, test_size=0.2, random_state=42):
    """
    Detecta outliers con LocalOutlierFactor (LOF) tras imputar y escalar.
    Usa novelty=True para entrenar en train y predecir en test.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un pandas.DataFrame")

    # 1) Selección de columnas
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(features) == 0:
        raise ValueError("No hay columnas numéricas disponibles para detectar outliers.")
    
    X = df[features].copy()

    # 2) División train/test
    X_train, X_test = train_test_split(
        X, test_size=float(test_size), random_state=int(random_state)
    )

    # Ajustar n_neighbors a tamaño de train para evitar errores de Scikit-Learn
    max_valid = max(5, min(int(n_neighbors), len(X_train) - 1))
    if max_valid < 2:
        raise ValueError("Muy pocas muestras en entrenamiento para aplicar LOF.")
    n_neighbors_adj = max_valid

    # 3) Pipeline (imputación + escalado + LOF con novelty=True)
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors_adj,
        contamination=float(contamination),
        novelty=True
    )
    
    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lof", lof)
    ])

    # 4) Entrenamiento y predicción
    pipe.fit(X_train)
    y_pred = pipe.predict(X_test).astype(int)                 # {1 = normal, -1 = outlier}
    scores = pipe.decision_function(X_test).astype(float)     # mayor => más “normal”

    return y_pred, scores, pipe
