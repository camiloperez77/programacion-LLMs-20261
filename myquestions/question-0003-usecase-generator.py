import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

def generar_caso_de_uso_analizar_churn_gradient_boosting(n_muestras=200, n_caracteristicas=5):
    """
    Genera un caso de prueba dinámico para evaluar la función analizar_churn_gradient_boosting.
    
    Args:
        n_muestras (int): Número de clientes (filas) a generar.
        n_caracteristicas (int): Número de variables del servicio (columnas).
        
    Returns:
        tuple: Un par (input_dict, output_esperado)
            - input_dict: Diccionario con los argumentos para la función {"X": X, "y": y}.
            - output_esperado: Una tupla (predicciones, importancias) calculadas correctamente.
    """
    # ---------------------------------------------------------
    # 1. GENERAR EL INPUT (Datos aleatorios)
    # ---------------------------------------------------------
    # make_classification genera un problema de clasificación binaria (1 y 0)
    # Al no fijar el random_state aquí, generará un dataset distinto en cada ejecución.
    X, y = make_classification(
        n_samples=n_muestras, 
        n_features=n_caracteristicas, 
        n_informative=3,     # Cuántas características son realmente útiles para predecir
        n_redundant=1,       # Características que son combinaciones lineales de otras
        n_classes=2,         # Clasificación binaria (Churn: 1 o 0)
        weights=[0.7, 0.3]   # Desbalance simulando que el 30% hace churn
    )
    
    # ---------------------------------------------------------
    # 2. GENERAR EL OUTPUT ESPERADO (La solución correcta)
    # ---------------------------------------------------------
    # Simulamos exactamente lo que debería hacer el código del estudiante
    
    # a. División 80/20 (Fijamos random_state=42 para que la respuesta esperada sea determinista respecto al input)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # b. Instanciar y entrenar el modelo (Fijando semilla tal como pide el enunciado)
    modelo_gb = GradientBoostingClassifier(random_state=42)
    modelo_gb.fit(X_train, y_train)
    
    # c. Realizar predicciones
    predicciones_esperadas = modelo_gb.predict(X_test)
    
    # d. Extraer importancias
    importancias_esperadas = modelo_gb.feature_importances_
    
    # Empaquetamos la salida esperada como pide el enunciado (tupla de 2 arrays)
    output_esperado = (predicciones_esperadas, importancias_esperadas)
    
    # ---------------------------------------------------------
    # 3. RETORNAR EL PAR INPUT / OUTPUT (CORREGIDO AQUÍ)
    # ---------------------------------------------------------
    # Empaquetamos el input como un diccionario con las claves de los parámetros exactos
    return {"X": X, "y": y}, output_esperado


if __name__ == "__main__":
    # Generamos un caso
    inputs_dict, output_esperado = generar_caso_de_uso_analizar_churn_gradient_boosting(150, 4)
    
    # Extraemos para imprimir
    X_input = inputs_dict["X"]
    y_input = inputs_dict["y"]
    preds_esperadas, importancias_esperadas = output_esperado

    print("--- INPUT PARA CASO DE USO (Diccionario) ---")
    print(f"Claves del diccionario: {list(inputs_dict.keys())}")
    print(f"Forma de X_input: {X_input.shape}")
    print(f"Forma de y_input: {y_input.shape}")
    print("\n--- OUTPUT ESPERADO ---")
    print(f"Primeras 5 predicciones esperadas: {preds_esperadas[:5]}")
    print(f"Importancia de variables esperada: {importancias_esperadas}")
