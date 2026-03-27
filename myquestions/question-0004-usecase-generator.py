import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def generar_caso_de_uso_agrupar_zonas_entrega_dbscan(n_puntos_densos=200, n_puntos_ruido=30):
    """
    Genera un caso de prueba dinámico para evaluar la función agrupar_zonas_entrega_dbscan.
    
    Args:
        n_puntos_densos (int): Cantidad de puntos que formarán agrupaciones lógicas (entregas comunes).
        n_puntos_ruido (int): Cantidad de puntos anómalos esparcidos aleatoriamente (ruido).
        
    Returns:
        tuple: Un par (input_args, output_esperado)
            - input_args: Una tupla (X, epsilon, min_muestras) lista para desempaquetar.
            - output_esperado: Un array de numpy con las etiquetas (clústeres y -1) correctas.
    """
    # ---------------------------------------------------------
    # 1. GENERAR EL INPUT (Datos aleatorios espaciales)
    # ---------------------------------------------------------
    # a. Generar clústeres densos simulando zonas de alta demanda (ej. vecindarios)
    # No usamos random_state para que los centros cambien en cada ejecución
    X_densos, _ = make_blobs(
        n_samples=n_puntos_densos, 
        centers=np.random.randint(3, 6), # Entre 3 y 5 zonas de alta densidad
        cluster_std=0.4,                 # Qué tan juntos están los puntos
        n_features=2                     # Latitud y longitud
    )
    
    # b. Generar puntos de ruido simulando entregas muy lejanas/aisladas
    # Usamos los límites máximos y mínimos de los datos densos para esparcirlos por toda el área
    min_coords = np.min(X_densos, axis=0) - 1.0
    max_coords = np.max(X_densos, axis=0) + 1.0
    
    X_ruido = np.random.uniform(
        low=min_coords, 
        high=max_coords, 
        size=(n_puntos_ruido, 2)
    )
    
    # c. Unir y mezclar los datos para que el ruido no quede solo al final del array
    X = np.vstack([X_densos, X_ruido])
    np.random.shuffle(X)
    
    # d. Definir los parámetros de DBSCAN para este caso de uso
    epsilon_val = np.random.uniform(0.25, 0.4) # Valor razonable para datos escalados
    min_muestras_val = np.random.randint(4, 7)
    
    # Empaquetamos los argumentos de entrada
    input_args = (X, epsilon_val, min_muestras_val)
    
    # ---------------------------------------------------------
    # 2. GENERAR EL OUTPUT ESPERADO (La solución correcta)
    # ---------------------------------------------------------
    # Simulamos el pipeline exacto que debe hacer el estudiante
    
    # a. Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # b. Instanciar y ajustar DBSCAN
    dbscan = DBSCAN(eps=epsilon_val, min_samples=min_muestras_val)
    
    # c. Extraer etiquetas (puede usar fit_predict o fit().labels_)
    etiquetas_esperadas = dbscan.fit_predict(X_scaled)
    
    # ---------------------------------------------------------
    # 3. RETORNAR EL PAR INPUT / OUTPUT
    # ---------------------------------------------------------
    return input_args, etiquetas_esperadas

if __name__ == "__main__":
    # Generamos un caso
    inputs, etiquetas_esperadas = generar_caso_de_uso_agrupar_zonas_entrega_dbscan(150, 25)
    X_in, eps_in, min_samples_in = inputs

    print(f"Forma de X (Lat/Lon): {X_in.shape}")
    print(f"Epsilon generado: {eps_in:.3f}")
    print(f"Min_muestras generado: {min_samples_in}")
    print(f"Primeras 10 etiquetas esperadas: {etiquetas_esperadas[:10]}")
    print(f"¿Detectó ruido (-1)?: {-1 in etiquetas_esperadas}")
