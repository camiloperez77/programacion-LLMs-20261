import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans

def isomap_kmeans_clustering(X, n_neighbors_isomap, n_clusters):
    """
    Escala los datos, aplica reducción no lineal (Isomap) a 2 dimensiones,
    y luego agrupa los puntos usando KMeans.
    """
    # 1. Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Reducción de dimensionalidad con Isomap (a 2 componentes)
    isomap = Isomap(n_neighbors=n_neighbors_isomap, n_components=2)
    X_iso = isomap.fit_transform(X_scaled)
    
    # 3. Clustering con KMeans sobre las 2 componentes principales
    # Se fija random_state=42 y n_init=10 para garantizar reproducibilidad exacta
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_iso)
    
    # 4. Resultado: Coordenadas reducidas y etiquetas de los clústeres
    return X_iso, labels
