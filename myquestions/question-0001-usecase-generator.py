import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generar_caso_de_uso_crear_reporte_ventas_pivot(n_filas=15):
    """
    Genera un caso de prueba dinámico para evaluar la función crear_reporte_ventas_pivot.
    
    Args:
        n_filas (int): Número de transacciones aleatorias a generar.
        
    Returns:
        tuple: Un par (input_dict, output_esperado_df)
            - input_dict: Diccionario con los argumentos para la función (ej. {"df": df_input}).
            - output_esperado_df: DataFrame con la tabla dinámica correctamente calculada.
    """
    # ---------------------------------------------------------
    # 1. GENERAR EL INPUT (Datos aleatorios)
    # ---------------------------------------------------------
    sucursales_posibles = ['Norte', 'Sur', 'Centro', 'Este', 'Oeste']
    categorias_posibles = ['Lácteos', 'Carnes', 'Bebidas', 'Limpieza', 'Panadería', 'Verduras']
    
    # Generar fechas aleatorias
    fecha_base = datetime(2024, 1, 1)
    fechas = [fecha_base + timedelta(days=random.randint(0, 10)) for _ in range(n_filas)]
    
    datos = {
        'fecha': fechas,
        'sucursal': np.random.choice(sucursales_posibles, size=n_filas),
        'categoria_producto': np.random.choice(categorias_posibles, size=n_filas),
        'ventas_totales': np.round(np.random.uniform(10.0, 500.0, size=n_filas), 2)
    }
    
    df_input = pd.DataFrame(datos)
    df_input['fecha'] = pd.to_datetime(df_input['fecha'])
    
    # ---------------------------------------------------------
    # 2. GENERAR EL OUTPUT ESPERADO (La solución correcta)
    # ---------------------------------------------------------
    # Utilizamos la lógica correcta de Pandas para resolver el problema
    # y así tener contra qué comparar el código del estudiante.
    df_output_esperado = pd.pivot_table(
        df_input, 
        values='ventas_totales', 
        index='sucursal', 
        columns='categoria_producto', 
        aggfunc='sum',     # Suma de ventas
        fill_value=0       # Relleno de nulos con 0
    )
    
    # ---------------------------------------------------------
    # 3. RETORNAR EL PAR INPUT / OUTPUT (CORREGIDO AQUÍ)
    # ---------------------------------------------------------
    # El validador exige que el input sea un diccionario de argumentos
    return {"df": df_input}, df_output_esperado


if __name__ == "__main__":
    # Generamos un caso
    entrada, salida_esperada = generar_caso_de_uso_crear_reporte_ventas_pivot()
    print("--- INPUT PARA CASO DE USO (Diccionario) ---\n")
    print(entrada)
    print("\n")
    print("--- OUTPUT ESPERADO FUNCIÓN crear_reporte_ventas_pivot ---\n")
    print(salida_esperada)
