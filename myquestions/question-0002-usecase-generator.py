import pandas as pd
import random

def generar_caso_de_uso_procesar_info_contacto(n_filas=15):
    """
    Genera un caso de prueba dinámico para evaluar la función procesar_info_contacto.
    
    Args:
        n_filas (int): Número de registros de empleados aleatorios a generar.
        
    Returns:
        tuple: Un par (input_df, output_esperado_df)
            - input_df: DataFrame con la columna 'info_contacto' desordenada.
            - output_esperado_df: DataFrame con las columnas divididas y filtrado por Ventas/IT.
    """
    # ---------------------------------------------------------
    # 1. GENERAR EL INPUT (Datos aleatorios)
    # ---------------------------------------------------------
    nombres_posibles = [
        "Juan Perez", "Maria Gomez", "Carlos Ruiz", "Ana Silva", 
        "Luis Diaz", "Sofia Castro", "Miguel Rojas", "Laura Marin"
    ]
    telefonos_posibles = [
        "5551234", "5559876", "5554321", "5551111", 
        "5552222", "5553333", "5554444", "5555555"
    ]
    # Incluimos Ventas e IT, pero también otros para verificar que el estudiante filtre bien
    departamentos_posibles = ["Ventas", "IT", "Finanzas", "RRHH", "Marketing", "Operaciones"]
    
    datos_contacto = []
    ids_empleado = []
    
    for i in range(1, n_filas + 1):
        nombre = random.choice(nombres_posibles)
        telefono = random.choice(telefonos_posibles)
        depto = random.choice(departamentos_posibles)
        
        # Construimos el string con el formato "Nombre-Telefono-Departamento"
        string_contacto = f"{nombre}-{telefono}-{depto}"
        
        datos_contacto.append(string_contacto)
        ids_empleado.append(1000 + i) # Simulamos un ID de empleado
        
    df_input = pd.DataFrame({
        'id_empleado': ids_empleado,
        'info_contacto': datos_contacto
    })
    
    # ---------------------------------------------------------
    # 2. GENERAR EL OUTPUT ESPERADO (La solución correcta)
    # ---------------------------------------------------------
    # Hacemos una copia para no alterar el input original
    df_output_esperado = df_input.copy()
    
    # a. Dividir la columna por el guion '-'
    # expand=True convierte la lista resultante en columnas separadas
    df_output_esperado[['nombre', 'telefono', 'departamento']] = df_output_esperado['info_contacto'].str.split('-', expand=True)
    
    # b. Eliminar la columna original
    df_output_esperado = df_output_esperado.drop(columns=['info_contacto'])
    
    # c. Filtrar por departamento "Ventas" o "IT"
    df_output_esperado = df_output_esperado[df_output_esperado['departamento'].isin(['Ventas', 'IT'])]
    
    # ---------------------------------------------------------
    # 3. RETORNAR EL PAR INPUT / OUTPUT
    # ---------------------------------------------------------
    return df_input, df_output_esperado

if __name__ == "__main__":
    # Generamos un caso
    df_in, df_out_esperado = generar_caso_de_uso_procesar_info_contacto(20)
    print("--- INPUT ---")
    print(df_in)
    print("\n--- OUTPUT ESPERADO ---")
    print(df_out_esperado)
