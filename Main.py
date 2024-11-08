import numpy as np

def gauss_elimination(a_matrix, b_matrix):
    # Evitando problemas
    if a_matrix.shape[0] != a_matrix.shape[1]:
        print("ERROR: La matriz no es cuadrada")
        return
    if b_matrix.shape[1] > 1 or b_matrix.shape[0] != a_matrix.shape[0]:
        print("El vector constante tiene tamaño incorrecto")
        return

    # Inicialización de variables
    n = len(b_matrix)
    m = n - 1
    i = 0
    x = np.zeros(n)
    new_line = "\n"

    # Creando la matriz ampliada usando numpy.concatenate
    augmented_matrix = np.concatenate((a_matrix, b_matrix), axis=1, dtype=float)
    print(f"La matriz ampliada inicial es: {new_line}{augmented_matrix}")
    print()
    print("Llevando a una forma triangular superior:")

    # Aplicando escalerización Gaussiana:
    while i < n:
        # Pivoteo parcial
        for p in range(i + 1, n):
            if abs(augmented_matrix[i, i]) < abs(augmented_matrix[p, i]):
                augmented_matrix[[p, i]] = augmented_matrix[[i, p]]
                
        # Error por dividir entre cero
        if augmented_matrix[i][i] == 0.0:
            print("Error por dividir entre cero")
            return
        
        for j in range(i + 1, n):
            scaling_factor = augmented_matrix[j][i] / augmented_matrix[i][i]
            augmented_matrix[j] = augmented_matrix[j] - (scaling_factor * augmented_matrix[i])
        
        # Para ver el proceso
        print(augmented_matrix)
        print()
        i += 1

    # Sustitución hacia atrás
    if augmented_matrix[m][m] == 0.0:
        print("Error: el sistema no es compatible determinado")
        return

    x[m] = augmented_matrix[m][n] / augmented_matrix[m][m]
    for k in range(n - 2, -1, -1):
        x[k] = augmented_matrix[k][n]
        for j in range(k + 1, n):
            x[k] = x[k] - augmented_matrix[k][j] * x[j]
        
        if augmented_matrix[k][k] == 0.0:
            print("Error: el sistema no es compatible determinado")
            return
        x[k] = x[k] / augmented_matrix[k][k]

    # Imprimiendo la solución
    print("El siguiente vector x resuelve el sistema:")
    for answer in range(n):
        print(f"x{answer + 1} is {x[answer]}")

# Para probar el código, ingresar aquí la matriz A del sistema y el vector constante b
"""if __name__ == '__main__':
    variable_matrix = np.array([[1, 1, 3], [0, 1, 3], [-1, 3, 0]])
    constant_matrix = np.array([[1], [3], [5]])
    gauss_elimination(variable_matrix, constant_matrix)"""

#Aca arranca nuestro codigo para la solucion del Juego Lights Out

def lights_out_solution(initial_state):
    # Dimensiones de la matriz (debe ser cuadrada nxn)
    n = initial_state.shape[0]
    
    # Construyendo la matriz ampliada del sistema binario
    augmented_matrix = np.zeros((n * n, n * n + 1), dtype=int)
    
    # Configurando la matriz A para Lights Out
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            augmented_matrix[idx, idx] = 1  # La luz misma
            # Luces adyacentes en la cuadrícula
            if i > 0:
                augmented_matrix[idx, (i - 1) * n + j] = 1  # arriba
            if i < n - 1:
                augmented_matrix[idx, (i + 1) * n + j] = 1  # abajo
            if j > 0:
                augmented_matrix[idx, i * n + (j - 1)] = 1  # izquierda
            if j < n - 1:
                augmented_matrix[idx, i * n + (j + 1)] = 1  # derecha
    
    # Agregar la columna de resultados inicial (convertimos la matriz inicial en un vector columna)
    augmented_matrix[:, -1] = initial_state.flatten()
    
    # Aplicar la eliminación Gaussiana en el sistema binario
    for i in range(n * n):
        # Comprobar si el elemento en la diagonal es 0 y realizar pivoteo binario si es necesario
        if augmented_matrix[i, i] == 0:
            for k in range(i + 1, n * n):
                if augmented_matrix[k, i] == 1:
                    augmented_matrix[[i, k]] = augmented_matrix[[k, i]]
                    break

        # Hacemos cero en todas las posiciones de la columna i de abajo hacia arriba y arriba hacia abajo
        for j in range(i + 1, n * n):
            if augmented_matrix[j, i] == 1:
                augmented_matrix[j] = (augmented_matrix[j] + augmented_matrix[i]) % 2  # Suma binaria (XOR)
    
    # Sustitución hacia atrás para encontrar la solución
    solution = np.zeros(n * n, dtype=int)
    for i in range(n * n - 1, -1, -1):
        solution[i] = augmented_matrix[i, -1]
        for j in range(i + 1, n * n):
            solution[i] ^= (augmented_matrix[i, j] * solution[j])  # Usar XOR para evitar resta
    
    # Devolver el vector de solución en formato n x n
    return solution.reshape((n, n))

# Ejemplo de uso
initial_state = np.array([
    [1, 0, 0,1],
    [1, 1, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 1, 1]
])
print("Estado inicial del Juego:")
print(initial_state)
solution = lights_out_solution(initial_state)
print("Solución del juego Lights Out:")
print(solution)