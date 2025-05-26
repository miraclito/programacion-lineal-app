import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations
import tkinter as tk
from tkinter import ttk, messagebox

class ProgramacionLinealApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Programación Lineal - Maximizar/Minimizar")
        
        # Variables de control
        self.modo = tk.StringVar(value="min")  # "min" o "max"
        self.num_vars = tk.IntVar(value=2)
        self.num_restricciones = tk.IntVar(value=2)
        
        # Crear interfaz
        self.crear_interfaz()
    
    def crear_interfaz(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Selección de modo (Max/Min)
        ttk.Label(main_frame, text="Objetivo:").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(main_frame, text="Minimizar", variable=self.modo, value="min").grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(main_frame, text="Maximizar", variable=self.modo, value="max").grid(row=0, column=2, sticky=tk.W)
        
        # Número de variables
        ttk.Label(main_frame, text="Número de variables:").grid(row=1, column=0, sticky=tk.W)
        ttk.Spinbox(main_frame, from_=2, to=10, textvariable=self.num_vars).grid(row=1, column=1, sticky=tk.W)
        
        # Número de restricciones
        ttk.Label(main_frame, text="Número de restricciones:").grid(row=2, column=0, sticky=tk.W)
        ttk.Spinbox(main_frame, from_=1, to=20, textvariable=self.num_restricciones).grid(row=2, column=1, sticky=tk.W)
        
        # Botón para configurar problema
        ttk.Button(main_frame, text="Configurar Problema", command=self.configurar_problema).grid(row=3, column=0, columnspan=3, pady=10)
        
        # Frame para entrada de datos (se llenará dinámicamente)
        self.entrada_frame = ttk.Frame(main_frame)
        self.entrada_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Botón para resolver
        ttk.Button(main_frame, text="Resolver", command=self.resolver_problema).grid(row=5, column=0, columnspan=3, pady=10)
    
    def configurar_problema(self):
        # Limpiar frame de entrada
        for widget in self.entrada_frame.winfo_children():
            widget.destroy()
        
        n_vars = self.num_vars.get()
        n_restricciones = self.num_restricciones.get()
        
        # Función objetivo
        ttk.Label(self.entrada_frame, text="Función Objetivo (coeficientes):").grid(row=0, column=0, sticky=tk.W)
        self.coef_obj = []
        for i in range(n_vars):
            ttk.Label(self.entrada_frame, text=f"x_{i+1}:").grid(row=0, column=2*i+1, sticky=tk.W)
            entry = ttk.Entry(self.entrada_frame, width=5)
            entry.grid(row=0, column=2*i+2, sticky=tk.W)
            self.coef_obj.append(entry)
        
        # Restricciones
        ttk.Label(self.entrada_frame, text="Restricciones:").grid(row=1, column=0, sticky=tk.W)
        self.coef_restricciones = []
        self.terms_restricciones = []
        self.signos_restricciones = []  # Almacenará los signos de desigualdad
        
        for j in range(n_restricciones):
            coef_row = []
            for i in range(n_vars):
                entry = ttk.Entry(self.entrada_frame, width=5)
                entry.grid(row=j+2, column=i+1, sticky=tk.W)
                coef_row.append(entry)
                if i < n_vars - 1:
                    ttk.Label(self.entrada_frame, text="x").grid(row=j+2, column=i+1, sticky=tk.E)
            self.coef_restricciones.append(coef_row)
            
            # Signo clickeable (<=, >=, =)
            signo_var = tk.StringVar(value="<=")
            signo_label = ttk.Label(self.entrada_frame, textvariable=signo_var, width=3, relief="solid")
            signo_label.grid(row=j+2, column=n_vars+1, sticky=tk.W)
            signo_label.bind("<Button-1>", lambda e, var=signo_var: self.alternar_signo(var))
            self.signos_restricciones.append(signo_var)
            
            # Término independiente
            term_entry = ttk.Entry(self.entrada_frame, width=5)
            term_entry.grid(row=j+2, column=n_vars+2, sticky=tk.W)
            self.terms_restricciones.append(term_entry)
    
    def alternar_signo(self, signo_var):
        # Alternar entre <=, >=, =
        signos = ["<=", ">=", "="]
        current = signo_var.get()
        next_signo = signos[(signos.index(current) + 1) % len(signos)]
        signo_var.set(next_signo)
    
    def resolver_problema(self):
        try:
            # Obtener coeficientes de la función objetivo
            c = [float(entry.get()) for entry in self.coef_obj]
            if self.modo.get() == "max":
                c = [-x for x in c]  # Negar para maximización
            
            # Obtener restricciones
            A_ub = []
            b_ub = []
            A_eq = []
            b_eq = []
            
            for j in range(len(self.coef_restricciones)):
                coef_row = [float(entry.get()) for entry in self.coef_restricciones[j]]
                signo = self.signos_restricciones[j].get()
                term = float(self.terms_restricciones[j].get())
                
                if signo == "<=":
                    A_ub.append(coef_row)
                    b_ub.append(term)
                elif signo == ">=":
                    # Convertir a <= multiplicando por -1
                    A_ub.append([-x for x in coef_row])
                    b_ub.append(-term)
                elif signo == "=":
                    A_eq.append(coef_row)
                    b_eq.append(term)
            
            A_ub = np.array(A_ub) if A_ub else None
            b_ub = np.array(b_ub) if b_ub else None
            A_eq = np.array(A_eq) if A_eq else None
            b_eq = np.array(b_eq) if b_eq else None
            
            # Resolver el problema
            resultado = resolver_programacion_lineal(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
            
            # Visualizar solución
            visualizar_solucion(A_ub if A_ub is not None else A_eq, 
                               b_ub if b_ub is not None else b_eq, 
                               resultado, c=c)
            
        except ValueError as e:
            messagebox.showerror("Error", f"Entrada inválida: {e}")

def resolver_programacion_lineal(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
    """
    Resuelve el problema de programación lineal, manejando tanto restricciones de
    desigualdad (<=, >=) como de igualdad (=).
    """
    resultado = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if resultado.success:
        print("Optimización exitosa!")
        es_maximizacion = any(x < 0 for x in c)
        valor_objetivo = -resultado.fun if es_maximizacion else resultado.fun
        print(f"Valor óptimo de la función objetivo: {valor_objetivo}")
        
        print("\nPunto óptimo (valores de las variables):")
        for i, valor in enumerate(resultado.x):
            print(f"x_{i+1} = {valor:.4f}")
        
        # Mostrar la función objetivo evaluada
        if es_maximizacion:
            coef_originales = [-x for x in c]
            terminos = [f"{coef:.2f}*{val:.4f}" for coef, val in zip(coef_originales, resultado.x)]
            print(f"\nFunción objetivo: Z = {' + '.join(terminos)} = {valor_objetivo:.4f}")
        else:
            terminos = [f"{coef:.2f}*{val:.4f}" for coef, val in zip(c, resultado.x)]
            print(f"\nFunción objetivo: Z = {' + '.join(terminos)} = {valor_objetivo:.4f}")
    else:
        print("La optimización no tuvo éxito.")
        print(f"Mensaje: {resultado.message}")
    
    return resultado

def graficar_solucion_2d(A_ub, b_ub, resultado, c=None, nombre_vars=None):
    """
    Grafica la región factible y la solución óptima para problemas de 2 variables.
    """
    if A_ub.shape[1] != 2:
        print("La función de graficación 2D solo funciona para problemas con 2 variables.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    max_x = max(10, resultado.x[0] * 1.5)
    max_y = max(10, resultado.x[1] * 1.5)
    x = np.linspace(0, max_x, 2000)
    
    for i in range(len(b_ub)):
        if A_ub[i, 0] == 0:
            ax.axhline(y=b_ub[i]/A_ub[i, 1], color='r', linestyle='-', alpha=0.3)
        elif A_ub[i, 1] == 0:
            ax.axvline(x=b_ub[i]/A_ub[i, 0], color='r', linestyle='-', alpha=0.3)
        else:
            y = (b_ub[i] - A_ub[i, 0] * x) / A_ub[i, 1]
            ax.plot(x, y, 'r-', alpha=0.3, label=f'Restricción {i+1}')
    
    # Corregido: Desempaquetar la tupla devuelta por meshgrid
    xx, yy = np.meshgrid(np.linspace(0, max_x, 100), np.linspace(0, max_y, 100))
    points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    inside = np.ones(len(points), dtype=bool)
    for i in range(len(b_ub)):
        inside = inside & (np.dot(points, A_ub[i]) <= b_ub[i])
    
    ax.scatter(points[inside, 0], points[inside, 1], c='lightblue', alpha=0.3, s=1)
    ax.plot(resultado.x[0], resultado.x[1], 'go', markersize=10, label='Punto óptimo')
    
    if nombre_vars:
        ax.set_xlabel(nombre_vars[0])
        ax.set_ylabel(nombre_vars[1])
    else:
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
    
    ax.annotate(f'({resultado.x[0]:.2f}, {resultado.x[1]:.2f})',
                xy=(resultado.x[0], resultado.x[1]),
                xytext=(resultado.x[0]+0.5, resultado.x[1]+0.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    
    ax.set_title('Problema de Programación Lineal (2D)')
    ax.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    plt.show()
def visualizar_solucion(A_ub, b_ub, resultado, c=None, nombre_vars=None):
    """
    Visualiza la solución de un problema de programación lineal.
    Selecciona automáticamente el tipo de visualización según el número de variables.
    """
    # Verificar si tenemos restricciones de desigualdad
    if A_ub is None or b_ub is None:
        print("No hay restricciones de desigualdad para graficar")
        return
    
    num_vars = A_ub.shape[1]
    
    if num_vars == 2:
        graficar_solucion_2d(A_ub, b_ub, resultado, c, nombre_vars)
    elif num_vars == 3:
        graficar_solucion_3d(A_ub, b_ub, resultado, c, nombre_vars)
        graficar_proyecciones(A_ub, b_ub, resultado, c, nombre_vars)
    elif num_vars > 3:
        print(f"El problema tiene {num_vars} variables, mostrando proyecciones 2D.")
        graficar_proyecciones(A_ub, b_ub, resultado, c, nombre_vars)
    else:
        print("El número de variables debe ser al menos 2 para visualización.")

def graficar_solucion_3d(A_ub, b_ub, resultado, c=None, nombre_vars=None):
    """
    Grafica la región factible y la solución óptima para problemas de 3 variables.
    """
    if A_ub.shape[1] != 3:
        print("La función de graficación 3D solo funciona para problemas con 3 variables.")
        return
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    max_x = max(10, resultado.x[0] * 1.5)
    max_y = max(10, resultado.x[1] * 1.5)
    max_z = max(10, resultado.x[2] * 1.5)
    
    xx, yy = np.meshgrid(np.linspace(0, max_x, 20), np.linspace(0, max_y, 20))
    
    for i in range(len(b_ub)):
        if A_ub[i, 2] != 0:
            z = (b_ub[i] - A_ub[i, 0] * xx - A_ub[i, 1] * yy) / A_ub[i, 2]
            ax.plot_surface(xx, yy, z, alpha=0.2, color='red')
    
    ax.scatter([resultado.x[0]], [resultado.x[1]], [resultado.x[2]], color='green', s=100, label='Punto óptimo')
    
    if nombre_vars:
        ax.set_xlabel(nombre_vars[0])
        ax.set_ylabel(nombre_vars[1])
        ax.set_zlabel(nombre_vars[2])
    else:
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_zlabel('x₃')
    
    ax.text(resultado.x[0], resultado.x[1], resultado.x[2], 
            f'({resultado.x[0]:.2f}, {resultado.x[1]:.2f}, {resultado.x[2]:.2f})',
            color='black')
    
    ax.set_title('Problema de Programación Lineal (3D)')
    ax.legend()
    plt.show()

def graficar_proyecciones(A_ub, b_ub, resultado, c=None, nombre_vars=None):
    """
    Grafica las proyecciones 2D para problemas de más de 2 variables.
    """
    num_vars = A_ub.shape[1]
    if num_vars <= 2:
        print("Para 2 variables, use graficar_solucion_2d en su lugar.")
        return
    
    combinaciones = list(combinations(range(num_vars), 2))
    n = len(combinaciones)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5*rows), squeeze=False)
    
    if nombre_vars is None:
        nombre_vars = [f'x_{i+1}' for i in range(num_vars)]
    
    for k, (i, j) in enumerate(combinaciones):
        row, col = k // cols, k % cols
        ax = axs[row, col]
        ax.plot(resultado.x[i], resultado.x[j], 'go', markersize=10)
        ax.annotate(f'({resultado.x[i]:.2f}, {resultado.x[j]:.2f})',
                    xy=(resultado.x[i], resultado.x[j]),
                    xytext=(resultado.x[i]+0.5, resultado.x[j]+0.5),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        ax.set_xlabel(nombre_vars[i])
        ax.set_ylabel(nombre_vars[j])
        ax.set_title(f'Proyección {nombre_vars[i]} vs {nombre_vars[j]}')
        ax.grid(True)
    
    for k in range(len(combinaciones), rows*cols):
        row, col = k // cols, k % cols
        axs[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = ProgramacionLinealApp(root)
    root.mainloop()