import streamlit as st
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

st.title("Programación Lineal - Maximizar/Minimizar")

modo = st.radio("Objetivo:", ("Minimizar", "Maximizar"))
num_vars = st.number_input("Número de variables", min_value=2, max_value=10, value=2, step=1)
num_restricciones = st.number_input("Número de restricciones", min_value=1, max_value=20, value=2, step=1)

st.write("Ingrese los coeficientes de la función objetivo:")
coef_obj = []
for i in range(num_vars):
    val = st.number_input(f"x{i+1}", value=0.0, key=f"obj_{i}")
    coef_obj.append(val)

st.write("Ingrese las restricciones:")

coef_restricciones = []
signos = []
term_indep = []

for j in range(num_restricciones):
    cols = st.columns(num_vars + 2)
    coefs = []
    for i in range(num_vars):
        coefs.append(cols[i].number_input(f"x{i+1} restricción {j+1}", value=0.0, key=f"res_{j}_{i}"))
    signo = cols[num_vars].selectbox("Signo", ["<=", ">=", "="], key=f"signo_{j}")
    term = cols[num_vars+1].number_input("Termino", value=0.0, key=f"term_{j}")
    coef_restricciones.append(coefs)
    signos.append(signo)
    term_indep.append(term)

if st.button("Resolver"):
    try:
        c = np.array(coef_obj)
        if modo == "Maximizar":
            c = -c

        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []

        for i in range(num_restricciones):
            coefs = coef_restricciones[i]
            signo = signos[i]
            term = term_indep[i]

            if signo == "<=":
                A_ub.append(coefs)
                b_ub.append(term)
            elif signo == ">=":
                A_ub.append([-x for x in coefs])
                b_ub.append(-term)
            else:
                A_eq.append(coefs)
                b_eq.append(term)

        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None

        resultado = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')

        if resultado.success:
            valor_obj = -resultado.fun if modo == "Maximizar" else resultado.fun
            st.success(f"Valor óptimo: {valor_obj:.4f}")
            for i, val in enumerate(resultado.x):
                st.write(f"x{i+1} = {val:.4f}")
        else:
            st.error("No se encontró solución óptima.")
            st.write(resultado.message)
    except Exception as e:
        st.error(f"Error: {e}")
