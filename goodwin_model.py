import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
import pylab as p
from scipy.integrate import odeint

## Parámetros iniciales

α = 0.02
β = 0.01
c = 4.8
d = 5
γ = 0.01
ν = 3

def F(x, t):
    """
    Derivada con respecto al tiempo del vector de estado.
        * x es el vector de estado (arreglo)
        * t es el tiempo (escalar)
    """
    L,w,a,N  = x

    """
    Definiciones e identidades
    """
    Y = a * L
    K = (1/ν) * Y
    Π = Y - w*L
    I = Π
    λ = L/N

    # Derivadas con respecto al tiempo
    dL = L*(  ((1-(w/a))/ν) - γ - α)
    dw = (-c + (d * λ))*w
    da = α*a
    dN = β*N

    return dL, dw, da, dN


# Condiciones iniciales de L,w,a,N
L_0 = 300
w_0 = 0.95
a_0 = 1
N_0 = 300

x_0 = L_0,w_0,a_0,N_0

## Se define una función para generar las trayectorias
def solve_path(t_vec, x_init=x_0):
    G = lambda x, t: F(x, t)
    L_path, w_path, a_path, N_path = odeint(G, x_init, t_vec).transpose()

    return L_path, w_path, a_path, N_path

## Se resuelve para 50 años
t_length = 50
t_vec = np.array([x for x in range(t_length)])

L_end,w_end,a_end,N_end = solve_path(t_vec)

# Calculamos la tasa de empleo
employment_rate = [(L/N)for L,N in zip(L_end,N_end)]
# Calculamos la participación de los salarios en el ingreso
wage_rate = [(L*w)/(a*L) for L,w,a in zip(L_end,w_end,a_end)]
# Calculamos el ingreso
Y = [(a*L) for L,a in zip(L_end,a_end)]

## Generamos la figura en el tiempo
f1 = p.figure()
p.plot(t_vec, employment_rate, 'r-', label='Tasa de empleo')
p.plot(t_vec, wage_rate  , 'b-', label='Wage share')
p.grid()
p.legend(loc='best')
p.xlabel('Tiempo')
p.ylabel('Tasa')
p.title('Evolución de la tasa de empleo y participación de los salarios en el ingreso')
plt.show()
f1.savefig('time_evolution.png')

## Scatter plot
f2 = p.figure()
p.plot(employment_rate, wage_rate, 'o', color='blue');
p.ylabel('Wage share')
p.xlabel('Tasa de empleo')
p.title('Ciclo empleo-distribución')
plt.show()
f2.savefig('ciclo.png')
