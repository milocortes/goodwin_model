using Turing
using DifferentialEquations
using Plots,StatsPlots


function goodwin(du,u,p,t)

    ## Parámetros iniciales
    α,β,c,d,γ,ν = p

    """
    Derivada con respecto al tiempo del vector de estado.
        * x es el vector de estado (arreglo)
        * t es el tiempo
    """
    L,w,a,N  = u

    """
    Definiciones e identidades
    """
    Y = a * L
    K = (1/ν) * Y
    Π = Y - w*L
    I = Π
    λ = L/N

    # Derivadas con respecto al tiempo
    du[1] = L*(  ((1-(w/a))/ν) - γ - α)
    du[2] = (-c + (d * λ))*w
    du[3] = α*a
    du[4] = β*N

end

# Condiciones iniciales de L,w,a,N
L_0 = 300
w_0 = 0.95
a_0 = 1
N_0 = 300

u0 = [L_0,w_0,a_0,N_0]
p = [0.02,0.01,4.8,5,0.01,3]

prob1 = ODEProblem(goodwin,u0,(0.0,20.0),p)
sol = solve(prob1,Tsit5())


# Calculamos la tasa de empleo
employment_rate = [sol.u[i][1]/sol.u[i][4] for i in 1:23]
# Calculamos la participación de los salarios en el ingreso
wage_rate =  [(sol.u[i][1]*sol.u[i][2]) /(sol.u[i][3]*sol.u[i][1]) for i in 1:23]
# Calculamos el ingreso
Y = [sol.u[i][1]*sol.u[i][3] for i in 1:23]

# Graficamos
plot([i for i in 1:23],hcat(employment_rate,wage_rate),
 title = "Evolución de la tasa de empleo y participación de los salarios en el ingreso",
 label = ["Tasa de empleo" "Wage share"],
  lw = 3)


# Direct Handling of Bayesian Estimation with Turing
sol1 = solve(prob1,Tsit5(),saveat=0.1)
odedata = Array(sol1) + 0.8 * randn(size(Array(sol1)))

Turing.setadbackend(:forwarddiff)

@model function fitlv(data, prob1)
    σ ~ InverseGamma(2, 3) # ~ is the tilde character

    α ~ truncated(Normal(0.02,0.5),0,2)
    β ~ truncated(Normal(0.01,0.005),0,1)
    c ~ truncated(Normal(4.8,0.5),0,10)
    d ~ truncated(Normal(5,0.5),0,10)
    γ ~ truncated(Normal(0.01,0.005),0,1)
    ν ~ truncated(Normal(3,0.5),0,10)

    p = [α,β,c,d,γ,ν]

    prob = remake(prob1, p=p)
    predicted = solve(prob,Tsit5(),saveat=0.1)

    for i = 1:length(predicted)
        data[:,i] ~ MvNormal(predicted[i], σ)
    end
end

model = fitlv(odedata, prob1)

# This next command runs 3 independent chains without using multithreading.
chain = mapreduce(c -> sample(model, NUTS(.65),1000), chainscat, 1:3)
plot(chain)

pl = scatter(sol1.t, odedata');

chain_array = Array(chain)
for k in 1:300
    resol = solve(remake(prob1,p=chain_array[rand(1:1500), 1:6]),Tsit5(),saveat=0.1)
    plot!(resol, alpha=0.1, color = "#BBBBBB", legend = false)
end
# display(pl)
plot!(sol1, w=1, legend = false)
