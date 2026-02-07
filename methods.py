import math
import numpy as np
import matplotlib.pyplot as plt

# Task 1: Numerical Solution of dy/dx = e^x - y^2, y(0) = 1

def y_exact(x):
    h = 0.0001  # very small step size for high accuracy
    t = 0.0
    y = 1.0
    while t < x:
        # Compute RK4 slopes
        k1 = math.exp(t) - y**2
        k2 = math.exp(t + h / 2.0) - (y + h * k1 / 2.0)**2
        k3 = math.exp(t + h / 2.0) - (y + h * k2 / 2.0)**2
        k4 = math.exp(t + h) - (y + h * k3)**2
        # RK4 update
        y += (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return y

# Function f(x, y) = e^x - y^2 (right-hand side of the ODE)
def f(x, y):
    return math.exp(x) - y**2


def picard_method(h, N=5, t0=0.0, y0=1.0, t_end=2.0):
    # initialize the grid T
    T = [t0]
    n_steps = int((t_end - t0) / h)
    for i in range(n_steps):
        T.append(T[-1] + h)

    # initial guess: y = y0 everywhere
    phi_prev = []
    for i in range(len(T)):
        phi_prev.append(y0)

    # perform N Picard iterations
    for n in range(N):
        # iteration with just the initial condition
        phi_new = [y0]

        # calculate the area from T[0] to T[i] for every i
        for i in range(1, len(T)):
            current_n = i

            # sum of internal points for the integral from 0 to T[i]
            internal_sum = 0.0
            for j in range(1, current_n):
                internal_sum += f(T[j], phi_prev[j])

            # trapezoid rule for integral
            f_start = f(T[0], phi_prev[0])
            f_end = f(T[i], phi_prev[i])

            area = (h / 2.0) * (f_start + f_end + 2 * internal_sum)
            phi_new.append(y0 + area)

        phi_prev = phi_new

    # 4. Initialize Y, YE, E and append in a loop as requested
    Y, YE, E = [], [], []
    for i in range(len(T)):
        current_y = phi_prev[i]
        exact_y = y_exact(T[i])

        Y.append(current_y)
        YE.append(exact_y)
        E.append(abs(current_y - exact_y))

    return T, Y, YE, E


# 2. Taylor's Method (3rd order)
def taylor_method(h, t0=0.0, y0=1.0, t_end=2.0):
    # 1st, 2nd, and 3rd derivatives
    def df_all(x, y):
        dy = math.exp(x) - y**2
        d2y = math.exp(x) - 2 * y * dy
        d3y = math.exp(x) - 2 * dy**2 - 2 * y * d2y
        return dy, d2y, d3y

    # Initialize lists to store results
    T, Y, YE, E = [t0], [y0], [y_exact(t0)], [0.0]
    t, y = t0, y0

    # Use 3rd-order Taylor expansion to compute next value
    n_steps = int((t_end - t0) / h)
    for step in range(n_steps):
        dy, d2y, d3y = df_all(t, y)
        y += h * dy + (h**2 / 2.0) * d2y + (h**3 / 6.0) * d3y
        t += h
        T.append(t)
        Y.append(y)
        YE.append(y_exact(t))
        E.append(abs(Y[-1] - YE[-1]))
    return T, Y, YE, E

# 3. Euler's Method
def euler_method(h, t0=0.0, y0=1.0, t_end=2.0):
    T, Y, YE, E = [t0], [y0], [y_exact(t0)], [0.0]
    t, y = t0, y0

    # Standard Euler iteration: y = y + h*f(x, y)
    n_steps = int((t_end - t0) / h)
    for step in range(n_steps):
        y += h * f(t, y)
        t += h
        T.append(t)
        Y.append(y)
        YE.append(y_exact(t))
        E.append(abs(Y[-1] - YE[-1]))
    return T, Y, YE, E

# 4. Modified Euler's Method
def modified_euler(h, t0=0.0, y0=1.0, t_end=2.0):
    T, Y, YE, E = [t0], [y0], [y_exact(t0)], [0.0]
    t, y = t0, y0

    # Predictor-corrector method
    n_steps = int((t_end - t0) / h)
    for step in range(n_steps):
        k1 = f(t, y)
        y_pred = y + h * k1  # Euler prediction
        k2 = f(t + h, y_pred)
        y += (h / 2.0) * (k1 + k2)  # average slope
        t += h
        T.append(t)
        Y.append(y)
        YE.append(y_exact(t))
        E.append(abs(Y[-1] - YE[-1]))
    return T, Y, YE, E

# 5. Runge-Kutta 3rd Order Method
def rk3(h, t0=0.0, y0=1.0, t_end=2.0):
    T, Y, YE, E = [t0], [y0], [y_exact(t0)], [0.0]
    t, y = t0, y0

    n_steps = int((t_end - t0) / h)
    for step in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h, y - h*k1 + 2*h*k2)
        y += h * (k1 + 4*k2 + k3) / 6  # RK3 update
        t += h
        T.append(t)
        Y.append(y)
        YE.append(y_exact(t))
        E.append(abs(Y[-1] - YE[-1]))
    return T, Y, YE, E

# 6. Runge-Kutta 4th Order Method
def rk4(h, t0=0.0, y0=1.0, t_end=2.0):
    T, Y, YE, E = [t0], [y0], [y_exact(t0)], [0.0]
    t, y = t0, y0

    n_steps = int((t_end - t0) / h)
    for step in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h*k1/2)
        k3 = f(t + h/2, y + h*k2/2)
        k4 = f(t + h, y + h*k3)
        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6  # RK4 update
        t += h
        T.append(t)
        Y.append(y)
        YE.append(y_exact(t))
        E.append(abs(Y[-1] - YE[-1]))
    return T, Y, YE, E

# all solvers for a given h and display final values
def print_table(method_name, T, Y, YE, E):
    print(f"\n--- {method_name} ---")
    print(f"{'Step':<6}{'x_i':<10}{'y_numerical':<18}{'y_exact':<18}{'abs_error':<18}")
    for i, (t, y, ye, e) in enumerate(zip(T, Y, YE, E)):
        print(f"{i:<6}{t:<10.4f}{y:<18.10f}{ye:<18.10f}{e:<18.10f}")


# Run all methods and show tables
def run_all_methods(h):
    print(f"\nResults for h = {h}")
    methods = {
        "Picard": picard_method,
        "Euler": euler_method,
        "Modified Euler": modified_euler,
        "Taylor 3rd Order": taylor_method,
        "Runge-Kutta 3rd Order": rk3,
        "Runge-Kutta 4th Order": rk4
    }


    for name, method in methods.items():
        T, Y, YE, E = method(h)
        print_table(name, T, Y, YE, E)

# Plot all method results and exact solution
def plot_solutions(h):
    T0, Y0, YE0, E0 = picard_method(h)
    T1, Y1, YE1, E1 = euler_method(h)
    T2, Y2, YE2, E2 = modified_euler(h)
    T3, Y3, YE3, E3 = taylor_method(h)
    T4, Y4, YE4, E4 = rk3(h)
    T5, Y5, YE5, E5 = rk4(h)
    TE = np.linspace(0, 2, 100)
    YE = [y_exact(t) for t in TE]

    plt.figure(figsize=(10, 6))
    plt.plot(TE, YE, label='Exact', color='black')
    plt.plot(T0, Y0, '--', label='Picard')
    plt.plot(T1, Y1, '--', label='Euler')
    plt.plot(T2, Y2, '--', label='Mod Euler')
    plt.plot(T3, Y3, '--', label='Taylor')
    plt.plot(T4, Y4, '--', label='RK3')
    plt.plot(T5, Y5, '--', label='RK4')
    plt.title(f"ODE Solutions for h = {h}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid()
    plt.show()

run_all_methods(0.1)
plot_solutions(0.1)
run_all_methods(0.2)
plot_solutions(0.2)

# Task 2: SIR Model solved with RK4

# dS/dt = -beta * S * I
# dI/dt =  beta * S * I - gamma * I
# dR/dt =  gamma * I

# Model parameters (given)
beta = 0.0003 # infection rate per day
gamma = 0.1 # recovery rate per day


# Initial conditions
S0 = 999000 # initial susceptible
I0 = 1000 # initial infected
R0 = 0 # initial recovered

# Simulation settings
h = 0.1 # step size in days
t_end = 100.0 # simulate for 100 days

N = S0 + I0 + R0  # total population


# A: Define SIR derivatives (right-hand side functions)
def sir_derivatives(t, S, I, R, beta_value, gamma_value, N_value):
    dS = -beta_value * S * I / N_value  # susceptible decreases by infections
    dI = beta_value * S * I / N_value - gamma_value * I   # infections minus recoveries
    dR = gamma_value * I  # recovered increase from infected
    return dS, dI, dR # return the three derivatives


# A: RK4 solver for the SIR system
def rk4_sir(S_init, I_init, R_init, beta_value, gamma_value, h, t_end, N_value):
    n_steps = int(t_end / h)  # number of steps
    T = np.zeros(n_steps + 1)  # time array
    S = np.zeros(n_steps + 1)  # susceptible array
    I = np.zeros(n_steps + 1)  # infected array
    R = np.zeros(n_steps + 1)  # recovered array

    #initial values
    T[0] = 0.0
    S[0] = S_init
    I[0] = I_init
    R[0] = R_init

    for n in range(n_steps):
        t = T[n]
        s = S[n]
        i = I[n]
        r = R[n]

        # k1 slopes at the beginning of the interval
        dS1, dI1, dR1 = sir_derivatives(t, s, i, r, beta_value, gamma_value, N_value)

        # k2 slopes at the midpoint using k1
        dS2, dI2, dR2 = sir_derivatives(
            t + h/2,
            s + (h/2)*dS1,
            i + (h/2)*dI1,
            r + (h/2)*dR1,
            beta_value,
            gamma_value,
            N_value
        )

        # k3 slopes at the midpoint using k2
        dS3, dI3, dR3 = sir_derivatives(
            t + h/2,
            s + (h/2)*dS2,
            i + (h/2)*dI2,
            r + (h/2)*dR2,
            beta_value,
            gamma_value,
            N_value
        )

        # k4 slopes at the end using k3
        dS4, dI4, dR4 = sir_derivatives(
            t + h,
            s + h*dS3,
            i + h*dI3,
            r + h*dR3,
            beta_value,
            gamma_value,
            N_value
        )

        # RK4 weighted update for S, I, R
        S[n+1] = s + (h/6)*(dS1 + 2*dS2 + 2*dS3 + dS4)
        I[n+1] = i + (h/6)*(dI1 + 2*dI2 + 2*dI3 + dI4)
        R[n+1] = r + (h/6)*(dR1 + 2*dR2 + 2*dR3 + dR4)
        T[n+1] = t + h # next time

    return T, S, I, R # return all arrays


# SECTION C: helper calculations (peak + total infected ever)
def peak_infected(T, I):
    idx = int(np.argmax(I)) # index where I is maximum
    return I[idx], T[idx] # return peak value and day

def total_infected_ever(N_value, S_end, R0_initial):
    return (N_value - S_end) - R0_initial # total ever infected


# SECTION D amd E:
T1, S1, I1, R1 = rk4_sir(S0, I0, R0, beta, gamma, h, t_end, N)  # solve original case
Ipeak1, tpeak1 = peak_infected(T1, I1) # peak infected + day
total1 = total_infected_ever(N, S1[-1], R0) # total infected ever


# SECTION F: vaccination: reduce initial S by 50%
# vaccinated people become immune immediately -> move from S to R.
S0_vac = 0.5 * S0 # susceptible reduced by 50%
vaccinated = S0 - S0_vac # number vaccinated
R0_vac = R0 + vaccinated  # vaccinated moved into R (immune)
I0_vac = I0 # infected unchanged

T2, S2, I2, R2 = rk4_sir(S0_vac, I0_vac, R0_vac, beta, gamma, h, t_end, N)
Ipeak2, tpeak2 = peak_infected(T2, I2)
total2 = total_infected_ever(N, S2[-1], R0_vac)


# SECTION G: social distancing: reduce beta by 50%
beta_sd = 0.5 * beta

T3, S3, I3, R3 = rk4_sir(S0, I0, R0, beta_sd, gamma, h, t_end, N)
Ipeak3, tpeak3 = peak_infected(T3, I3)
total3 = total_infected_ever(N, S3[-1], R0)


# print answers (peak + totals)
print("Scenario 1: Original")
print(f"Peak infected: {Ipeak1:,.0f} people on day {tpeak1:.1f}")
print(f"Total infected at some point: {total1:,.0f} people")

print("\nScenario 2: Vaccination (S(0) reduced by 50%)")
print(f"Peak infected: {Ipeak2:,.0f} people on day {tpeak2:.1f}")
print(f"Total infected at some point: {total2:,.0f} people")

print("\nScenario 3: Social distancing (beta reduced by 50%)")
print(f"Peak infected: {Ipeak3:,.0f} people on day {tpeak3:.1f}")
print(f"Total infected at some point: {total3:,.0f} people")


# Plot S(t), I(t), R(t) for the ORIGINAL scenario
# Plot S(t) alone
plt.figure(figsize=(10, 6))
plt.plot(T1, S1, label="S(t) Susceptible")
plt.title("S(t) over time (Original)")
plt.xlabel("Time (days)")
plt.ylabel("Susceptible people")
plt.grid(True)
plt.legend()
plt.ticklabel_format(axis='y', style='plain', useOffset=False)
plt.show()

# Plot I(t) alone
plt.figure(figsize=(10, 6))
plt.plot(T1, I1, label="I(t) Infected")
plt.title("I(t) over time (Original)")
plt.xlabel("Time (days)")
plt.ylabel("Infected people")
plt.grid(True)
plt.legend()
plt.show()

# Plot R(t) alone
plt.figure(figsize=(10, 6))
plt.plot(T1, R1, label="R(t) Recovered")
plt.title("R(t) over time (Original)")
plt.xlabel("Time (days)")
plt.ylabel("Recovered people")
plt.grid(True)
plt.legend()
plt.show()


# Compare infected curves between scenarios
plt.figure(figsize=(10, 6))
plt.plot(T1, I1, label="Infected (Original)")
plt.plot(T2, I2, label="Infected (Vaccination 50% S0)")
plt.plot(T3, I3, label="Infected (Social distancing 50% beta)")
plt.title("Comparison of Infected I(t) Across Scenarios")
plt.xlabel("Time (days)")
plt.ylabel("Infected people")
plt.grid(True)
plt.legend()
plt.show()