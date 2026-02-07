import numpy as np
import matplotlib.pyplot as plt
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
    # To get R0 = 3 (standard outbreak)
    real_beta = 0.0003 * 1000  # Scaling beta so R0 > 1

    dS = -real_beta * S * I / N_value
    dI = real_beta * S * I / N_value - gamma_value * I
    dR = gamma_value * I
    return dS, dI, dR


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