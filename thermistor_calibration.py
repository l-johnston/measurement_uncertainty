"""Thermistor calibration

A. Possolo and H. K. Iyer, Concepts and tools for the evaluation of measurement uncertainty,
Rev. Sci. Instrum. 88, 011301 (2017); doi: 10.1063/1.4974274.

VI B 3
"""
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize

# Table II
# PRT °C
tau = np.array([20.91, 25.42, 30.50, 34.96, 40.23, 34.93, 30.05, 25.03, 20.87, 16.41, 16.40, 39.34])

# Thermistor °C
t = np.array(
    [
        20.85,
        25.52,
        30.70,
        35.22,
        40.47,
        35.18,
        30.25,
        25.10,
        20.81,
        16.23,
        16.22,
        39.56,
    ]
)

# 3rd order regression model parameters
params = Parameters()
params.add("ß0", value=0, min=-1, max=1, vary=True)
params.add("ß1", value=1, min=0, max=2, vary=True)
params.add("ß2", value=0, min=-1, max=1, vary=True)
params.add("ß3", value=0, min=-1, max=1, vary=True)


def residual(parameters, x, data=None):
    """least squares objective function"""
    ß0 = parameters["ß0"]
    ß1 = parameters["ß1"]
    ß2 = parameters["ß2"]
    ß3 = parameters["ß3"]
    model = ß0 + ß1 * x + ß2 * x**2 + ß3 * x**3
    if data is None:
        return model
    return data - model


fit_result = minimize(residual, params, args=(tau,), kws={"data": t})


def phi(x, pars=None):
    """thermistor t vs. PRT tau"""
    if pars is None:
        ß0 = fit_result.params["ß0"].value
        ß1 = fit_result.params["ß1"].value
        ß2 = fit_result.params["ß2"].value
        ß3 = fit_result.params["ß3"].value
    else:
        ß0 = pars["ß0"].value
        ß1 = pars["ß1"].value
        ß2 = pars["ß2"].value
        ß3 = pars["ß3"].value
    return ß0 + ß1 * x + ß2 * x**2 + ß3 * x**3


def phi_inv(y, pars=None):
    """PRT tau vs. thermistor t"""
    if pars is None:
        ß0 = fit_result.params["ß0"].value
        ß1 = fit_result.params["ß1"].value
        ß2 = fit_result.params["ß2"].value
        ß3 = fit_result.params["ß3"].value
    else:
        ß0 = pars["ß0"].value
        ß1 = pars["ß1"].value
        ß2 = pars["ß2"].value
        ß3 = pars["ß3"].value
    p = np.polynomial.Polynomial(coef=[ß0 - y, ß1, ß2, ß3])
    root = None
    for r in p.roots():
        if tau.min() * 0.9 <= r <= tau.max() * 1.1:
            root = r
            break
    return root


# Evaluate measurement uncertainty, VII B 3
u_prt = 0.0015  # °C, PRT standard uncertainty
u_cable = 0.01  # °C, cable standard uncertainty
δ_std = np.std(residual(fit_result.params, tau, t), ddof=1)  # 0.0103 °C corrected sample std_dev
# δ_std = 0.012  # °C, reported value in paper
γ = np.sqrt(δ_std**2 + u_cable**2)  # °C
m = 100
Θ = np.linspace(t.min(), t.max(), m)
K = 10000
rng = np.random.default_rng()
tau_rand = np.array([rng.normal(loc=t, scale=u_prt, size=K) for t in tau]).T
t_rand = np.array([rng.normal(loc=t, scale=γ, size=K) for t in phi(tau)]).T
uncertainty_envelope = [(phi_inv(y), phi_inv(y)) for y in Θ]
K_fits = [
    minimize(residual, params, args=(x,), kws={"data": y}).params for x, y in zip(tau_rand, t_rand)
]
for i in range(K):
    for j, y in enumerate(Θ):
        x = phi_inv(y, pars=K_fits[i])
        ue_min, ue_max = uncertainty_envelope[j]
        if x < ue_min:
            ue_min = x
        if x > ue_max:
            ue_max = x
        uncertainty_envelope[j] = (ue_min, ue_max)
uncertainty_envelope = np.array(uncertainty_envelope).T
fig, ax = plt.subplots()
tau_predicted = np.array([phi_inv(y) for y in Θ])
ax.scatter(t, tau, color="blue", zorder=2, label="Measurements")
ax.plot(Θ, tau_predicted, color="red", zorder=1, label="Calibration")
ue_lower = 50 * (uncertainty_envelope[0] - tau_predicted) + tau_predicted
ue_upper = 50 * (uncertainty_envelope[1] - tau_predicted) + tau_predicted
ax.fill_between(
    Θ, ue_lower, ue_upper, color="red", alpha=0.2, zorder=0, label="50x Coverage Envelope"
)
ax.set_xlim(12.5, 42.5)
ax.set_ylim(12.5, 42.5)
ax.set(box_aspect=1)
ax.set_xlabel("Thermistor Indication t (°C)")
ax.set_ylabel("Temperature τ (°C)")
ax.legend()
plt.show()
print(np.max((ue_upper - ue_lower) / (2 * 50)))  # k = 2 for 95 % coverage
# 0.050 °C report states 95 % coverage
