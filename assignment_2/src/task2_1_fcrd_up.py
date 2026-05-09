"""
FCR-D UP reserve bid for a stochastic flexible load under the P90 requirement.
Implements Task 2.1 of Assignment 2.

Two methods are used to determine the optimal reserve bid:

    ALSO-X: empirical chance-constrained formulation. The reserve bid is the
    largest value such that at most 10% of in-sample profiles violate the
    availability constraint (i.e. the (floor(epsilon * N_in) + 1)-th order
    statistic of the sorted minimum-load vector).

    CVaR approximation: more conservative alternative. Instead of only
    limiting the number of violations, it penalises their magnitude via
    the expected shortfall in the tail. The resulting bid is the mean of
    the worst tail_size minimum-load values.

Load profiles are generated as bounded random walks with:
    - consumption in [220, 600] kW
    - minute-to-minute changes <= 35 kW
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Parameters
N_PROFILES = 300
N_MINUTES = 60
LOAD_MIN = 220
LOAD_MAX = 600
MAX_STEP = 35

N_IN    = 100      # number of in-sample profiles (first 100)
P_REQ   = 0.90     # required reliability level (P90)
EPSILON = 1 - P_REQ  # maximum allowed violation rate (0.10)


def generate_load_profiles(
    n_profiles: int = N_PROFILES,
    n_minutes:  int = N_MINUTES,
    load_min:   float = LOAD_MIN,
    load_max:   float = LOAD_MAX,
    max_step:   float = MAX_STEP,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate stochastic load profiles as bounded random walks.

    At each minute, a Gaussian step (clipped to [-max_step, max_step]) is
    added to the previous value. Boundary violations are corrected by
    reflection rather than clipping, to avoid artificial pile-up at the bounds.

    Parameters
    ----------
    n_profiles : int
        Number of load profiles to generate (default 300).
    n_minutes : int
        Length of each profile in minutes (default 60).
    load_min, load_max : float
        Consumption bounds in kW (default [220, 600]).
    max_step : float
        Maximum allowed minute-to-minute change in kW (default 35).
    seed : int
        RNG seed for reproducibility (default 42).

    Returns
    -------
    np.ndarray of shape (n_profiles, n_minutes), consumption in kW.
    """
    rng = np.random.default_rng(seed)

    profiles = np.zeros((n_profiles, n_minutes))
    profiles[:, 0] = rng.uniform(300.0, 580.0, size=n_profiles)

    for m in range(1, n_minutes):
        step = rng.normal(loc=0.0, scale=18.0, size=n_profiles)
        step = np.clip(step, -max_step, max_step)

        new_value = profiles[:, m - 1] + step

        # Reflect values at the bounds instead of clipping them
        below = new_value < load_min
        new_value[below] = load_min + (load_min - new_value[below])

        # Reflect at upper bound: values above load_max bounce back down
        above = new_value > load_max
        new_value[above] = load_max - (new_value[above] - load_max)

        # Safety clipping in case reflection overshoots
        profiles[:, m] = np.clip(new_value, load_min, load_max)

    return profiles


# Generate profiles
profiles = generate_load_profiles(
    N_PROFILES,
    N_MINUTES,
    LOAD_MIN,
    LOAD_MAX,
    MAX_STEP
)

# Split into in-sample and out-of-sample
profiles_in = profiles[:N_IN]
profiles_out = profiles[N_IN:]

# Available FCR-D UP reserve per profile
# Since reserve must be available during the whole hour,
# the available reserve is the minimum load in the hour.
A_in = profiles_in.min(axis=1)

# ALSO-X / empirical P90 solution
A_sorted = np.sort(A_in)
allowed_violations = int(np.floor(EPSILON * N_IN))

R_alsox = A_sorted[allowed_violations]

# CVaR conservative solution
tail_size = int(np.ceil(EPSILON * N_IN))
R_cvar = A_sorted[:tail_size].mean()

# Reliability checks
reliability_alsox = np.mean(A_in >= R_alsox)
reliability_cvar = np.mean(A_in >= R_cvar)

shortfall_alsox = np.mean(np.maximum(R_alsox - A_in, 0))
shortfall_cvar = np.mean(np.maximum(R_cvar - A_in, 0))

results = pd.DataFrame({
    "Method": ["ALSO-X", "CVaR"],
    "Reserve bid [kW]": [R_alsox, R_cvar],
    "In-sample reliability": [reliability_alsox, reliability_cvar],
    "In-sample violation rate": [1 - reliability_alsox, 1 - reliability_cvar],
    "In-sample expected shortfall [kW]": [shortfall_alsox, shortfall_cvar]
})

print("Minimum load:", profiles.min())
print("Maximum load:", profiles.max())
print("Maximum minute-to-minute change:", np.max(np.abs(np.diff(profiles, axis=1))))
print()
print(results.round(4))