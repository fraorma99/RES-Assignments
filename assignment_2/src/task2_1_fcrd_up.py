import numpy as np
import pandas as pd

np.random.seed(42)

# Parameters
N_PROFILES = 300
N_MINUTES = 60
LOAD_MIN = 220
LOAD_MAX = 600
MAX_STEP = 35

N_IN = 100
P_REQ = 0.90
EPSILON = 1 - P_REQ


def generate_load_profiles(
    n_profiles: int = 300,
    n_minutes: int = 60,
    load_min: float = 220.0,
    load_max: float = 600.0,
    max_step: float = 35.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate stochastic load profiles.

    The profile is generated with a bounded random walk using reflection
    at the lower and upper limits. This avoids artificial pile-up at 220 kW
    while still satisfying:
        220 <= load <= 600 kW
        |load[m] - load[m-1]| <= 35 kW
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

        above = new_value > load_max
        new_value[above] = load_max - (new_value[above] - load_max)

        # Safety clipping in case reflection overshoots
        profiles[:, m] = np.clip(new_value, load_min, load_max)

    return profiles


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