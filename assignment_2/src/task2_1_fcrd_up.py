import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




N_PROFILES = 300
N_MINUTES = 60
LOAD_MIN = 220.0
LOAD_MAX = 600.0
MAX_STEP = 35.0

N_IN = 100
P_REQ = 0.90
EPSILON = 1.0 - P_REQ



# Load profile generation


def generate_load_profiles(
    n_profiles: int = N_PROFILES,
    n_minutes: int = N_MINUTES,
    load_min: float = LOAD_MIN,
    load_max: float = LOAD_MAX,
    max_step: float = MAX_STEP,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate stochastic load profiles.

    The profiles are generated with a bounded random walk using reflection
    at the lower and upper limits. This avoids an artificial pile-up at the
    lower bound while satisfying the assignment requirements:

        220 <= load <= 600 kW
        |load[m] - load[m-1]| <= 35 kW
    """
    rng = np.random.default_rng(seed)

    profiles = np.zeros((n_profiles, n_minutes))

    # Start away from the lower bound to avoid all profiles immediately
    # collapsing to the minimum load.
    profiles[:, 0] = rng.uniform(300.0, 580.0, size=n_profiles)

    for m in range(1, n_minutes):
        step = rng.normal(loc=0.0, scale=18.0, size=n_profiles)
        step = np.clip(step, -max_step, max_step)

        new_value = profiles[:, m - 1] + step

        # Reflect at lower bound instead of clipping
        below = new_value < load_min
        new_value[below] = load_min + (load_min - new_value[below])

        # Reflect at upper bound instead of clipping
        above = new_value > load_max
        new_value[above] = load_max - (new_value[above] - load_max)

        # Safety clipping
        profiles[:, m] = np.clip(new_value, load_min, load_max)

    return profiles




def run_task21(seed: int = 42) -> dict:
    """
    Run Task 2.1.

    Returns a dictionary containing the generated profiles, in-sample and
    out-of-sample split, reserve bids, and result table.
    """
    profiles = generate_load_profiles(seed=seed)

    # Split into in-sample and out-of-sample
    profiles_in = profiles[:N_IN]
    profiles_out = profiles[N_IN:]

    # Available FCR-D UP reserve per profile.
    # The reserve must be available during the full hour, so the available
    # reserve of each profile is the minimum load over the 60 minutes.
    A_in = profiles_in.min(axis=1)

    # ALSO-X / empirical P90 solution
    A_sorted = np.sort(A_in)
    allowed_violations = int(np.floor(EPSILON * N_IN))
    R_alsox = A_sorted[allowed_violations]

    # CVaR conservative solution:
    # average of the worst epsilon-tail available reserves
    tail_size = int(np.ceil(EPSILON * N_IN))
    R_cvar = A_sorted[:tail_size].mean()

    # Reliability checks
    reliability_alsox = np.mean(A_in >= R_alsox)
    reliability_cvar = np.mean(A_in >= R_cvar)

    shortfall_alsox = np.mean(np.maximum(R_alsox - A_in, 0.0))
    shortfall_cvar = np.mean(np.maximum(R_cvar - A_in, 0.0))

    results = pd.DataFrame({
        "Method": ["ALSO-X", "CVaR"],
        "Reserve bid [kW]": [R_alsox, R_cvar],
        "In-sample reliability": [reliability_alsox, reliability_cvar],
        "In-sample violation rate": [
            1.0 - reliability_alsox,
            1.0 - reliability_cvar,
        ],
        "In-sample expected shortfall [kW]": [
            shortfall_alsox,
            shortfall_cvar,
        ],
    })

    return {
        "profiles": profiles,
        "profiles_in": profiles_in,
        "profiles_out": profiles_out,
        "A_in": A_in,
        "R_alsox": R_alsox,
        "R_cvar": R_cvar,
        "results": results,
    }




def plot_in_sample_profiles(
    profiles_in: np.ndarray,
    save_path: str | None = None,
):
    """
    Plot the 100 in-sample stochastic load profiles.
    """
    minutes = np.arange(1, profiles_in.shape[1] + 1)

    plt.figure(figsize=(10, 5))

    for profile in profiles_in:
        plt.plot(minutes, profile, linewidth=1, alpha=0.35)

    plt.xlabel("Minute")
    plt.ylabel("Load [kW]")
    plt.title("Task 2.1: In-sample stochastic load profiles")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()



if __name__ == "__main__":

    task21_output = run_task21(seed=42)

    profiles = task21_output["profiles"]
    profiles_in = task21_output["profiles_in"]
    profiles_out = task21_output["profiles_out"]

    A_in = task21_output["A_in"]
    R_alsox = task21_output["R_alsox"]
    R_cvar = task21_output["R_cvar"]
    results = task21_output["results"]

    print("Minimum load:", round(profiles.min(), 4))
    print("Maximum load:", round(profiles.max(), 4))
    print(
        "Maximum minute-to-minute change:",
        round(np.max(np.abs(np.diff(profiles, axis=1))), 4),
    )
    print()
    print(results.round(4))

    # Plot the 100 in-sample profiles
    plot_in_sample_profiles(profiles_in)