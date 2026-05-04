"""
Scenario generation for Step 1 (Tasks 1.1-1.4).

Builds 1600 combined scenarios = 20 wind × 20 price × 4 imbalance,
following the assignment specification:

- Wind: sample 20 historical days (without replacement) from the 365
  hourly capacity-factor days in DK01, scale by P_nom = 500 MW.
- Price: independently sample 20 historical days (without replacement)
  from the 365 daily DK2 day-ahead price profiles.
- Imbalance: generate 4 binary scenarios, each consisting of 24 i.i.d.
  Bernoulli draws with p estimated empirically from mFRR activations.
- Combine via Cartesian product, all scenarios equiprobable (1/1600).

The three sources are sampled with independent random seeds, consistent
with the assignment's "mutually uncorrelated" assumption.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from data_loader import RawData



# Output container

@dataclass
class ScenarioSet:
    """
    Container for the combined scenarios used in the offering problem.

    Attributes
    ----------
    wind : np.ndarray
        Wind production [MW], shape (n_scenarios, 24).
        Each row is one combined scenario, each column is one hour.
    price_da : np.ndarray
        Day-ahead market clearing price [EUR/MWh], shape (n_scenarios, 24).
    imbalance : np.ndarray
        Binary system imbalance signal, shape (n_scenarios, 24).
        SI = 1 means deficit, SI = 0 means surplus.
    probabilities : np.ndarray
        Per-scenario probability, shape (n_scenarios,). Uniform 1/n_scenarios.
    P_nom : float
        Nominal wind farm capacity [MW] used to scale the capacity factor.
    wind_unique : np.ndarray
        The n_wind unique daily wind trajectories before the Cartesian
        product, shape (n_wind, 24). Useful for plotting and diagnostics.
    price_unique : np.ndarray
        The n_price unique daily DA price trajectories, shape (n_price, 24).
    imbalance_unique : np.ndarray
        The n_imb unique binary imbalance trajectories, shape (n_imb, 24).
    """
    wind: np.ndarray
    price_da: np.ndarray
    imbalance: np.ndarray
    probabilities: np.ndarray
    P_nom: float = 500.0
    wind_unique: Optional[np.ndarray] = None
    price_unique: Optional[np.ndarray] = None
    imbalance_unique: Optional[np.ndarray] = None

    @property
    def n_scenarios(self) -> int:
        return self.wind.shape[0]

    @property
    def n_hours(self) -> int:
        return self.wind.shape[1]



# Helpers

def _reshape_to_daily(series: pd.Series) -> np.ndarray:
    """
    Reshape an hourly series of length 8760 into a (365, 24) matrix
    where each row is one calendar day (00:00 to 23:00) and each column
    is one hour of the day.
    """
    if len(series) != 8760:
        raise ValueError(f"Expected 8760 hours, got {len(series)}")
    return series.values.reshape(365, 24)


def estimate_imbalance_p(imbalance_df: pd.DataFrame) -> float:
    """
    Estimate the Bernoulli probability p = P(SI = 1 | mFRR activated)
    from the Energinet imbalance dataframe, using only hours in which
    mFRR was activated (either Up or Down).

    See report's "Caveat 4" for the rationale of excluding non-activated
    hours and discarding the ImbalanceMWh column for this purpose.
    """
    mask_up = imbalance_df["mFRRUpActBal"] > 0      # deficit
    mask_dn = imbalance_df["mFRRDownActBal"] > 0    # surplus

    n_deficit = mask_up.sum()
    n_surplus = mask_dn.sum()
    n_active = n_deficit + n_surplus

    if n_active == 0:
        raise ValueError("No mFRR-active hours found, cannot estimate p")

    return n_deficit / n_active



# Per-source scenario sampling

def sample_wind_scenarios(wind_cf: pd.Series,
                          n_scenarios: int = 20,
                          P_nom: float = 500.0,
                          seed: int = 43) -> np.ndarray:
    """
    Sample `n_scenarios` daily wind production trajectories (in MW)
    without replacement from the 365 historical days in `wind_cf`.

    Parameters
    ----------
    wind_cf : pd.Series
        Hourly capacity factor in [0, 1], length 8760.
    n_scenarios : int
        Number of daily scenarios to sample.
    P_nom : float
        Nominal wind farm capacity in MW used to convert capacity
        factor into MW.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_scenarios, 24), wind production in MW.
    """
    rng = np.random.default_rng(seed)
    daily = _reshape_to_daily(wind_cf)             # (365, 24) capacity factors
    chosen = rng.choice(365, size=n_scenarios, replace=False)
    return daily[chosen] * P_nom                   # (n_scenarios, 24) MW


def sample_price_scenarios(da_prices: pd.Series,
                           n_scenarios: int = 20,
                           seed: int = 85) -> np.ndarray:
    """
    Sample `n_scenarios` daily day-ahead price trajectories (EUR/MWh)
    without replacement from the 365 historical days in `da_prices`.

    Parameters
    ----------
    da_prices : pd.Series
        Hourly DA spot price in EUR/MWh, length 8760.
    n_scenarios : int
        Number of daily scenarios to sample.
    seed : int
        RNG seed for reproducibility, independent from the wind seed.

    Returns
    -------
    np.ndarray of shape (n_scenarios, 24), price in EUR/MWh.
    """
    rng = np.random.default_rng(seed)
    daily = _reshape_to_daily(da_prices)           # (365, 24) prices
    chosen = rng.choice(365, size=n_scenarios, replace=False)
    return daily[chosen]                           # (n_scenarios, 24) EUR/MWh


def generate_imbalance_scenarios(p: float,
                                 n_scenarios: int = 4,
                                 n_hours: int = 24,
                                 seed: int = 127) -> np.ndarray:
    """
    Generate `n_scenarios` synthetic system-imbalance trajectories,
    each composed of `n_hours` i.i.d. Bernoulli(p) draws.
    SI = 1 means deficit, SI = 0 means surplus.

    Parameters
    ----------
    p : float
        Probability of deficit per hour, in [0, 1].
    n_scenarios : int
        Number of binary scenarios to generate.
    n_hours : int
        Length of each scenario (24 for our daily problem).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_scenarios, n_hours) with values in {0, 1}.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")

    rng = np.random.default_rng(seed)
    return rng.binomial(n=1, p=p, size=(n_scenarios, n_hours))



# Combination via Cartesian product

def _cartesian_product(wind_scen: np.ndarray,
                       price_scen: np.ndarray,
                       imb_scen: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the full Cartesian product of three independent scenario
    matrices and return three arrays of shape (n_total, 24) where
    n_total = n_wind * n_price * n_imb.

    The combination order is (wind, price, imbalance), with imbalance
    varying fastest. This ordering is irrelevant for the optimization
    (scenarios are equiprobable and unordered) but kept consistent
    for reproducibility.
    """
    n_wind, _ = wind_scen.shape
    n_price, _ = price_scen.shape
    n_imb, _ = imb_scen.shape

    # broadcast indices for the cartesian product
    iw, ip, ii = np.meshgrid(
        np.arange(n_wind),
        np.arange(n_price),
        np.arange(n_imb),
        indexing="ij",
    )
    iw = iw.ravel()
    ip = ip.ravel()
    ii = ii.ravel()

    return wind_scen[iw], price_scen[ip], imb_scen[ii]



# Top-level builder

def build_scenarios(raw: RawData,
                    n_wind: int = 20,
                    n_price: int = 20,
                    n_imb: int = 4,
                    P_nom: float = 500.0,
                    master_seed: int = 42) -> ScenarioSet:
    """
    Top-level convenience function. Builds a ScenarioSet of size
    n_wind * n_price * n_imb (default 20 * 20 * 4 = 1600) by independently
    sampling each source and taking their Cartesian product.

    Parameters
    ----------
    raw : RawData
        Output of data_loader.load_all().
    n_wind, n_price, n_imb : int
        Number of scenarios per source. Total combined scenarios is the
        product. Assignment requires >= 1600.
    P_nom : float
        Nominal wind farm capacity in MW.
    master_seed : int
        Master RNG seed. Per-source seeds are derived deterministically
        from this. Changing master_seed produces a fresh, independent
        scenario realization (used in Task 1.4 to test sensitivity of
        risk-averse solutions to in-sample scenarios).

    Returns
    -------
    ScenarioSet
    """
    # Derive independent seeds for the three sources from the master seed
    seeds = {
        "wind":  master_seed * 1 + 1,
        "price": master_seed * 2 + 1,
        "imb":   master_seed * 3 + 1,
    }

    # Estimate p from the historical Energinet data
    p = estimate_imbalance_p(raw.imbalance)

    # Sample per-source scenarios
    wind_scen  = sample_wind_scenarios(raw.wind_cf,  n_wind,  P_nom, seeds["wind"])
    price_scen = sample_price_scenarios(raw.da_prices, n_price,        seeds["price"])
    imb_scen   = generate_imbalance_scenarios(p, n_imb, 24,             seeds["imb"])

    # Cartesian product
    wind_full, price_full, imb_full = _cartesian_product(wind_scen, price_scen, imb_scen)

    # Uniform probabilities
    n_total = wind_full.shape[0]
    probs = np.full(n_total, 1.0 / n_total)

    return ScenarioSet(
        wind=wind_full,
        price_da=price_full,
        imbalance=imb_full,
        probabilities=probs,
        P_nom=P_nom,
        wind_unique=wind_scen,
        price_unique=price_scen,
        imbalance_unique=imb_scen,
    )