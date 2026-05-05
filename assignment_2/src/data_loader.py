"""
Data loader for Assignment 2.

Reads the three raw CSV files (Renewables.ninja wind capacity factor,
Nord Pool DK2 day-ahead prices, Energinet system imbalance) and returns
a clean RawData container holding three pandas objects, each indexed
on a uniform 365-day, 8760-hour calendar grid for the year 2024.

Calendar alignment note:
    2024 is a leap year (8784 hours), but the Nord Pool DK2 file we
    downloaded only covers Jan 1 - Dec 30 (8760 hours, Feb 29 included).
    To keep all three datasets on the same shape, we drop Dec 31 from the
    Renewables.ninja and Energinet files. Losing one day has no practical
    impact: scenarios are sampled at random from 365 daily trajectories.

This module is purely I/O: no scenario generation, no SI binarization,
no scaling. Those transformations live in scenario_generation.py.
"""

import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass
class RawData:
    """
    Cleaned raw inputs for the stochastic offering problem.

    Dataset descriptions:

    wind_cf : pd.Series
        Hourly bias-corrected wind capacity factor [0, 1] for the
        DK01 NUTS-2 region (within the DK2 price zone).
        Length 8760, indexed by hourly UTC timestamps.
    da_prices : pd.Series
        Hourly day-ahead spot price [EUR/MWh] for the DK2 price zone.
        Length 8760, indexed by hourly HourDK timestamps.
    imbalance : pd.DataFrame
        Hourly system imbalance signals for the DK2 price zone, with
        columns ['mFRRUpActBal', 'mFRRDownActBal', 'ImbalanceMWh'].
        Length 8760, indexed by hourly HourDK timestamps.
    """
    wind_cf: pd.Series
    da_prices: pd.Series
    imbalance: pd.DataFrame



# Helpers

def _expected_hours(year: int) -> int:
    """
    Expected number of hours per dataset for Assignment 2.
    Hard-coded to 8760, the size of the Nord Pool DK2 2024 file
    (Jan 1 - Dec 30, includes Feb 29). All datasets are aligned
    to this calendar grid.
    """
    return 8760



# Individual loaders


def load_wind_capacity_factor(path: str | Path,
                              region: str = "DK01",
                              year: int = 2024) -> pd.Series:
    """
    Load Renewables.ninja country-zones wind file and return the hourly
    bias-corrected capacity factor for the requested NUTS-2 region and year.

    The file has 3 metadata lines before the actual header. Time is in UTC.
    For consistency with the Nord Pool DK2 file (which covers Jan 1 - Dec 30,
    365 days including Feb 29), we drop Dec 31 if present.
    """
    df = pd.read_csv(path, skiprows=3, parse_dates=["time"])
    df = df.set_index("time")

    if region not in df.columns:
        raise KeyError(f"Region '{region}' not in file. Available: {list(df.columns)}")

    series = df[region]
    series = series[series.index.year == year]

    # Drop Dec 31 to align with the Nord Pool DK2 file (which ends on Dec 30)
    series = series[~((series.index.month == 12) & (series.index.day == 31))]
    series.name = "wind_cf"

    expected = _expected_hours(year)
    if len(series) != expected:
        raise ValueError(f"Expected {expected} hours for {year}, got {len(series)}")

    return series


def load_da_prices(path: str | Path,
                   price_area: str = "DK2",
                   year: int = 2024) -> pd.Series:
    """
    Load Nord Pool spot prices and return hourly EUR/MWh for the requested
    price area and year. Indexed on HourDK (local Danish time).

    The 2024 DK2 file covers Jan 1 - Dec 30 (8760 hours, includes Feb 29).
    No drop is needed: the file already has the target shape.
    """
    df = pd.read_csv(path,
                     sep=";",
                     decimal=",",
                     parse_dates=["HourUTC", "HourDK"])

    df = df[df["PriceArea"] == price_area].copy()
    df = df.set_index("HourDK").sort_index()

    series = df["SpotPriceEUR"]
    series = series[series.index.year == year]
    series.name = "da_price_eur"

    expected = _expected_hours(year)
    if len(series) != expected:
        raise ValueError(f"Expected {expected} hours for {year}, got {len(series)}")

    return series


def load_imbalance(path: str | Path,
                   price_area: str = "DK2",
                   year: int = 2024) -> pd.DataFrame:
    """
    Load Energinet 'Regulating and Balance Power, Overall Data' file and
    return imbalance-related columns for the requested price area and year.
    Indexed on HourDK.

    For consistency with the Nord Pool DK2 file (Jan 1 - Dec 30), we drop
    Dec 31 if present.
    """
    df = pd.read_csv(path,
                     sep=";",
                     decimal=",",
                     parse_dates=["HourUTC", "HourDK"])

    df = df[df["PriceArea"] == price_area].copy()
    df = df.set_index("HourDK").sort_index()

    cols = ["mFRRUpActBal", "mFRRDownActBal", "ImbalanceMWh"]
    df = df[cols]
    df = df[df.index.year == year]

    # Drop Dec 31 to align with the Nord Pool DK2 file
    df = df[~((df.index.month == 12) & (df.index.day == 31))]

    expected = _expected_hours(year)
    if len(df) != expected:
        raise ValueError(f"Expected {expected} hours for {year}, got {len(df)}")

    return df



# Loader for main script


def load_all(data_dir: str | Path = "data/raw",
             year: int = 2024) -> RawData:
    """
    Convenience wrapper: load all three raw datasets and return a RawData.

    Parameters
    ----------
    data_dir : path
        Directory containing the three raw CSV files.
    year : int
        Reference year (default 2024). All three series are aligned
        to a uniform 8760-hour calendar grid (Jan 1 - Dec 30).
    """
    data_dir = Path(data_dir)

    wind_cf = load_wind_capacity_factor(
        data_dir / "ninja_dk2_2024.csv", region="DK01", year=year
    )
    da_prices = load_da_prices(
        data_dir / "nordpool_dk2_2024.csv", price_area="DK2", year=year
    )
    imbalance = load_imbalance(
        data_dir / "energinet_si_2024.csv", price_area="DK2", year=year
    )

    return RawData(wind_cf=wind_cf, da_prices=da_prices, imbalance=imbalance)