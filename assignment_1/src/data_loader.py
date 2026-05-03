import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class RTS24Data:
    buses: List[int]
    slack_bus: int
    generators: pd.DataFrame
    lines: pd.DataFrame
    system_load: pd.DataFrame
    nodal_load: pd.DataFrame
    demand_bids: pd.DataFrame
    wind_farms: pd.DataFrame
    wind_availability: pd.DataFrame


def load_rts24_from_csv(data_dir: str | Path, slack_bus: int = 13) -> RTS24Data:
    """
    Load RTS 24-bus dataset from CSV files (source of truth).
    """

    data_dir = Path(data_dir)

    # --- Read CSVs ---
    generators = pd.read_csv(data_dir / "generators.csv")
    system_load = pd.read_csv(data_dir / "system_load_24h.csv")
    nodal_load = pd.read_csv(data_dir / "nodal_load_24h.csv")
    lines = pd.read_csv(data_dir / "lines.csv")
    wind_farms = pd.read_csv(data_dir / "wind_farms.csv")
    wind_availability = pd.read_csv(data_dir / "wind_availability_24h.csv")
    demand_bids = pd.read_csv(data_dir / "demand_bids_24h.csv")

    buses = list(range(1, 25))

    return RTS24Data(
        buses=buses,
        slack_bus=slack_bus,
        generators=generators,
        lines=lines,
        system_load=system_load,
        nodal_load=nodal_load,
        demand_bids=demand_bids,
        wind_farms=wind_farms,
        wind_availability=wind_availability,
    )