# Assignment 2 — Renewables in Electricity Markets (46755)
**Group 39** | DTU, Spring 2026  
Raul Alcazar Martinez · Francesco Orma · Dimokritos Psomatakis · Nikolaos Gaitanidis

## Installation

Requires a valid Gurobi licence.

    pip install -r requirements.txt

## How to run

Open `notebooks/main.ipynb` and run all cells top to bottom. Each section is labelled by task (1.1 through 2.3) and reproduces the results in the report exactly.

## Repository structure

    assignment_2/
    ├── data/raw/
    │   ├── energinet_si_2024.csv       # mFRR imbalance data — Energinet
    │   ├── ninja_dk2_2024.csv          # Wind capacity factors — Renewables.ninja (DK01)
    │   └── nordpool_dk2_2024.csv       # Day-ahead prices — Nord Pool DK2
    ├── notebooks/
    │   └── main.ipynb                  # Main notebook — run this
    ├── src/
    │   ├── data_loader.py              # Loads raw data into RawData dataclass
    │   ├── reporting.py                # Plotting and results formatting
    │   ├── scenario_generation.py      # Builds 1600 combined scenarios
    │   ├── task1_1_offering_oneprice.py
    │   ├── task1_2_offering_twoprice.py
    │   ├── task1_3_cross_validation.py
    │   ├── task1_4_risk_averse.py
    │   ├── task2_1_fcrd_up.py
    │   ├── task2_2_verification.py
    │   └── task2_3_en_pers.py
    ├── requirements.txt
    └── README.md

## Reproducibility

Scenarios are generated from `master_seed = 42`, from which three independent per-source seeds are derived as follows:

| Source | Seed derivation | Seed value |
|---|---|---|
| Wind | `master_seed * 1 + 1` | 43 |
| Day-ahead price | `master_seed * 2 + 1` | 85 |
| Imbalance | `master_seed * 3 + 1` | 127 |

To test sensitivity to scenario sampling (Task 1.4), change `master_seed` at the top of `notebooks/main.ipynb` and re-run the CVaR section.