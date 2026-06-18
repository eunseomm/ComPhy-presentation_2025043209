"""
Astrophage Solar Simulation
---------------------------
This project uses KMA ASOS daily observation data to simulate
solar energy attenuation inspired by the fictional "astrophage" idea.

Data source:
- Korea Meteorological Administration / Public Data Portal
- ASOS daily observation API
- Main variables:
  - sumGsr: daily total solar radiation
  - sumSsHr: daily sunshine duration

Important:
Do NOT upload your API key to GitHub.
Put your key in a local .env file or enter it when running the program.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


API_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"

STATION_IDS = {
    "Seoul": "108",
    "Busan": "159",
    "Daegu": "143",
    "Gwangju": "156",
    "Daejeon": "133",
    "Jeju": "184",
}


def load_env_file(env_path: str = ".env") -> None:
    """
    Very small .env loader.
    Example .env file:
        KMA_SERVICE_KEY=your_api_key_here
    """
    path = Path(env_path)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_service_key(cli_key: Optional[str] = None) -> str:
    """
    Priority:
    1. command line argument
    2. .env file
    3. environment variable
    """
    load_env_file()

    service_key = cli_key or os.getenv("KMA_SERVICE_KEY")
    if not service_key:
        raise ValueError(
            "KMA service key is missing. "
            "Create a .env file with KMA_SERVICE_KEY=your_key_here "
            "or run with --service-key your_key_here."
        )
    return service_key


def fetch_kma_asos_daily(
    service_key: str,
    start_date: str,
    end_date: str,
    stn_id: str = "108",
    rows_per_page: int = 999,
) -> pd.DataFrame:
    """
    Fetch daily ASOS data from the KMA Public Data Portal API.

    Parameters
    ----------
    service_key:
        API key from data.go.kr.
    start_date:
        Start date in YYYYMMDD format.
    end_date:
        End date in YYYYMMDD format.
    stn_id:
        ASOS station ID. Seoul is 108.
    rows_per_page:
        Number of rows per page.

    Returns
    -------
    pandas.DataFrame
        Raw daily ASOS data.
    """
    all_items: List[Dict] = []
    page_no = 1

    while True:
        params = {
            "serviceKey": service_key,
            "pageNo": page_no,
            "numOfRows": rows_per_page,
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "DAY",
            "startDt": start_date,
            "endDt": end_date,
            "stnIds": stn_id,
        }

        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        header = data["response"]["header"]
        result_code = header.get("resultCode")
        result_msg = header.get("resultMsg")

        if result_code != "00":
            raise RuntimeError(f"KMA API error: {result_code} / {result_msg}")

        body = data["response"]["body"]
        total_count = int(body.get("totalCount", 0))

        items = body.get("items", {}).get("item", [])
        if isinstance(items, dict):
            items = [items]

        all_items.extend(items)

        if len(all_items) >= total_count or not items:
            break

        page_no += 1

    return pd.DataFrame(all_items)


def clean_kma_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select useful variables and convert them to numeric values.
    """
    required_columns = ["tm", "stnId", "stnNm", "sumGsr", "sumSsHr", "avgTa", "avgTca"]
    existing_columns = [col for col in required_columns if col in raw_df.columns]
    df = raw_df[existing_columns].copy()

    df = df.rename(
        columns={
            "tm": "date",
            "stnId": "station_id",
            "stnNm": "station_name",
            "sumGsr": "solar_radiation",
            "sumSsHr": "sunshine_duration",
            "avgTa": "mean_temperature",
            "avgTca": "mean_cloud_cover",
        }
    )

    df["date"] = pd.to_datetime(df["date"])

    numeric_columns = [
        "solar_radiation",
        "sunshine_duration",
        "mean_temperature",
        "mean_cloud_cover",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # This project mainly needs solar radiation.
    df = df.dropna(subset=["solar_radiation"])

    return df


def make_monthly_data(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily data into monthly mean values.
    Monthly means make the climate trend easier to present in 5 minutes.
    """
    monthly = (
        clean_df
        .set_index("date")
        .resample("MS")
        .agg(
            solar_radiation_mean=("solar_radiation", "mean"),
            solar_radiation_sum=("solar_radiation", "sum"),
            sunshine_duration_mean=("sunshine_duration", "mean"),
            mean_temperature=("mean_temperature", "mean"),
            mean_cloud_cover=("mean_cloud_cover", "mean"),
        )
        .reset_index()
    )

    monthly["month_index"] = np.arange(len(monthly))
    return monthly


def simulate_astrophage_attenuation(
    monthly_df: pd.DataFrame,
    rates: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Simulate solar radiation attenuation.

    Model:
        E_simulated(t) = E_observed(t) * exp(-lambda * t)

    Here, t is the month index and lambda is the attenuation rate per month.
    """
    if rates is None:
        rates = [0.000, 0.003, 0.006, 0.010]

    result = monthly_df[["date", "month_index", "solar_radiation_mean"]].copy()

    for rate in rates:
        label = f"lambda_{rate:.3f}"
        attenuation_factor = np.exp(-rate * result["month_index"])
        result[label] = result["solar_radiation_mean"] * attenuation_factor

    return result


def summarize_energy_loss(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate total solar energy loss percentage for each scenario.
    """
    original_total = sim_df["solar_radiation_mean"].sum()

    rows = []
    for col in sim_df.columns:
        if col.startswith("lambda_"):
            simulated_total = sim_df[col].sum()
            loss_percent = (1 - simulated_total / original_total) * 100
            rows.append(
                {
                    "scenario": col,
                    "total_original": original_total,
                    "total_simulated": simulated_total,
                    "energy_loss_percent": loss_percent,
                }
            )

    return pd.DataFrame(rows)


def simulate_aerosol_transmission(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Additional simple climate-engineering scenario.

    This is not astrophage growth. It is a constant atmospheric transmission model:
        E_surface = E_observed * T

    T = 0.98 means 2% reduction of incoming surface solar energy.
    """
    transmission_values = [1.00, 0.98, 0.95, 0.90]

    result = monthly_df[["date", "solar_radiation_mean"]].copy()

    for t in transmission_values:
        label = f"transmission_{t:.2f}"
        result[label] = result["solar_radiation_mean"] * t

    return result


def plot_original_monthly(monthly_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_df["date"], monthly_df["solar_radiation_mean"], marker="o")
    plt.xlabel("Date")
    plt.ylabel("Monthly Mean Solar Radiation")
    plt.title("Observed Monthly Mean Solar Radiation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_astrophage_simulation(sim_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(
        sim_df["date"],
        sim_df["solar_radiation_mean"],
        label="Observed",
        linewidth=2,
    )

    for col in sim_df.columns:
        if col.startswith("lambda_"):
            plt.plot(sim_df["date"], sim_df[col], label=col)

    plt.xlabel("Date")
    plt.ylabel("Monthly Mean Solar Radiation")
    plt.title("Astrophage Solar Energy Attenuation Simulation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_energy_loss(summary_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["scenario"], summary_df["energy_loss_percent"])
    plt.xlabel("Scenario")
    plt.ylabel("Total Energy Loss (%)")
    plt.title("Cumulative Solar Energy Loss by Scenario")
    plt.xticks(rotation=30)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_sunshine_scatter(clean_df: pd.DataFrame, output_path: Path) -> None:
    if "sunshine_duration" not in clean_df.columns:
        return

    scatter_df = clean_df.dropna(subset=["solar_radiation", "sunshine_duration"])

    plt.figure(figsize=(7, 5))
    plt.scatter(
        scatter_df["sunshine_duration"],
        scatter_df["solar_radiation"],
        alpha=0.5,
    )
    plt.xlabel("Sunshine Duration")
    plt.ylabel("Daily Solar Radiation")
    plt.title("Relationship Between Sunshine Duration and Solar Radiation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_project(args: argparse.Namespace) -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    figure_dir = Path("figures")

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    station_id = args.station_id
    raw_path = raw_dir / f"kma_asos_{station_id}_{args.start_date}_{args.end_date}.csv"
    clean_path = processed_dir / "clean_daily_solar_data.csv"
    monthly_path = processed_dir / "monthly_solar_data.csv"
    sim_path = processed_dir / "astrophage_simulation_result.csv"
    aerosol_path = processed_dir / "aerosol_transmission_result.csv"
    summary_path = processed_dir / "energy_loss_summary.csv"

    if args.skip_fetch and raw_path.exists():
        print(f"Loading existing raw data: {raw_path}")
        raw_df = pd.read_csv(raw_path)
    else:
        service_key = get_service_key(args.service_key)
        print("Fetching KMA ASOS daily data...")
        raw_df = fetch_kma_asos_daily(
            service_key=service_key,
            start_date=args.start_date,
            end_date=args.end_date,
            stn_id=station_id,
        )
        raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
        print(f"Saved raw data: {raw_path}")

    clean_df = clean_kma_data(raw_df)
    clean_df.to_csv(clean_path, index=False, encoding="utf-8-sig")

    monthly_df = make_monthly_data(clean_df)
    monthly_df.to_csv(monthly_path, index=False, encoding="utf-8-sig")

    sim_df = simulate_astrophage_attenuation(monthly_df)
    sim_df.to_csv(sim_path, index=False, encoding="utf-8-sig")

    aerosol_df = simulate_aerosol_transmission(monthly_df)
    aerosol_df.to_csv(aerosol_path, index=False, encoding="utf-8-sig")

    summary_df = summarize_energy_loss(sim_df)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plot_original_monthly(monthly_df, figure_dir / "01_observed_monthly_solar.png")
    plot_astrophage_simulation(sim_df, figure_dir / "02_astrophage_simulation.png")
    plot_energy_loss(summary_df, figure_dir / "03_energy_loss_summary.png")
    plot_sunshine_scatter(clean_df, figure_dir / "04_sunshine_vs_solar_radiation.png")

    print("\nProject finished.")
    print(f"Raw data: {raw_path}")
    print(f"Processed data: {processed_dir}")
    print(f"Figures: {figure_dir}")
    print("\nEnergy loss summary:")
    print(summary_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Astrophage solar energy attenuation simulation using KMA ASOS data."
    )

    parser.add_argument(
        "--service-key",
        type=str,
        default=None,
        help="KMA API service key. Do not upload this key to GitHub.",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default="20200101",
        help="Start date in YYYYMMDD format.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default="20241231",
        help="End date in YYYYMMDD format.",
    )

    parser.add_argument(
        "--station-id",
        type=str,
        default="108",
        help="KMA ASOS station ID. Seoul is 108.",
    )

    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Use existing raw CSV file instead of fetching API data again.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_project(args)
