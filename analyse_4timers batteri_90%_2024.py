import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data 2024")
OUT_DIR = Path("data/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_FILES = [
    DATA_DIR / "spotpriser_tyskland_2024_riktig.csv",
]

SOLAR_FILES = [
    DATA_DIR / "solproduksjon_tyskland_2024_riktig.csv",
]

SOLAR_PEAK_MW = 1.0

BAT_P_MW = 1.0
BAT_E_MWH = 4.0
ETA_RT = 0.90
ETA_C = np.sqrt(ETA_RT)
ETA_D = np.sqrt(ETA_RT)

DS = 0.05
SOC_GRID = np.round(np.arange(0, BAT_E_MWH + DS / 2, DS), 3)


def fmt_num(x, nd=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "-"
    s = f"{x:,.{nd}f}"
    return s.replace(",", " ").replace(".", ",")


def print_year_report(row: dict, title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Antall timer:              {int(row['timer'])}")
    print(f"Solproduksjon:             {fmt_num(row['sol_mwh'], 1)} MWh")
    print(f"Gjennomsnittlig spotpris:  {fmt_num(row['avg_spot_eur_mwh'], 2)} EUR/MWh")

    print("\nUTEN BATTERI")
    print(f"  Total inntekt:           {fmt_num(row['inntekt_uten_batt'], 2)} EUR")
    print(f"  Capture price:           {fmt_num(row['capture_price_uten_batt'], 2)} EUR/MWh")
    print(f"  Capture rate:            {fmt_num(row['capture_rate_uten_batt'], 3)}")

    print(f"\nMED BATTERI ({BAT_P_MW:.0f} MW / {BAT_E_MWH:.0f} MWh)")
    print(f"  Total inntekt:           {fmt_num(row['inntekt_med_batt'], 2)} EUR")
    print(f"  Capture price:           {fmt_num(row['capture_price_med_batt'], 2)} EUR/MWh")
    print(f"  Capture rate:            {fmt_num(row['capture_rate_med_batt'], 3)}")

    print("\nEFFEKT AV BATTERI")
    print(f"  Ekstra inntekt:          {fmt_num(row['ekstra_inntekt'], 2)} EUR")
    print(f"  Løft i CR:               {fmt_num(row['lift_pct_points'], 2)} prosentpoeng")
    print(f"  Ekstra inntekt per MWh:  {fmt_num(row['ekstra_inntekt_eur_per_mwh_sol'], 2)} EUR/MWh")
    print(f"  Energi inn i batteri:    {fmt_num(row['charged_mwh'], 1)} MWh")
    print(f"  Energi levert fra batt:  {fmt_num(row['delivered_from_batt_mwh'], 1)} MWh")
    print("=" * 70)


def print_quarter_table(df: pd.DataFrame):
    dfq = df[df["case"].isin(["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"])].copy()
    order = ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"]
    dfq["case"] = pd.Categorical(dfq["case"], categories=order, ordered=True)
    dfq = dfq.sort_values("case")

    header = f"{'Periode':<8} {'Sol MWh':>10} {'Spot (EUR/MWh)':>14} {'Uten batt (EUR)':>16} {'Med batt (EUR)':>16} {'Ekstra (EUR)':>13} {'CR uten':>8} {'CR med':>8}"
    print(header)
    print("-" * len(header))

    for _, r in dfq.iterrows():
        print(
            f"{r['case'][:2]:<8} "
            f"{r['sol_mwh']:>10.1f} "
            f"{r['avg_spot_eur_mwh']:>14.2f} "
            f"{r['inntekt_uten_batt']:>16.2f} "
            f"{r['inntekt_med_batt']:>16.2f} "
            f"{r['ekstra_inntekt']:>13.2f} "
            f"{r['capture_rate_uten_batt']:>8.3f} "
            f"{r['capture_rate_med_batt']:>8.3f}"
        )


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fant ikke fil: {path}")

    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep=";")

    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")

    df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    return df


def find_time_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)

    for c in ["time", "Time", "Hour", "#Hour", "Datetime", "datetime", "Date", "date"]:
        if c in cols:
            return c

    lowered = {c.lower(): c for c in cols}
    for key in ["hour", "time", "date", "datetime", "#hour"]:
        for low, orig in lowered.items():
            if key in low:
                return orig

    raise ValueError(f"Fant ingen tidkolonne. Kolonner: {cols}")


def parse_time_series(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, errors="coerce", dayfirst=True)

    if t.isna().mean() > 0.5:
        t2 = pd.to_datetime(s, errors="coerce")
        if t2.isna().mean() < t.isna().mean():
            t = t2

    return t


def read_series_from_files(files: list[Path], value_col: str) -> pd.DataFrame:
    parts = []

    for f in files:
        df = read_csv_flexible(f)
        tc = find_time_column(df)

        if value_col not in df.columns:
            raise ValueError(f"Mangler {value_col} i {f.name}. Kolonner: {df.columns.tolist()}")

        out = df[[tc, value_col]].copy()
        out["time"] = parse_time_series(out[tc]).dt.floor("h")
        out = out.dropna(subset=["time"])
        out = out[["time", value_col]]
        parts.append(out)

    out = pd.concat(parts, ignore_index=True)
    out = out.groupby("time", as_index=False)[value_col].mean()
    out = out.sort_values("time").reset_index(drop=True)
    return out


def build_year_df_2024() -> pd.DataFrame:
    price = read_series_from_files(PRICE_FILES, "SPOTDE")
    solar = read_series_from_files(SOLAR_FILES, "PRODESOL")

    df = pd.merge(price, solar, on="time", how="inner").sort_values("time").reset_index(drop=True)

    start = pd.Timestamp("2024-01-01 00:00:00")
    end = pd.Timestamp("2025-01-01 00:00:00")

    df = df[(df["time"] >= start) & (df["time"] < end)].copy().reset_index(drop=True)
    return df


def scale_solar_to_1mw_peak(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    peak = float(df2["PRODESOL"].max())

    if peak <= 0:
        raise ValueError("PRODESOL peak <= 0. Sjekk soldata.")

    df2["PRODESOL"] = df2["PRODESOL"] * (SOLAR_PEAK_MW / peak)
    return df2


def baseline_metrics(df: pd.DataFrame) -> dict:
    p = df["SPOTDE"].to_numpy(float)
    q = df["PRODESOL"].to_numpy(float)

    rev = float(np.sum(p * q))
    avg_spot = float(np.mean(p))
    sol_mwh = float(np.sum(q))

    cap_price = float(np.sum(p * q) / sol_mwh) if sol_mwh > 0 else float("nan")
    cap_rate = float(cap_price / avg_spot) if avg_spot != 0 else float("nan")

    return {
        "sol_mwh": sol_mwh,
        "avg_spot": avg_spot,
        "revenue": rev,
        "capture_price": cap_price,
        "capture_rate": cap_rate,
    }


def optimal_battery_dp(df: pd.DataFrame) -> tuple[float, np.ndarray]:
    p = df["SPOTDE"].to_numpy(float)
    q = df["PRODESOL"].to_numpy(float)
    T = len(df)

    soc = SOC_GRID
    nS = len(soc)

    v_next = np.zeros(nS)
    policy = np.zeros((T, nS), dtype=np.int16)

    for t in range(T - 1, -1, -1):
        price = p[t]
        solar = q[t]

        v_cur = np.full(nS, -1e18)
        best_next = np.zeros(nS, dtype=np.int16)

        for si in range(nS):
            s = soc[si]

            ch_max = min(BAT_P_MW, solar, (BAT_E_MWH - s) / ETA_C if ETA_C > 0 else 0.0)
            ch_max = max(ch_max, 0.0)

            dis_max = min(BAT_P_MW, s)
            dis_max = max(dis_max, 0.0)

            s_min = s - dis_max
            s_max = s + ch_max * ETA_C

            i_min = max(int(np.ceil(s_min / DS - 1e-9)), 0)
            i_max = min(int(np.floor(s_max / DS + 1e-9)), nS - 1)

            for sj2 in range(i_min, i_max + 1):
                s2 = soc[sj2]
                delta = s2 - s

                if delta >= 0:
                    charge = delta / ETA_C if ETA_C > 0 else 0.0
                    discharge = 0.0
                else:
                    charge = 0.0
                    discharge = -delta

                if charge > solar + 1e-9:
                    continue
                if charge > BAT_P_MW + 1e-9 or discharge > BAT_P_MW + 1e-9:
                    continue

                market = (solar - charge) + discharge * ETA_D
                rev = price * market + v_next[sj2]

                if rev > v_cur[si]:
                    v_cur[si] = rev
                    best_next[si] = sj2

        policy[t] = best_next
        v_next = v_cur

    total_rev = float(v_next[0])

    soc_path = np.zeros(T + 1)
    si = 0

    for t in range(T):
        si = int(policy[t, si])
        soc_path[t + 1] = soc[si]

    return total_rev, soc_path


def battery_metrics(df: pd.DataFrame) -> dict:
    total_rev, soc_path = optimal_battery_dp(df)

    p = df["SPOTDE"].to_numpy(float)
    solar = df["PRODESOL"].to_numpy(float)
    T = len(df)

    market = np.zeros(T)
    charge = np.zeros(T)
    discharge = np.zeros(T)

    for t in range(T):
        s = soc_path[t]
        s2 = soc_path[t + 1]
        delta = s2 - s

        if delta >= 0:
            charge[t] = delta / ETA_C if ETA_C > 0 else 0.0
            discharge[t] = 0.0
        else:
            charge[t] = 0.0
            discharge[t] = -delta

        market[t] = (solar[t] - charge[t]) + discharge[t] * ETA_D

    avg_spot = float(np.mean(p))
    total_market = float(np.sum(market))

    cap_price = float(np.sum(p * market) / total_market) if total_market > 0 else float("nan")
    cap_rate = float(cap_price / avg_spot) if avg_spot != 0 else float("nan")

    charged_mwh = float(np.sum(charge))
    delivered_mwh = float(np.sum(discharge * ETA_D))

    return {
        "revenue": float(total_rev),
        "capture_price": cap_price,
        "capture_rate": cap_rate,
        "charged_mwh": charged_mwh,
        "delivered_from_batt_mwh": delivered_mwh,
    }


def run_case(df_period: pd.DataFrame, label: str) -> dict:
    base = baseline_metrics(df_period)
    bat = battery_metrics(df_period)

    extra = bat["revenue"] - base["revenue"]
    extra_per_mwh = extra / base["sol_mwh"] if base["sol_mwh"] > 0 else float("nan")

    return {
        "case": label,
        "timer": len(df_period),
        "sol_mwh": base["sol_mwh"],
        "avg_spot_eur_mwh": base["avg_spot"],
        "inntekt_uten_batt": base["revenue"],
        "inntekt_med_batt": bat["revenue"],
        "ekstra_inntekt": extra,
        "ekstra_inntekt_eur_per_mwh_sol": extra_per_mwh,
        "capture_price_uten_batt": base["capture_price"],
        "capture_price_med_batt": bat["capture_price"],
        "capture_rate_uten_batt": base["capture_rate"],
        "capture_rate_med_batt": bat["capture_rate"],
        "lift_pct_points": (bat["capture_rate"] - base["capture_rate"]) * 100.0,
        "charged_mwh": bat["charged_mwh"],
        "delivered_from_batt_mwh": bat["delivered_from_batt_mwh"],
    }


def main():
    print("=== Bygger helår 2024 fra pris + sol ===")

    df_2024 = build_year_df_2024()

    print("Timer i merged helår:", len(df_2024))
    print("Min/max:", df_2024["time"].min(), df_2024["time"].max())

    df_2024 = scale_solar_to_1mw_peak(df_2024)

    print("Sol peak etter skalering:", float(df_2024["PRODESOL"].max()))
    print("Sol MWh etter skalering:", float(df_2024["PRODESOL"].sum()))

    periods = [
        ("Q1 2024", "2024-01-01 00:00:00", "2024-04-01 00:00:00"),
        ("Q2 2024", "2024-04-01 00:00:00", "2024-07-01 00:00:00"),
        ("Q3 2024", "2024-07-01 00:00:00", "2024-10-01 00:00:00"),
        ("Q4 2024", "2024-10-01 00:00:00", "2025-01-01 00:00:00"),
        ("Hele 2024", "2024-01-01 00:00:00", "2025-01-01 00:00:00"),
    ]

    results = []

    for label, start, end in periods:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)

        df_p = df_2024[(df_2024["time"] >= s) & (df_2024["time"] < e)].copy().reset_index(drop=True)
        results.append(run_case(df_p, label))

    summary_df = pd.DataFrame(results)

    order_case = ["Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024", "Hele 2024"]
    summary_df["case_order"] = summary_df["case"].map({c: i for i, c in enumerate(order_case)})
    summary_df = summary_df.sort_values("case_order").drop(columns=["case_order"]).reset_index(drop=True)

    out_path = OUT_DIR / "summary_battery_cases_2024_1mw_4h_eta90.csv"
    summary_df.to_csv(out_path, index=False)

    print("\nLagret CSV:", out_path)

    year_row = summary_df[summary_df["case"] == "Hele 2024"].iloc[0].to_dict()

    print_year_report(
        year_row,
        "BASE CASE: 1 MW sol (peak) + 1 MW / 4 MWh batteri | η = 90 % | Hele 2024"
    )

    print_quarter_table(summary_df)


if __name__ == "__main__":
    main()