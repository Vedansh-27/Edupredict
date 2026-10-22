"""
EduPredict Cloud — Synthetic Student Performance Dataset Generator
==================================================================
This script generates a reproducible synthetic dataset that mirrors the
structure and statistical properties of the UCI Student Performance dataset.

The data is SYNTHETIC. It is not real student data and must not be
represented as such. It is used only because the UCI dataset may not
always be directly downloadable in automated CI/CD pipelines.

Dataset documentation:
  P. Cortez and A. Silva. Using Data Mining to Predict Secondary School
  Student Performance. In A. Brito and J. Teixeira Eds., Proceedings of
  5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12,
  Porto, Portugal, April 2008, EUROSIS, ISBN 978-9077381-39-7.
  http://www3.dsi.uminho.pt/pcortez/student.pdf

Target variable: G3 (final period grade, 0–20 scale)
Inputs used:     G1, G2, studytime, failures, absences, sex, age,
                 Medu, Fedu, famrel, freetime, goout, Dalc, Walc, health
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_SAMPLES = 649  # matches the math dataset size from UCI


def generate_dataset(n: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Demographic features ---
    sex = rng.choice(["M", "F"], size=n)
    age = rng.integers(15, 23, size=n)

    # --- Parental education (0=none … 4=higher) ---
    Medu = rng.integers(0, 5, size=n)
    Fedu = rng.integers(0, 5, size=n)

    # --- Study behaviour ---
    studytime = rng.integers(1, 5, size=n)          # 1=<2h … 4=>10h
    failures = rng.choice([0, 1, 2, 3], size=n,
                          p=[0.67, 0.18, 0.09, 0.06])

    # --- Home environment ---
    famrel   = rng.integers(1, 6, size=n)           # 1=very bad … 5=excellent
    freetime = rng.integers(1, 6, size=n)
    goout    = rng.integers(1, 6, size=n)

    # --- Lifestyle ---
    Dalc   = rng.integers(1, 6, size=n)             # weekday alcohol
    Walc   = rng.integers(1, 6, size=n)             # weekend alcohol
    health = rng.integers(1, 6, size=n)             # 1=very bad … 5=very good
    absences = rng.integers(0, 75, size=n)

    # --- Grades (correlated) ---
    # G1: first period grade – base around 11, influenced by studytime & failures
    G1_base = (
        10.5
        + studytime * 0.8
        - failures * 2.2
        + (Medu + Fedu) * 0.15
        - absences * 0.04
        + rng.normal(0, 2.0, size=n)
    )
    G1 = np.clip(G1_base, 0, 20).round().astype(int)

    # G2: second period grade – strongly correlated with G1
    G2_base = G1 + rng.normal(0, 1.5, size=n)
    G2 = np.clip(G2_base, 0, 20).round().astype(int)

    # G3: final grade – strongly correlated with G2, somewhat with G1
    G3_base = 0.6 * G2 + 0.3 * G1 + rng.normal(0, 1.2, size=n)
    # students who failed before are more likely to score 0
    G3_base -= failures * 1.0
    G3 = np.clip(G3_base, 0, 20).round().astype(int)

    df = pd.DataFrame({
        "sex":      sex,
        "age":      age,
        "Medu":     Medu,
        "Fedu":     Fedu,
        "studytime": studytime,
        "failures": failures,
        "famrel":   famrel,
        "freetime": freetime,
        "goout":    goout,
        "Dalc":     Dalc,
        "Walc":     Walc,
        "health":   health,
        "absences": absences,
        "G1":       G1,
        "G2":       G2,
        "G3":       G3,
    })

    return df


def main():
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "student_performance.csv"

    df = generate_dataset()
    df.to_csv(out_path, index=False)
    print(f"[INFO] Synthetic dataset written to: {out_path}")
    print(f"[INFO] Shape: {df.shape}")
    print(df.describe())


if __name__ == "__main__":
    main()
