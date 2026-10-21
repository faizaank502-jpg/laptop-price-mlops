# Step 11: Data cleaning script – src/data_prep.py
import re
import glob
import numpy as np
import pandas as pd


def _storage_gb(text, kind):
    """Total GB of one storage type (SSD / HDD) from text like '128GB SSD +  1TB HDD'."""
    total = 0.0
    for part in str(text).split("+"):
        part = part.strip()
        if kind in part:
            m = re.search(r"([\d.]+)\s*(GB|TB)", part)
            if m:
                size = float(m.group(1))
                total += size * 1000 if m.group(2) == "TB" else size
    return total


def _cpu_brand(text):
    text = str(text)
    if "Intel Core i7" in text:
        return "Intel Core i7"
    if "Intel Core i5" in text:
        return "Intel Core i5"
    if "Intel Core i3" in text:
        return "Intel Core i3"
    if text.startswith("Intel"):
        return "Other Intel"
    return "AMD"


def _os_group(text):
    text = str(text)
    if text.startswith("Windows"):
        return "Windows"
    if text in ("macOS", "Mac OS X"):
        return "Mac"
    return "Other/Linux/No OS"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Support both popular Kaggle versions of the dataset
    if "Price_euros" in df.columns:
        df = df.rename(columns={"Price_euros": "Price"})
    df = df.drop(columns=[c for c in df.columns
                          if c.startswith("Unnamed") or c in ("laptop_ID", "Product")])

    df["Ram"] = pd.to_numeric(df["Ram"].astype(str).str.replace("GB", ""), errors="coerce")
    df["Weight"] = pd.to_numeric(df["Weight"].astype(str).str.replace("kg", ""), errors="coerce")
    df["Inches"] = pd.to_numeric(df["Inches"], errors="coerce")

    screen = df["ScreenResolution"].astype(str)
    df["Touchscreen"] = screen.str.contains("Touchscreen").astype(int)
    df["IPS"] = screen.str.contains("IPS").astype(int)
    res = screen.str.extract(r"(\d+)x(\d+)").astype(float)
    df["PPI"] = np.sqrt(res[0] ** 2 + res[1] ** 2) / df["Inches"]

    df["Cpu_brand"] = df["Cpu"].apply(_cpu_brand)
    df["SSD"] = df["Memory"].apply(lambda x: _storage_gb(x, "SSD") + _storage_gb(x, "Flash"))
    df["HDD"] = df["Memory"].apply(lambda x: _storage_gb(x, "HDD"))
    df["Gpu_brand"] = df["Gpu"].astype(str).str.split().str[0]
    df["OS"] = df["OpSys"].apply(_os_group)

    df = df.drop(columns=["ScreenResolution", "Cpu", "Memory", "Gpu", "OpSys", "Inches"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw_path = glob.glob("data/raw/*.csv")[0]
    raw = pd.read_csv(raw_path, encoding="latin-1")
    out = clean(raw)
    out.to_csv("data/processed/laptops_clean.csv", index=False)
    print(f"Cleaned {len(raw)} rows -> {len(out)} rows")
    print("Saved data/processed/laptops_clean.csv")
