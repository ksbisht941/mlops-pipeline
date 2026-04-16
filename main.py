import pandas as pd
import os


# ── Single-responsibility helpers ─────────────────────────────────────────────

def create_base_data() -> pd.DataFrame:
    """Return the initial product DataFrame (V1)."""
    data = {
        "product_id":   ["P001", "P002", "P003"],
        "product_name": ["Laptop", "Headphones", "Mouse"],
        "category":     ["Electronics", "Electronics", "Accessories"],
        "transaction":  ["sell", "purchase", "sell"],
        "quantity":     [2, 5, 10],
        "unit_price":   [999.99, 49.99, 19.99],
        "total_amount": [1999.98, 249.95, 199.90],
    }
    return pd.DataFrame(data)


def append_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Append a single row dict to *df* and return the updated DataFrame."""
    df.loc[len(df)] = row
    return df


def ensure_directory(path: str) -> None:
    """Create *path* (and any parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def save_to_csv(df: pd.DataFrame, directory: str, filename: str) -> str:
    """Save *df* as a CSV inside *directory* and return the full file path."""
    ensure_directory(directory)
    file_path = os.path.join(directory, filename)
    df.to_csv(file_path, index=False)
    return file_path


# ── Row definitions (V2 / V3) ─────────────────────────────────────────────────

NEW_ROW_V2 = {
    "product_id":   "P004",
    "product_name": "Keyboard",
    "category":     "Accessories",
    "transaction":  "purchase",
    "quantity":     7,
    "unit_price":   39.99,
    "total_amount": 279.93,
}

NEW_ROW_V3 = {
    "product_id":   "P005",
    "product_name": "Monitor",
    "category":     "Electronics",
    "transaction":  "sell",
    "quantity":     3,
    "unit_price":   299.99,
    "total_amount": 899.97,
}


# ── Orchestrator ──────────────────────────────────────────────────────────────

def main():
    df = create_base_data()
    df = append_row(df, NEW_ROW_V2)
    df = append_row(df, NEW_ROW_V3)

    file_path = save_to_csv(df, directory="data", filename="sample_data.csv")
    print(f"CSV file saved to {file_path}")


if __name__ == "__main__":
    main()
