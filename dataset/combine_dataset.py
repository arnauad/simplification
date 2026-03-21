import pandas as pd

def combine_csvs(file1_path, file2_path, output_path):
    """
    Combines two CSV files with different schemas into a unified CSV
    with columns: id, original, simplified.
    """

    # Load both CSVs
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Normalize first dataset
    df1_clean = df1.rename(columns={
        "original_sent": "original",
        "simplified_sent": "simplified"
    })[["original", "simplified"]]

    # Normalize second dataset
    df2_clean = df2.rename(columns={
        "Segmento": "original",
        "Propuesta": "simplified"
    })[["original", "simplified"]]

    # Combine both datasets
    combined_df = pd.concat([df1_clean, df2_clean], ignore_index=True)

    # Drop rows with missing values (optional but recommended)
    combined_df = combined_df.dropna(subset=["original", "simplified"])

    # Create new unique ID
    combined_df.insert(0, "id", range(1, len(combined_df) + 1))

    # Save to CSV
    combined_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Combined CSV saved to: {output_path}")

if __name__ == '__main__':
    combine_csvs("../data/simplext.csv", "../data/feina.csv", "../data/training_data_es.csv")