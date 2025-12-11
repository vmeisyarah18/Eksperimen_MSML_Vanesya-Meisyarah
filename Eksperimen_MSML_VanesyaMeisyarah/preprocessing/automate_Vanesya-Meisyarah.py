import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_dataset(path):
    """
    Load dataset diabetes.csv
    """
    return pd.read_csv(path)


def clean_dataset(df):
    """
    Membersihkan data sesuai preprocessing manual:
    - Replace nilai 0 pada kolom tertentu → NaN
    - Drop NA
    - Drop duplicates
    """

    cols_zero_invalid = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_zero_invalid] = df[cols_zero_invalid].replace(0, np.nan)

    # Drop missing values
    df = df.dropna()

    # Drop duplicates
    df = df.drop_duplicates()

    return df


def remove_outliers(df, target_column="Outcome"):
    """
    Menghapus outlier menggunakan metode IQR untuk seluruh kolom
    kecuali kolom target.
    """
    for col in df.columns:
        if col != target_column:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def scale_features(df, target_column="Outcome"):
    """
    Scaling fitur numerik menggunakan StandardScaler.
    """
    scaler = StandardScaler()
    numerical_cols = df.drop(columns=[target_column]).columns

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, scaler


def preprocess_pipeline(path="../diabetes.csv", save_output=True):
    """
    Pipeline preprocessing otomatis:
    1. Load dataset
    2. Replace zero → NaN → dropna → drop duplicates
    3. Remove outliers (IQR)
    4. Scaling fitur numerik
    5. Save hasil jika diminta
    """

    df = load_dataset(path)
    df = clean_dataset(df)
    df = remove_outliers(df)
    df, scaler = scale_features(df)

    if save_output:
        output_path = "diabetes_preprocessing.csv"
        df.to_csv(output_path, index=False)
        print("Dataset preprocessing selesai disimpan di:", output_path)

    # Split fitur & target untuk model training
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


# Auto-run ketika file dieksekusi langsung
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_pipeline("Eksperimen_MSML_VanesyaMeisyarah/diabetes.csv")
print("Preprocessing selesai — dataset siap digunakan!")

