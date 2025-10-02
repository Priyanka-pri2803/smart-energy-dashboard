import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Path for dataset
CSV_PATH = r"C:\Users\Aishwarya R\smart-energy-dashboard\Smart Home Dataset.csv"

def load_and_preprocess():
    """
    Loads the Smart Home dataset, cleans it, encodes numeric + categorical features,
    and returns X_train, X_test, y_train, y_test, preprocessor, and feature names.
    """
    # Load dataset
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Show available columns (for debugging)
    print(" Columns in dataset:", df.columns.tolist())

    # Drop rows with missing values (optional)
    df = df.dropna()

    # Target column (energy consumption)
    target_col = "use [kW]"

    # Features to use (remove date/time columns)
    feature_cols = [col for col in df.columns if col not in [target_col, "date", "Date", "time"]]

    # Define X and y
    X = df[feature_cols]
    y = df[target_col]

    # Separate numeric and categorical features
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Apply preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Feature names after encoding
    feature_names = list(numeric_cols) + list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
    )

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names