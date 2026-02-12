import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor():
    numeric_features = [
        "age",
        "restingBP",
        "serumcholestrol",
        "maxheartrate",
        "oldpeak",
    ]
    categorical_features = [
        "gender",
        "chestpain",
        "fastingbloodsugar",
        "restingrelectro",
        "exerciseangia",
        "slope",
        "noofmajorvessels",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def load_data(test_size=0.2, random_state=42):
    df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")

    # Drop patient id (identifier only).
    df = df.drop("patientid", axis=1)

    # Treat zero cholesterol as missing (clinically invalid in this dataset).
    df.loc[df["serumcholestrol"] == 0, "serumcholestrol"] = pd.NA

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    preprocessor = build_preprocessor()

    return X_train, X_test, y_train, y_test, preprocessor