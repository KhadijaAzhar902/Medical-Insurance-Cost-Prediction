from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "insurance.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Load and clean the same dataset used in the notebook.
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().reset_index(drop=True)

# Match the notebook preprocessing exactly.
df_encoded = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int,
)

X = df_encoded.drop(columns="charges")
y = df_encoded["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

metrics = {
    "mae": float(mean_absolute_error(y_test, y_pred)),
    "mse": float(mean_squared_error(y_test, y_pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    "r2": float(r2_score(y_test, y_pred)),
}

joblib.dump(model, MODEL_DIR / "insurance_model.pkl")
(MODEL_DIR / "metadata.json").write_text(
    json.dumps(
        {
            "feature_order": X.columns.tolist(),
            "test_size": 0.20,
            "random_state": 42,
            "metrics": metrics,
        },
        indent=2,
    ),
    encoding="utf-8",
)

print("Model saved to model/insurance_model.pkl")
print(f"MAE:  {metrics['mae']:.2f}")
print(f"RMSE: {metrics['rmse']:.2f}")
print(f"R²:   {metrics['r2']:.4f}")
