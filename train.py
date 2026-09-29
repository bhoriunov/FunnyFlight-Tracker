from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
#TRAIN
Xf = pd.read_csv(BASE_DIR / "training_data.csv")
Xf["aircraft"] = Xf["model_aircraft"]

aircraft_counts = Xf["aircraft"].value_counts()
operator_counts = Xf["operator_code"].value_counts()

Xf["type_frequency"] = Xf["aircraft"].map(aircraft_counts)
Xf["operator_frequency"] = Xf["operator_code"].map(operator_counts)
Xf["type_rarity"] = 1 / (Xf["type_frequency"] + 1)
Xf["operator_rarity"] = 1 / (Xf["operator_frequency"] + 1)
Xf["pair_frequency"] = Xf.groupby(["aircraft", "operator_code"]).transform("size")
Xf["pair_rarity"] = 1 / (Xf["pair_frequency"] + 1)

features = Xf[[
    'type_rarity',
    'operator_rarity',
    'pair_rarity'
]]

model = IsolationForest(contamination='auto', random_state=42)
model.fit(features)

Xf["prediction"] = model.predict(features)
Xf["anomaly_score"] = model.decision_function(features)
Xf["rarity_score"] = 100 * (
        Xf["anomaly_score"].max() - Xf["anomaly_score"]
    ) / (
        Xf["anomaly_score"].max() - Xf["anomaly_score"].min()
    )

result = Xf.sort_values(by="rarity_score", ascending=False)
result.to_csv(BASE_DIR / "result.csv", index=False)

pair_counts = Xf.groupby(["aircraft", "operator_code"]).size().to_dict()

artifacts = {
    "model": model,
    "aircraft_counts": aircraft_counts,
    "operator_counts": operator_counts,
    "pair_counts": pair_counts,
    "train_min": Xf["anomaly_score"].min(),
    "train_max": Xf["anomaly_score"].max(),
}

joblib.dump(artifacts, BASE_DIR / "model_artifacts.joblib")