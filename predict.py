from pathlib import Path

import joblib
import pandas as pd

artifacts = joblib.load(Path(__file__).resolve().parent / "model_artifacts.joblib")

model = artifacts["model"]
aircraft_counts = artifacts["aircraft_counts"]
operator_counts = artifacts["operator_counts"]
pair_counts = artifacts["pair_counts"]
train_min = artifacts["train_min"]
train_max = artifacts["train_max"]

def score_plane(aircraft: str, operator_code: str) -> dict:
    new_plane = pd.DataFrame([
        {
            "aircraft": aircraft,
            "operator_code": operator_code
        }
    ])

    new_plane["type_frequency"] = aircraft_counts.get(aircraft, 0)
    new_plane["type_rarity"] = 1 / (new_plane["type_frequency"] + 1)
    new_plane["operator_frequency"] = operator_counts.get(operator_code, 0)
    new_plane["operator_rarity"] = 1 / (new_plane["operator_frequency"] + 1)
    new_plane["pair_frequency"] = pair_counts.get((aircraft, operator_code), 0)
    new_plane["pair_rarity"] = 1 / (new_plane["pair_frequency"] + 1)
    
    features_new_plane = new_plane[[
        "type_rarity",
        "operator_rarity",
        "pair_rarity"
    ]]

    new_score = model.decision_function(features_new_plane)[0]
    new_rarity = 100 * (train_max - new_score) / (train_max - train_min)

    return {
        "aircraft": aircraft,
        "operator_code": operator_code,
        "type_frequency": int(new_plane["type_frequency"].iloc[0]),
        "operator_frequency": int(new_plane["operator_frequency"].iloc[0]),
        "pair_frequency": int(new_plane["pair_frequency"].iloc[0]),
        "anomaly_score": float(new_score),
        "rarity_score": float(new_rarity),
    }
