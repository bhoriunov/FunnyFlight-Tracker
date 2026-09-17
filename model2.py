import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

#TRAIN
X = {
    "aircraft": [
        # A320 - 55
        *["A320"] * 55,
        # A321 - 30
        *["A321"] * 30,
        # B737 - 50
        *["B737"] * 50,
        # E190 - 24
        *["E190"] * 24,
        # A330 - 16
        *["A330"] * 16,
        # B787 - 14
        *["B787"] * 14,
        # A350 - 10
        *["A350"] * 10,
        # B777 - 8
        *["B777"] * 8,
        # B747 - 3
        *["B747"] * 3,
        # A380 - 2
        *["A380"] * 2,
        # AN124 - 1
        "AN124",
        # IL76 - 1
        "IL76",
        # MD11 - 1
        "MD11",
    ],

    "operator": [
        # A320
        *["Eurowings"] * 14,
        *["Lufthansa"] * 11,
        *["KLM"] * 9,
        *["Air France"] * 8,
        *["Swiss"] * 5,
        *["Austrian"] * 4,
        *["LOT"] * 4,

        # A321
        *["Wizz Air"] * 10,
        *["Lufthansa"] * 6,
        *["Swiss"] * 5,
        *["Turkish"] * 4,
        *["LOT"] * 3,
        *["Austrian"] * 2,

        # B737
        *["Ryanair"] * 18,
        *["Smartwings"] * 12,
        *["LOT"] * 7,
        *["Turkish"] * 5,
        *["KLM"] * 4,
        *["Lufthansa"] * 4,

        # E190
        *["LOT"] * 8,
        *["Austrian"] * 6,
        *["KLM"] * 5,
        *["Air France"] * 3,
        *["Lufthansa"] * 2,

        # A330
        *["Turkish"] * 5,
        *["Air France"] * 4,
        *["Lufthansa"] * 3,
        *["Swiss"] * 2,
        *["KLM"] * 2,

        # B787
        *["British Airways"] * 4,
        *["LOT"] * 4,
        *["Turkish"] * 3,
        *["Qatar"] * 2,
        *["Air Canada"] * 1,

        # A350
        *["Qatar"] * 3,
        *["Lufthansa"] * 2,
        *["British Airways"] * 2,
        *["Air France"] * 1,
        *["Swiss"] * 1,
        *["Finnair"] * 1,

        # B777
        *["Emirates"] * 3,
        *["Qatar"] * 2,
        *["Turkish"] * 2,
        *["British Airways"] * 1,

        # B747
        *["Lufthansa"] * 2,
        *["Cargolux"] * 1,

        # A380
        "Emirates",
        "British Airways",

        # AN124
        "Antonov Airlines",

        # IL76
        "Silk Way",

        # MD11
        "Western Global",
    ],
}
Xf = pd.DataFrame(data=X)

aircraft_counts = Xf["aircraft"].value_counts()
operator_counts = Xf["operator"].value_counts()
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_data = encoder.fit_transform(Xf[["aircraft", "operator"]])
encoded_Xf = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(["aircraft", "operator"]))
# print(encoded_Xf)

Xf["type_frequency"] = Xf["aircraft"].map(aircraft_counts)
Xf["operator_frequency"] = Xf["operator"].map(operator_counts)
Xf["type_rarity"] = 1 / (Xf["type_frequency"] + 1)
Xf["operator_rarity"] = 1 / (Xf["operator_frequency"] + 1)
Xf["pair_frequency"] = Xf.groupby(["aircraft", "operator"]).transform("size")
Xf["pair_rarity"] = 1 / (Xf["pair_frequency"] + 1)

features = Xf[['type_rarity', 'operator_rarity', 'pair_rarity']]
features = pd.concat([features, encoded_Xf], axis=1)

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
result.to_csv("result.csv", index=False)

pair_counts = Xf.groupby(["aircraft", "operator"]).size().to_dict()

artifacts = {
    "model": model,
    "encoder": encoder,
    "aircraft_counts": aircraft_counts,
    "operator_counts": operator_counts,
    "pair_counts": pair_counts,
    "train_min": Xf["anomaly_score"].min(),
    "train_max": Xf["anomaly_score"].max(),
}

joblib.dump(artifacts, "model_artifacts.joblib")

#INFERENCE
def score_plane(aircraft: str, operator: str) -> dict:
    new_plane = pd.DataFrame([
        {
            "aircraft": aircraft,
            "operator": operator
        }
    ])

    new_plane["type_frequency"] = aircraft_counts.get(aircraft, 0)
    new_plane["type_rarity"] = 1 / (new_plane["type_frequency"] + 1)
    new_plane["operator_frequency"] = operator_counts.get(operator, 0)
    new_plane["operator_rarity"] = 1 / (new_plane["operator_frequency"] + 1)
    new_plane["pair_frequency"] = pair_counts.get((aircraft, operator), 0)
    new_plane["pair_rarity"] = 1 / (new_plane["pair_frequency"] + 1)

    encoded_data_new_plane = encoder.transform(new_plane[["aircraft", "operator"]])
    encoded_new_plane = pd.DataFrame(
        encoded_data_new_plane, columns=encoder.get_feature_names_out(["aircraft", "operator"]))

    features_new_plane = new_plane[['type_rarity', 'operator_rarity', 'pair_rarity']]
    features_new_plane = pd.concat([features_new_plane, encoded_new_plane], axis=1)

    train_min = artifacts["train_min"]
    train_max = artifacts["train_max"]

    new_score = model.decision_function(features_new_plane)[0]
    new_rarity = 100 * (train_max - new_score) / (train_max - train_min)

    return {
        "aircraft": aircraft,
        "operator": operator,
        "type_frequency": int(new_plane["type_frequency"].iloc[0]),
        "operator_frequency": int(new_plane["operator_frequency"].iloc[0]),
        "pair_frequency": int(new_plane["pair_frequency"].iloc[0]),
        "anomaly_score": float(new_score),
        "rarity_score": float(new_rarity),
    }

print(score_plane("A320", "Lufthansa"))
print(score_plane("B747", "Cargolux"))
print(score_plane("AN225", "Unknown Cargo"))