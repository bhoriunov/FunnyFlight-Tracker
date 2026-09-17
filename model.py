import pandas as pd
from sklearn.ensemble import IsolationForest

X = {
    "aircraft": ["A320", "A321", "A330", "B737", "B747", "E195", "A350"],
    "type_frequency": [1200, 700, 400, 1500, 5, 800, 40],
    "operator_frequency": [200, 150, 100, 250, 1, 200, 10]
}
Xf = pd.DataFrame(data=X)
Xf['type_rarity'] = 1 / ( Xf['type_frequency'] + 1 )
Xf['operator_rarity'] = 1 / ( Xf['operator_frequency'] + 1 )

model = IsolationForest(contamination=1/7, random_state=42)
model.fit(Xf[['type_rarity', 'operator_rarity']])

Xf['prediction'] = model.predict(Xf[['type_rarity', 'operator_rarity']])
Xf['anomaly_score'] = model.decision_function(Xf[['type_rarity', 'operator_rarity']])

print(Xf.sort_values(by='anomaly_score'))
