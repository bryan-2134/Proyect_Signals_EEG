import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

df = pd.read_csv(r'B:\Datos\DatasetEEG_niveles.csv')
features = [c for c in df.columns if 'POW.' in c]
X = df[features]
y = df['nivel_estres']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Distribucion del conjunto de entrenamiento:")
print(y_train.value_counts().sort_index())

model = XGBClassifier(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)
print("\nEntrenando XGBoost...")
model.fit(X_train_scaled, y_train)
print("Entrenamiento completo")

y_pred = model.predict(X_test_scaled)

print("\nResultados modelo:")
print(classification_report(y_test, y_pred,
      target_names=['Sin estres', 'Bajo', 'Medio', 'Alto']))

joblib.dump(model,  r'B:\Datos\modelo_XGBoost.pkl')
joblib.dump(scaler, r'B:\Datos\scaler_XGBoost.pkl')
print("Guardado en B:\\Datos")