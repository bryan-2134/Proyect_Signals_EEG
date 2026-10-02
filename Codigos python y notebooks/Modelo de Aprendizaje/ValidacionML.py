import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, accuracy_score, precision_score, f1_score, confusion_matrix)
from xgboost import XGBClassifier

df = pd.read_csv(r'B:\Datos\DatasetEEG_niveles.csv')
features = [c for c in df.columns if 'POW.' in c]
X = df[features]
y = df['nivel_estres']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42, stratify=y )

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Distribucion de datos:")
print(y_train.value_counts().sort_index())

model = XGBClassifier(
    n_estimators=600, max_depth=8, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, eval_metric='mlogloss', random_state=42, n_jobs=-1 )

print("\nEntrenando XGBoost...")
model.fit(X_train_scaled, y_train)
print("Entrenamiento completo")

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nMetricas de validacion")
print("Exactitud (Accuracy):", round(accuracy, 5))
print("Precision           :", round(precision, 5))
print("F1-Score            :", round(f1, 5))

print("\nReporte ")
print(classification_report( y_test, y_pred, 
target_names=['Sin estres', 'Estres Bajo', 'Estres Medio', 'Estres Alto']))

print("\nMatriz de confusion")
print(confusion_matrix(y_test, y_pred))

scores = cross_val_score(model, X_train_scaled, y_train, cv=8)
print("\nValidacion cruzada")
print("Accuracy promedio:", round(scores.mean(), 4))
print("Desviacion estandar:", round(scores.std(), 4))