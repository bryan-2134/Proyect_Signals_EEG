import pandas as pd
import numpy as np
df = pd.read_csv(r"B:\Datos\DatasetEEG.csv")

canales_betaH = [c for c in df.columns if 'BetaH' in c]
canales_betaL = [c for c in df.columns if 'BetaL' in c]
canales_alpha = [c for c in df.columns if 'Alpha' in c]
canales_theta = [c for c in df.columns if 'Theta' in c]

df['mean_betaH'] = df[canales_betaH].mean(axis=1)
df['mean_betaL'] = df[canales_betaL].mean(axis=1)
df['mean_alpha'] = df[canales_alpha].mean(axis=1)
df['mean_theta'] = df[canales_theta].mean(axis=1)

df['relacion_beta_alpha'] = (df['mean_betaH'] + df['mean_betaL']) / (df['mean_alpha'] + 1e-6)
df['relacion_theta_beta'] = df['mean_theta'] / ((df['mean_betaH'] + df['mean_betaL']) + 1e-6)

estres_idx = df['estres'] == 1

p33 = df.loc[estres_idx, 'relacion_beta_alpha'].quantile(0.33)
p66 = df.loc[estres_idx, 'relacion_beta_alpha'].quantile(0.66)


print("\nLimites")
print(f"Sin estres: estres = 0")
print(f"Estres bajo: relacion_beta_alpha <= {p33:.4f}")
print(f"Estres medio: {p33:.4f} < relacion_beta_alpha <= {p66:.4f}")
print(f"Estres alto: relacion_beta_alpha > {p66:.4f}")

def asignar_nivel(row):
    if row['estres'] == 0:
        return 0  # sin estres
    elif row['relacion_beta_alpha'] <= p33:
        return 1  # estres bajo
    elif row['relacion_beta_alpha'] <= p66:
        return 2  # estres medio
    else:
        return 3  # estrés alto

df['nivel_estres'] = df.apply(asignar_nivel, axis=1)

print("\n-- Distribucion")
print(df['nivel_estres'].value_counts().sort_index())
print("\n0 = sin estres | 1 = bajo | 2 = medio | 3 = alto")