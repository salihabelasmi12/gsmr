import pandas as pd

print("Verification colonnes Expandium...")

df = pd.read_csv('Expandium_2026.csv', sep=';', skiprows=1, nrows=5)

print(f"\nNombre de colonnes : {len(df.columns)}")
print("\nListe des colonnes :")
print("="*60)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n" + "="*60)
print("\nPremiere ligne :")
print(df.head(1).T)