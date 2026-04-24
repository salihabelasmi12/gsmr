import pandas as pd

fichier = "data/raw/etcs_data.csv"

df = pd.read_csv(fichier, sep=";", skiprows=1)

print("Colonnes détectées :")
print(df.columns)

print("\nAperçu :")
print(df.head())
