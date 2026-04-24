import pandas as pd

print("="*70)
print("📊 ANALYSE FICHIER DECONNEXIONS ERTMS")
print("="*70)

file_path = "Suivi_des_déconnexions_ERTMS_et_actions_correctives__1_.xlsx"

print("\n📥 Chargement du fichier Excel...")
df = pd.read_excel(file_path)

print("✅ Fichier chargé")

print("\n📊 DIMENSIONS")
print("Lignes :", df.shape[0])
print("Colonnes :", df.shape[1])

print("\n📋 COLONNES :")
for i,col in enumerate(df.columns):
    print(f"{i+1}. {col}")

print("\n👀 APERÇU DES DONNÉES")
print(df.head())

print("\n📊 TYPES DE DONNÉES")
print(df.dtypes)

print("\n⚠️ VALEURS MANQUANTES")
print(df.isnull().sum())

print("\n📊 STATISTIQUES")
print(df.describe(include="all"))

print("\n"+"="*70)
print("✅ ANALYSE TERMINÉE")
print("="*70)