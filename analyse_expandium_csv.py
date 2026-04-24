"""
Analyse du fichier ETCS Call Tracing CSV d'Expandium
"""

import pandas as pd
import numpy as np

"""
Analyse du fichier ETCS Call Tracing CSV d'Expandium
"""

import pandas as pd
import numpy as np

print("="*70)
print("📊 ANALYSE FICHIER EXPANDIUM")
print("="*70)

# Charger en sautant la première ligne (sep=;)
print("\n📥 Chargement...")

try:
    # Lire avec séparateur ; et skip première ligne
    df = pd.read_csv('data/raw/ETCS-Call-tracing_.csv', sep=';', skiprows=1)
    print(f"✅ {len(df):,} lignes chargées")
    print(f"✅ {len(df.columns)} colonnes")
except Exception as e:
    print(f"❌ ERREUR : {e}")
    exit(1)

# Afficher colonnes
print("\n📋 COLONNES DISPONIBLES :\n")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2}. {col}")

# Afficher premières lignes (juste les colonnes importantes)
print("\n📊 APERÇU DES 5 PREMIÈRES LIGNES :\n")
important_cols = ['Start Time', 'Stop Time', 'Call Setup Duration (ms)', 
                  'Transaction Duration (ms)', 'GSM-R Connected', 
                  'ETCS Connected', 'End Event', 'End Cause']

# Vérifier quelles colonnes existent
existing_cols = [col for col in important_cols if col in df.columns]

if existing_cols:
    print(df[existing_cols].head().to_string())
else:
    print(df.head().to_string())

# Statistiques
print("\n📊 STATISTIQUES DE BASE :")
print(f"   Total lignes       : {len(df):,}")
print(f"   Total colonnes     : {len(df.columns)}")

# Valeurs uniques des colonnes importantes
if 'GSM-R Connected' in df.columns:
    print(f"\n📊 GSM-R Connected :")
    print(df['GSM-R Connected'].value_counts())

if 'ETCS Connected' in df.columns:
    print(f"\n📊 ETCS Connected :")
    print(df['ETCS Connected'].value_counts())

if 'End Event' in df.columns:
    print(f"\n📊 End Event (top 10) :")
    print(df['End Event'].value_counts().head(10))

if 'End Cause' in df.columns:
    print(f"\n📊 End Cause (top 10) :")
    print(df['End Cause'].value_counts().head(10))

# Sauvegarder