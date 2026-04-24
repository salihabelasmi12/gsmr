"""
Script Simplifié - Préparation Dataset ML
Sans séries temporelles Expandium
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

print("="*70)
print("📊 PRÉPARATION DATASET ML - VERSION SIMPLIFIÉE")
print("="*70)

# ══════════════════════════════════════════════════════════════
# ÉTAPE 1 : CHARGER EXCEL DÉCONNEXIONS
# ══════════════════════════════════════════════════════════════

print("\n📥 Chargement Excel déconnexions...")

excel_file = 'Suivi_des_déconnexions_ERTMS_et_actions_correctives__1_.xlsx'
df = pd.read_excel(excel_file, sheet_name='Data')

print(f"✅ {len(df):,} déconnexions chargées")

# ══════════════════════════════════════════════════════════════
# ÉTAPE 2 : PARSER RXQUAL ET RXLEV
# ══════════════════════════════════════════════════════════════

print("\n🔧 Parsing RxQual et RxLev...")

def parse_rxqual(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    if 'rxqual' in val_str:
        try:
            return int(val_str.split('=')[1].split()[0])
        except:
            return np.nan
    return np.nan

def parse_rxlev(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    if 'level' in val_str or 'dbm' in val_str:
        try:
            import re
            match = re.search(r'(-?\d+)', val_str)
            if match:
                return int(match.group(1))
        except:
            return np.nan
    return np.nan

df['rxqual_value'] = df['RxQual'].apply(parse_rxqual)
df['rxlev_dbm'] = df['RxLev'].apply(parse_rxlev)

print(f"   RxQual parsé : {df['rxqual_value'].notna().sum():,} valeurs")
print(f"   RxLev parsé  : {df['rxlev_dbm'].notna().sum():,} valeurs")

# ══════════════════════════════════════════════════════════════
# ÉTAPE 3 : FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

print("\n⚙️  Feature engineering...")

# Timestamp
df['timestamp'] = pd.to_datetime(
    df['Date'].astype(str) + ' ' + df['Heure'].astype(str),
    errors='coerce'
)

df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Position
df['km_value'] = pd.to_numeric(df['Km'], errors='coerce')

# Signal quality
df['signal_quality'] = pd.cut(df['rxqual_value'], 
                              bins=[-1, 2, 4, 10], 
                              labels=['Good', 'Fair', 'Poor'])

# Communication
df['has_hdlc_retrans'] = df['Retransmission trames HDLC/T.70'].notna().astype(int)
df['com_dte_dce_ok'] = (df['Com DTE-DCE'] == 'OK').astype(int)
df['voisinage_ok'] = (df['Voisinage (10sec)'] == 'OK').astype(int)

print("   ✅ Features créées")

# ══════════════════════════════════════════════════════════════
# ÉTAPE 4 : CRÉER LABELS
# ══════════════════════════════════════════════════════════════

print("\n🏷️  Création labels...")

# Tous les événements dans Excel sont des DÉCONNEXIONS
df['label'] = 1  # DÉCONNEXION

# Identifier type de problème (GSMR vs BORD)
df['is_gsmr_issue'] = df['Sous Système mis en cause'].str.contains(
    'GSMR', case=False, na=False
).astype(int)

print(f"   Total déconnexions : {len(df):,}")
print(f"   Issues GSMR : {df['is_gsmr_issue'].sum():,}")
print(f"   Issues BORD : {(1-df['is_gsmr_issue']).sum():,}")

# ══════════════════════════════════════════════════════════════
# ÉTAPE 5 : SÉLECTIONNER FEATURES POUR ML
# ══════════════════════════════════════════════════════════════

print("\n📋 Sélection features ML...")

FEATURES = [
    'rxqual_value',
    'rxlev_dbm',
    'km_value',
    'hour',
    'day_of_week',
    'is_weekend',
    'has_hdlc_retrans',
    'com_dte_dce_ok',
    'voisinage_ok'
]

# Créer dataset
df_ml = df[FEATURES + ['label', 'is_gsmr_issue']].copy()

# Supprimer NaN
df_ml = df_ml.dropna()

print(f"   Dataset ML : {df_ml.shape}")
print(f"   Features : {len(FEATURES)}")

# ══════════════════════════════════════════════════════════════
# ÉTAPE 6 : NORMALISATION
# ══════════════════════════════════════════════════════════════

print("\n🔧 Normalisation...")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_ml[FEATURES] = scaler.fit_transform(df_ml[FEATURES])

print("   ✅ Features normalisées")

# Sauvegarder scaler
import pickle
os.makedirs('models', exist_ok=True)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ══════════════════════════════════════════════════════════════
# ÉTAPE 7 : SAUVEGARDER
# ══════════════════════════════════════════════════════════════

print("\n💾 Sauvegarde...")

os.makedirs('data/processed', exist_ok=True)
output_file = 'data/processed/dataset_ml_ready.csv'
df_ml.to_csv(output_file, index=False)

print(f"   ✅ Sauvegardé : {output_file}")

# ══════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("✅ PRÉPARATION TERMINÉE !")
print("="*70)

print(f"""
📊 DATASET FINAL :
   Lignes   : {len(df_ml):,}
   Features : {len(FEATURES)}
   Labels   : DÉCONNEXION (tous = 1)

📁 FICHIER CRÉÉ :
   {output_file}

🚀 PROCHAINE ÉTAPE :
   Entraîner le modèle 1D CNN
   
   Commande : python train_1dcnn_model.py
""")

print("="*70)