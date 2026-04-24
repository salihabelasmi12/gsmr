"""
Création Dataset ML Final
Expandium + Excel ERTMS
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re

print("="*70)
print("🔗 CRÉATION DATASET ML FINAL")
print("="*70)

# ══════════════════════════════════════════════════════════════
# CHARGER EXPANDIUM
# ══════════════════════════════════════════════════════════════

print("\n📥 Chargement Expandium...")

df_exp = pd.read_csv('data/raw/ETCS-Call-tracing_.csv', sep=';', skiprows=1)

print(f"✅ {len(df_exp):,} sessions Expandium")

# ══════════════════════════════════════════════════════════════
# PARSER DURÉES
# ══════════════════════════════════════════════════════════════

print("\n🔧 Parsing durées...")

def parse_duration_to_seconds(duration_str):
    """Convertit '15min 57s 948ms' en secondes"""
    if pd.isna(duration_str):
        return np.nan
    
    total_seconds = 0
    duration_str = str(duration_str)
    
    # Heures
    if 'h' in duration_str:
        match = re.search(r'(\d+)h', duration_str)
        if match:
            total_seconds += int(match.group(1)) * 3600
    
    # Minutes
    if 'min' in duration_str:
        match = re.search(r'(\d+)min', duration_str)
        if match:
            total_seconds += int(match.group(1)) * 60
    
    # Secondes
    if 's' in duration_str:
        match = re.search(r'(\d+)s', duration_str)
        if match:
            total_seconds += int(match.group(1))
    
    # Millisecondes
    if 'ms' in duration_str:
        match = re.search(r'(\d+)ms', duration_str)
        if match:
            total_seconds += int(match.group(1)) / 1000
    
    return total_seconds

df_exp['call_setup_duration_sec'] = df_exp['Call Setup Duration (ms)'].apply(parse_duration_to_seconds)
df_exp['transaction_duration_sec'] = df_exp['Transaction Duration (ms)'].apply(parse_duration_to_seconds)

print(f"   ✅ Call Setup Duration : {df_exp['call_setup_duration_sec'].notna().sum():,} valeurs")
print(f"   ✅ Transaction Duration: {df_exp['transaction_duration_sec'].notna().sum():,} valeurs")

# ══════════════════════════════════════════════════════════════
# CRÉER LABELS
# ══════════════════════════════════════════════════════════════

print("\n🏷️  Création labels...")

# Identifier déconnexions
df_exp['is_disconnect'] = (
    (df_exp['End Event'] == 'Disconnect') |
    (df_exp['End Cause'] == 'Radio interface failure') |
    (df_exp['ETCS Connected'] == 'Not Connected')
).astype(int)

# Distribution
n_disconnect = df_exp['is_disconnect'].sum()
n_normal = len(df_exp) - n_disconnect

print(f"\n📊 Distribution :")
print(f"   DÉCONNEXION (1) : {n_disconnect:,} ({n_disconnect/len(df_exp)*100:.1f}%)")
print(f"   NORMAL (0)      : {n_normal:,} ({n_normal/len(df_exp)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

print("\n⚙️  Feature engineering...")

# Timestamps
df_exp['start_time'] = pd.to_datetime(df_exp['Start Time'], errors='coerce')
df_exp['stop_time'] = pd.to_datetime(df_exp['Stop Time'], errors='coerce')

df_exp['hour'] = df_exp['start_time'].dt.hour
df_exp['day_of_week'] = df_exp['start_time'].dt.dayofweek
df_exp['is_weekend'] = df_exp['day_of_week'].isin([5, 6]).astype(int)
df_exp['is_night'] = df_exp['hour'].between(0, 6).astype(int)

# Durée anormale
df_exp['duration_very_short'] = (df_exp['transaction_duration_sec'] < 60).astype(int)
df_exp['duration_very_long'] = (df_exp['transaction_duration_sec'] > 3600).astype(int)

# Setup lent
df_exp['slow_setup'] = (df_exp['call_setup_duration_sec'] > 5).astype(int)

# End Event catégories
df_exp['end_event_is_disconnect'] = (df_exp['End Event'] == 'Disconnect').astype(int)
df_exp['end_event_is_clear'] = (df_exp['End Event'].str.contains('Clear', na=False)).astype(int)

# End Cause catégories
df_exp['cause_radio_failure'] = (df_exp['End Cause'] == 'Radio interface failure').astype(int)
df_exp['cause_network_error'] = (df_exp['End Cause'].str.contains('Network|out of order', case=False, na=False)).astype(int)

print("   ✅ Features créées")

# ══════════════════════════════════════════════════════════════
# SÉLECTIONNER FEATURES POUR ML
# ══════════════════════════════════════════════════════════════

print("\n📋 Sélection features ML...")

FEATURES = [
    'call_setup_duration_sec',
    'transaction_duration_sec',
    'hour',
    'day_of_week',
    'is_weekend',
    'is_night',
    'duration_very_short',
    'duration_very_long',
    'slow_setup',
    'end_event_is_disconnect',
    'end_event_is_clear',
    'cause_radio_failure',
    'cause_network_error'
]

# Créer dataset
df_ml = df_exp[FEATURES + ['is_disconnect']].copy()

# Supprimer NaN
df_ml = df_ml.dropna()

print(f"   Dataset ML : {df_ml.shape}")
print(f"   Features   : {len(FEATURES)}")

# Distribution finale
n_disconnect_final = df_ml['is_disconnect'].sum()
n_normal_final = len(df_ml) - n_disconnect_final

print(f"\n📊 Distribution finale :")
print(f"   DÉCONNEXION (1) : {n_disconnect_final:,} ({n_disconnect_final/len(df_ml)*100:.1f}%)")
print(f"   NORMAL (0)      : {n_normal_final:,} ({n_normal_final/len(df_ml)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# ÉQUILIBRER DATASET (optionnel)
# ══════════════════════════════════════════════════════════════

print("\n⚖️  Équilibrage dataset...")

# Sous-échantillonner la classe majoritaire
df_disconnect = df_ml[df_ml['is_disconnect'] == 1]
df_normal = df_ml[df_ml['is_disconnect'] == 0]

# Prendre autant de normaux que de déconnexions
n_samples = min(len(df_disconnect), len(df_normal))
n_samples = max(n_samples, 500)  # Minimum 500 par classe

df_normal_sampled = df_normal.sample(n=min(len(df_normal), n_samples), random_state=42)
df_disconnect_sampled = df_disconnect.sample(n=min(len(df_disconnect), n_samples), random_state=42)

df_balanced = pd.concat([df_normal_sampled, df_disconnect_sampled], ignore_index=True)

# Mélanger
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   Dataset équilibré : {len(df_balanced):,} lignes")
print(f"   DÉCONNEXION : {df_balanced['is_disconnect'].sum():,}")
print(f"   NORMAL      : {(1-df_balanced['is_disconnect']).sum():,}")

# ══════════════════════════════════════════════════════════════
# NORMALISER
# ══════════════════════════════════════════════════════════════

print("\n🔧 Normalisation...")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_balanced[FEATURES] = scaler.fit_transform(df_balanced[FEATURES])

print("   ✅ Features normalisées")

# Sauvegarder scaler
import pickle
import os
os.makedirs('models', exist_ok=True)
with open('models/scaler_expandium.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ══════════════════════════════════════════════════════════════
# SAUVEGARDER
# ══════════════════════════════════════════════════════════════

print("\n💾 Sauvegarde...")

os.makedirs('data/processed', exist_ok=True)

# Dataset complet
df_ml.to_csv('data/processed/expandium_ml_full.csv', index=False)
print(f"   ✅ data/processed/expandium_ml_full.csv ({len(df_ml):,} lignes)")

# Dataset équilibré
df_balanced.to_csv('data/processed/expandium_ml_balanced.csv', index=False)
print(f"   ✅ data/processed/expandium_ml_balanced.csv ({len(df_balanced):,} lignes)")

# ══════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("✅ DATASET ML CRÉÉ !")
print("="*70)

print(f"""
📊 DATASET FINAL :
   Dataset complet   : {len(df_ml):,} lignes
   Dataset équilibré : {len(df_balanced):,} lignes
   Features          : {len(FEATURES)}
   
🏷️  LABELS :
   DÉCONNEXION (1) : Sessions avec problèmes
   NORMAL (0)      : Sessions réussies
   
📁 FICHIERS CRÉÉS :
   • data/processed/expandium_ml_full.csv
   • data/processed/expandium_ml_balanced.csv
   • models/scaler_expandium.pkl
   
🚀 PROCHAINE ÉTAPE :
   Entraîner modèle Random Forest
   
   Commande : python train_rf_expandium.py
""")

print("="*70)