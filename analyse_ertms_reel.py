"""
Analyse Complète - Vraies Déconnexions ERTMS
=============================================

Dataset : 1,594 déconnexions réelles (2024)
Source : Suivi ONCF

Auteur: PFE GSM-R
Date: 2026-02-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

print("="*70)
print("🚂 ANALYSE DÉCONNEXIONS ERTMS - DONNÉES RÉELLES ONCF")
print("="*70)

# ══════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════

print("\n📥 Chargement données...")

file_path = 'Suivi_des_déconnexions_ERTMS_et_actions_correctives__1_.xlsx'

# Lire feuille principale
df = pd.read_excel(file_path, sheet_name='Data')

print(f"✅ {len(df):,} déconnexions chargées")
print(f"✅ {len(df.columns)} colonnes")
print(f"✅ Période : {df['Date'].min()} → {df['Date'].max()}")

# ══════════════════════════════════════════════════════════════
# NETTOYAGE SIGNAL
# ══════════════════════════════════════════════════════════════

print("\n🧹 Nettoyage métriques signal...")

# Parser RxQual (dl Rxqual=3 → 3)
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

df['rxqual_value'] = df['RxQual'].apply(parse_rxqual)

# Parser RxLev (dl rx level= -63 dbm → -63)
def parse_rxlev(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).lower()
    if 'level' in val_str or 'dbm' in val_str:
        try:
            # Extraire nombre (peut être négatif)
            import re
            match = re.search(r'(-?\d+)', val_str)
            if match:
                return int(match.group(1))
        except:
            return np.nan
    return np.nan

df['rxlev_dbm'] = df['RxLev'].apply(parse_rxlev)

print(f"   ✅ RxQual parsé : {df['rxqual_value'].notna().sum():,} valeurs")
print(f"   ✅ RxLev parsé  : {df['rxlev_dbm'].notna().sum():,} valeurs")

# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

print("\n⚙️  Feature engineering...")

# Temps
df['hour'] = pd.to_datetime(df['Heure'], format='%H:%M:%S', errors='coerce').dt.hour
df['day_of_week'] = df['Date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Position
df['km_value'] = pd.to_numeric(df['Km'], errors='coerce')

# Signal quality categories
df['signal_quality'] = pd.cut(df['rxqual_value'], 
                              bins=[-1, 2, 4, 10], 
                              labels=['Good', 'Fair', 'Poor'])

df['signal_strength'] = pd.cut(df['rxlev_dbm'],
                               bins=[-200, -85, -70, 0],
                               labels=['Weak', 'Fair', 'Strong'])

# Communication issues
df['has_hdlc_retrans'] = df['Retransmission trames HDLC/T.70'].notna().astype(int)
df['com_dte_dce_ok'] = (df['Com DTE-DCE'] == 'OK').astype(int)
df['voisinage_ok'] = (df['Voisinage (10sec)'] == 'OK').astype(int)

print(f"   ✅ Features créées")

# ══════════════════════════════════════════════════════════════
# LABELS (TYPES DE DÉCONNEXION)
# ══════════════════════════════════════════════════════════════

print("\n🏷️  Création labels...")

# Label principal : Type de cause
df['label_cause'] = df['Sous Système mis en cause'].fillna('UNKNOWN')

# Label binaire : Problème GSMR ou pas
df['is_gsmr_issue'] = df['label_cause'].str.contains('GSMR', case=False, na=False).astype(int)

# Distribution
print(f"\n📊 Distribution par sous-système :")
print(df['label_cause'].value_counts())

print(f"\n📊 Issues GSMR vs Autres :")
print(f"   GSMR      : {df['is_gsmr_issue'].sum():,} ({df['is_gsmr_issue'].mean()*100:.1f}%)")
print(f"   Autres    : {(1-df['is_gsmr_issue']).sum():,} ({(1-df['is_gsmr_issue']).mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# ANALYSES STATISTIQUES
# ══════════════════════════════════════════════════════════════

print("\n" + "─"*70)
print("📊 ANALYSES STATISTIQUES")
print("─"*70)

# 1. Par zone
print("\n1️⃣  Déconnexions par intervalle (Top 10) :")
top_zones = df['Intervalle'].value_counts().head(10)
for zone, count in top_zones.items():
    pct = (count / len(df)) * 100
    print(f"   {zone:20} : {count:4} ({pct:5.1f}%)")

# 2. Par train
print("\n2️⃣  Déconnexions par rame (Top 10) :")
top_rames = df['Rame'].value_counts().head(10)
for rame, count in top_rames.items():
    pct = (count / len(df)) * 100
    print(f"   {rame:20} : {count:4} ({pct:5.1f}%)")

# 3. RxQual moyen par zone
print("\n3️⃣  RxQual moyen par intervalle (Top 10 pires) :")
rxqual_by_zone = df.groupby('Intervalle')['rxqual_value'].mean().sort_values(ascending=False).head(10)
for zone, rxq in rxqual_by_zone.items():
    print(f"   {zone:20} : {rxq:.2f}")

# 4. Causes principales
print("\n4️⃣  Top 5 causes racines :")
top_causes = df['Cause_racine'].value_counts().head(5)
for cause, count in top_causes.items():
    cause_str = str(cause)[:60]  # Tronquer
    pct = (count / len(df)) * 100
    print(f"   {cause_str:60} : {count:4} ({pct:5.1f}%)")

# ══════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════

print("\n📊 Génération graphiques...")

os.makedirs('data/analysis', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Distribution RxQual
axes[0, 0].hist(df['rxqual_value'].dropna(), bins=8, edgecolor='black', color='steelblue')
axes[0, 0].set_title('Distribution RxQual')
axes[0, 0].set_xlabel('RxQual (0=best, 7=worst)')
axes[0, 0].set_ylabel('Fréquence')
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribution RxLev
axes[0, 1].hist(df['rxlev_dbm'].dropna(), bins=30, edgecolor='black', color='coral')
axes[0, 1].set_title('Distribution RxLev')
axes[0, 1].set_xlabel('RxLev (dBm)')
axes[0, 1].set_ylabel('Fréquence')
axes[0, 1].axvline(x=-85, color='red', linestyle='--', label='Seuil critique')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Déconnexions par heure
hour_counts = df['hour'].value_counts().sort_index()
axes[0, 2].bar(hour_counts.index, hour_counts.values, color='green', alpha=0.7, edgecolor='black')
axes[0, 2].set_title('Déconnexions par Heure')
axes[0, 2].set_xlabel('Heure')
axes[0, 2].set_ylabel('Nombre')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# 4. Top 10 zones
top10_zones = df['Intervalle'].value_counts().head(10)
axes[1, 0].barh(range(len(top10_zones)), top10_zones.values, color='purple', alpha=0.7, edgecolor='black')
axes[1, 0].set_yticks(range(len(top10_zones)))
axes[1, 0].set_yticklabels(top10_zones.index)
axes[1, 0].set_title('Top 10 Intervalles (Déconnexions)')
axes[1, 0].set_xlabel('Nombre de déconnexions')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 5. Par sous-système
subsys_counts = df['label_cause'].value_counts().head(5)
axes[1, 1].pie(subsys_counts.values, labels=subsys_counts.index, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Répartition par Sous-Système')

# 6. Timeline
df_timeline = df.set_index('Date').resample('D').size()
axes[1, 2].plot(df_timeline.index, df_timeline.values, linewidth=2, color='darkblue')
axes[1, 2].set_title('Timeline - Déconnexions par Jour')
axes[1, 2].set_xlabel('Date')
axes[1, 2].set_ylabel('Nombre de déconnexions')
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('data/analysis/ertms_analysis.png', dpi=300, bbox_inches='tight')
print("   ✅ Graphiques : data/analysis/ertms_analysis.png")
plt.close()

# ══════════════════════════════════════════════════════════════
# DATASET ML
# ══════════════════════════════════════════════════════════════

print("\n💾 Préparation dataset ML...")

# Sélectionner features
ML_FEATURES = [
    'rxqual_value',
    'rxlev_dbm',
    'km_value',
    'hour',
    'day_of_week',
    'is_weekend',
    'has_hdlc_retrans',
    'com_dte_dce_ok',
    'voisinage_ok',
]

# Créer dataset
df_ml = df[ML_FEATURES + ['is_gsmr_issue']].copy()

# Renommer label
df_ml = df_ml.rename(columns={'is_gsmr_issue': 'label'})

# Supprimer NaN
df_ml = df_ml.dropna()

print(f"   Dataset ML : {df_ml.shape}")
print(f"   Features   : {len(ML_FEATURES)}")

# Sauvegarder
output_file = 'data/processed/ertms_ml_ready.csv'
os.makedirs('data/processed', exist_ok=True)
df_ml.to_csv(output_file, index=False)

print(f"   ✅ Sauvegardé : {output_file}")

# ══════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("✅ ANALYSE TERMINÉE !")
print("="*70)

print(f"""
📊 STATISTIQUES GLOBALES :
   Déconnexions totales : {len(df):,}
   Période              : {df['Date'].min()} → {df['Date'].max()}
   Trains uniques       : {df['N° Train'].nunique()}
   Rames uniques        : {df['Rame'].nunique()}
   Zones affectées      : {df['Intervalle'].nunique()}

🔥 ZONES LES PLUS PROBLÉMATIQUES :
   {chr(10).join('   ' + str(i) + '. ' + str(zone) + ' (' + str(count) + ' déconnexions)' 
                for i, (zone, count) in enumerate(top_zones.head(5).items(), 1))}

📡 QUALITÉ SIGNAL :
   RxQual moyen  : {df['rxqual_value'].mean():.2f} (0=best, 7=worst)
   RxLev moyen   : {df['rxlev_dbm'].mean():.1f} dBm
   < -85 dBm     : {(df['rxlev_dbm'] < -85).sum()} déconnexions ({(df['rxlev_dbm'] < -85).mean()*100:.1f}%)

📁 FICHIERS CRÉÉS :
   • data/analysis/ertms_analysis.png
   • data/processed/ertms_ml_ready.csv

🎯 DATASET ML PRÊT :
   Lignes    : {len(df_ml):,}
   Features  : {len(ML_FEATURES)}
   Label     : GSMR issue (0/1)

🚀 PROCHAINE ÉTAPE :
   Créer séquences temporelles + Entraîner 1D CNN
""")

print("="*70)