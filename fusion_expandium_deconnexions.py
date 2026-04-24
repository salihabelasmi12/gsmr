"""
import pandas as pd
import numpy as np

print("="*70)
print("📊 FUSION EXPANDIUM + DÉCONNEXIONS")
print("="*70)

# ═══════════════════════════════════════
# ÉTAPE 1 : CHARGEMENT EXPANDIUM
# ═══════════════════════════════════════

print("\n📥 Chargement Expandium...")

df_2024 = pd.read_excel('data/raw/Expandium_2024.xlsx')
df_2025 = pd.read_excel('data/raw/Expandium_2025.xlsx')
df_2026 = pd.read_csv('data/raw/Expandium_2026.csv', sep=';', skiprows=1)

# Nettoyage colonnes
df_2024.columns = df_2024.columns.str.strip()
df_2025.columns = df_2025.columns.str.strip()
df_2026.columns = df_2026.columns.str.strip()

# Fusion
df_expandium_all = pd.concat([df_2024, df_2025, df_2026], ignore_index=True)

# Supprimer doublons
df_expandium_all = df_expandium_all.drop_duplicates()

print(f"✅ Total Expandium : {len(df_expandium_all):,}")

# ═══════════════════════════════════════
# ÉTAPE 2 : CHARGEMENT DÉCONNEXIONS
# ═══════════════════════════════════════

print("\n📥 Chargement Déconnexions Excel...")

try:
    df_deconnexions = pd.read_excel('data/raw/Deconnexions_2024_2026.xlsx')
    print(f"✅ {len(df_deconnexions):,} déconnexions")
except Exception as e:
    print(f"⚠️ Erreur : {e}")
    df_deconnexions = pd.DataFrame()

# ═══════════════════════════════════════
# ÉTAPE 3 : SAUVEGARDE FUSION
# ═══════════════════════════════════════

print("\n💾 Sauvegarde...")

df_expandium_all.to_csv('Expandium_2024_2026_FUSION.csv', index=False)

if not df_deconnexions.empty:
    df_deconnexions.to_excel('Deconnexions_2024_2026_COPIE.xlsx', index=False)

# ═══════════════════════════════════════
# ÉTAPE 4 : STATISTIQUES
# ═══════════════════════════════════════

print("\n📊 STATISTIQUES")

print(f"Total sessions : {len(df_expandium_all):,}")

if 'Start Time' in df_expandium_all.columns:
    df_expandium_all['Start Time'] = pd.to_datetime(df_expandium_all['Start Time'], errors='coerce')
    print(f"Période : {df_expandium_all['Start Time'].min()} → {df_expandium_all['Start Time'].max()}")

# ═══════════════════════════════════════
# ÉTAPE 5 : LABELLISATION
# ═══════════════════════════════════════

print("\n🏷️ Labellisation...")

df_expandium_all['is_disconnect'] = 0

if 'End Event' in df_expandium_all.columns:
    df_expandium_all.loc[df_expandium_all['End Event'] == 'Disconnect', 'is_disconnect'] = 1

if 'End Cause' in df_expandium_all.columns:
    df_expandium_all.loc[df_expandium_all['End Cause'] == 'Radio interface failure', 'is_disconnect'] = 1

if 'ETCS Connected' in df_expandium_all.columns:
    df_expandium_all.loc[df_expandium_all['ETCS Connected'] == 'Not Connected', 'is_disconnect'] = 1

n_deconnexions = df_expandium_all['is_disconnect'].sum()
n_normales = len(df_expandium_all) - n_deconnexions

print(f"Déconnexions : {n_deconnexions:,}")
print(f"Normales     : {n_normales:,}")

# Sauvegarde avec labels
df_expandium_all.to_csv('Expandium_LABELED.csv', index=False)

# ═══════════════════════════════════════
# ÉTAPE 6 : DATASET ÉQUILIBRÉ
# ═══════════════════════════════════════

print("\n⚖️ Dataset équilibré...")

df_deco = df_expandium_all[df_expandium_all['is_disconnect'] == 1]
df_norm = df_expandium_all[df_expandium_all['is_disconnect'] == 0]

n_min = min(len(df_deco), len(df_norm))

df_balanced = pd.DataFrame()

if n_min > 0:
    df_norm_sampled = df_norm.sample(n=n_min, random_state=42)

    df_balanced = pd.concat([df_deco, df_norm_sampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42)

    df_balanced.to_csv('Expandium_BALANCED.csv', index=False)

    print(f"✅ Dataset équilibré : {len(df_balanced):,}")

# ═══════════════════════════════════════
# FIN
# ═══════════════════════════════════════

print("\n" + "="*70)
print("✅ TERMINÉ")
print("="*70)