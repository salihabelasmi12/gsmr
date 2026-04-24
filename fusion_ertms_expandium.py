"""
Fusion ERTMS Excel + Expandium
Linkage intelligent par timestamp et IMEI
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

print("="*70)
print("🔗 FUSION ERTMS + EXPANDIUM")
print("="*70)

# ══════════════════════════════════════════════════════════════
# CHARGER ERTMS EXCEL
# ══════════════════════════════════════════════════════════════

print("\n📥 Chargement ERTMS Excel...")

df_ertms = pd.read_excel('Suivi_des_déconnexions_ERTMS_et_actions_correctives__1_.xlsx', 
                         sheet_name='Data')

print(f"✅ {len(df_ertms):,} déconnexions ERTMS")

# Créer timestamp
df_ertms['timestamp'] = pd.to_datetime(
    df_ertms['Date'].astype(str) + ' ' + df_ertms['Heure'].astype(str),
    errors='coerce'
)

# Nettoyer IMEI
df_ertms['imei_clean'] = df_ertms['IMEI'].astype(str).str.strip()

print(f"   Période : {df_ertms['timestamp'].min()} → {df_ertms['timestamp'].max()}")
print(f"   IMEI uniques : {df_ertms['imei_clean'].nunique()}")

# ══════════════════════════════════════════════════════════════
# CHARGER EXPANDIUM
# ══════════════════════════════════════════════════════════════

print("\n📥 Chargement Expandium...")

df_exp = pd.read_csv('data/raw/ETCS-Call-tracing_.csv', sep=';', skiprows=1)

print(f"✅ {len(df_exp):,} sessions Expandium")

# Parser timestamps
df_exp['start_time'] = pd.to_datetime(df_exp['Start Time'], errors='coerce')
df_exp['stop_time'] = pd.to_datetime(df_exp['Stop Time'], errors='coerce')

# Nettoyer IMEI
df_exp['imei_clean'] = df_exp['IMEI'].astype(str).str.strip()

print(f"   Période : {df_exp['start_time'].min()} → {df_exp['stop_time'].max()}")
print(f"   IMEI uniques : {df_exp['imei_clean'].nunique()}")

# ══════════════════════════════════════════════════════════════
# FONCTION DE LINKAGE
# ══════════════════════════════════════════════════════════════

def find_matching_session(ertms_row, df_expandium, tolerance_minutes=30):
    """
    Trouve la session Expandium correspondant à une déconnexion ERTMS
    
    Critères :
    1. IMEI identique (priorité)
    2. Timestamp ERTMS dans [start_time - tolerance, stop_time + tolerance]
    """
    
    ertms_time = ertms_row['timestamp']
    ertms_imei = ertms_row['imei_clean']
    
    if pd.isna(ertms_time):
        return None
    
    # Filtre 1 : IMEI
    if ertms_imei and ertms_imei != 'nan':
        df_filtered = df_expandium[df_expandium['imei_clean'] == ertms_imei]
    else:
        df_filtered = df_expandium
    
    if len(df_filtered) == 0:
        # Pas de match IMEI, chercher sans IMEI
        df_filtered = df_expandium
    
    # Filtre 2 : Timestamp dans fenêtre
    tolerance = timedelta(minutes=tolerance_minutes)
    
    mask = (
        (df_filtered['start_time'] - tolerance <= ertms_time) &
        (df_filtered['stop_time'] + tolerance >= ertms_time)
    )
    
    matches = df_filtered[mask]
    
    if len(matches) == 0:
        return None
    
    # Si plusieurs matches, prendre le plus proche temporellement
    matches['time_diff'] = abs((matches['start_time'] - ertms_time).dt.total_seconds())
    best_match = matches.loc[matches['time_diff'].idxmin()]
    
    return best_match

# ══════════════════════════════════════════════════════════════
# LINKAGE
# ══════════════════════════════════════════════════════════════

print("\n🔗 Linkage ERTMS ↔ Expandium...")
print("   (Cela peut prendre 2-3 minutes...)")

matched_sessions = []
unmatched_count = 0

for idx, row in df_ertms.iterrows():
    if (idx + 1) % 100 == 0:
        print(f"   Progression : {idx+1}/{len(df_ertms)}")
    
    match = find_matching_session(row, df_exp, tolerance_minutes=5)
    
    if match is not None:
        # Créer entrée enrichie
        enriched = {
            # Données ERTMS
            'source': 'ERTMS',
            'timestamp': row['timestamp'],
            'km': row.get('Km', np.nan),
            'zone_gsmr': row.get('Intervalle', ''),
            'cause_racine': row.get('Cause_racine', ''),
            'sous_systeme': row.get('Sous Système mis en cause', ''),
            'rxqual_ertms': row.get('RxQual', ''),
            'rxlev_ertms': row.get('RxLev', ''),
            
            # Données Expandium
            'start_time': match['start_time'],
            'stop_time': match['stop_time'],
            'call_setup_duration': match['Call Setup Duration (ms)'],
            'transaction_duration': match['Transaction Duration (ms)'],
            'gsmr_connected': match['GSM-R Connected'],
            'etcs_connected': match['ETCS Connected'],
            'end_event': match['End Event'],
            'end_cause': match['End Cause'],
            'imei': match['IMEI'],
            
            # Label
            'label': 1  # Déconnexion
        }
        
        matched_sessions.append(enriched)
    else:
        unmatched_count += 1

print(f"\n✅ Linkage terminé :")
print(f"   Matched   : {len(matched_sessions):,} déconnexions")
print(f"   Unmatched : {unmatched_count:,} déconnexions")
print(f"   Taux match: {len(matched_sessions)/len(df_ertms)*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# AJOUTER SESSIONS NORMALES EXPANDIUM
# ══════════════════════════════════════════════════════════════

print("\n➕ Ajout sessions NORMALES Expandium...")

# IDs des sessions déjà matchées
matched_indices = set()
for session in matched_sessions:
    # Retrouver l'index dans df_exp
    mask = (
        (df_exp['start_time'] == session['start_time']) &
        (df_exp['IMEI'] == session['imei'])
    )
    matched_idx = df_exp[mask].index
    if len(matched_idx) > 0:
        matched_indices.add(matched_idx[0])

# Sessions non matchées = sessions normales
df_normal = df_exp[~df_exp.index.isin(matched_indices)]

print(f"   Sessions Expandium non matchées : {len(df_normal):,}")

# Identifier vraiment normales (pas de problème)
df_normal_clean = df_normal[
    (df_normal['End Event'] == 'User Plane SUBSET026 Term Session') &
    (df_normal['ETCS Connected'] == 'Connected') &
    (df_normal['End Cause'].isna())
]

print(f"   Sessions vraiment normales : {len(df_normal_clean):,}")

# Échantillonner pour équilibrer
n_normal_needed = len(matched_sessions)
n_normal_available = min(len(df_normal_clean), n_normal_needed)

df_normal_sampled = df_normal_clean.sample(n=n_normal_available, random_state=42)

print(f"   Sessions normales sélectionnées : {len(df_normal_sampled):,}")

# Convertir en format unifié
normal_sessions = []

for idx, row in df_normal_sampled.iterrows():
    normal = {
        'source': 'Expandium_Normal',
        'timestamp': row['start_time'],
        'km': np.nan,
        'zone_gsmr': '',
        'cause_racine': '',
        'sous_systeme': '',
        'rxqual_ertms': '',
        'rxlev_ertms': '',
        
        'start_time': row['start_time'],
        'stop_time': row['stop_time'],
        'call_setup_duration': row['Call Setup Duration (ms)'],
        'transaction_duration': row['Transaction Duration (ms)'],
        'gsmr_connected': row['GSM-R Connected'],
        'etcs_connected': row['ETCS Connected'],
        'end_event': row['End Event'],
        'end_cause': row['End Cause'],
        'imei': row['IMEI'],
        
        'label': 0  # Normal
    }
    
    normal_sessions.append(normal)

# ══════════════════════════════════════════════════════════════
# COMBINER
# ══════════════════════════════════════════════════════════════

print("\n🔀 Combinaison datasets...")

all_sessions = matched_sessions + normal_sessions

df_final = pd.DataFrame(all_sessions)

# Mélanger
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n✅ Dataset final :")
print(f"   Total       : {len(df_final):,} sessions")
print(f"   Déconnexion : {(df_final['label']==1).sum():,} ({(df_final['label']==1).sum()/len(df_final)*100:.1f}%)")
print(f"   Normal      : {(df_final['label']==0).sum():,} ({(df_final['label']==0).sum()/len(df_final)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

print("\n⚙️  Feature engineering...")

# Parser durées
def parse_duration_to_seconds(duration_str):
    if pd.isna(duration_str):
        return np.nan
    
    total_seconds = 0
    duration_str = str(duration_str)
    
    if 'h' in duration_str:
        match = re.search(r'(\d+)h', duration_str)
        if match:
            total_seconds += int(match.group(1)) * 3600
    
    if 'min' in duration_str:
        match = re.search(r'(\d+)min', duration_str)
        if match:
            total_seconds += int(match.group(1)) * 60
    
    if 's' in duration_str:
        match = re.search(r'(\d+)s', duration_str)
        if match:
            total_seconds += int(match.group(1))
    
    if 'ms' in duration_str:
        match = re.search(r'(\d+)ms', duration_str)
        if match:
            total_seconds += int(match.group(1)) / 1000
    
    return total_seconds

df_final['call_setup_sec'] = df_final['call_setup_duration'].apply(parse_duration_to_seconds)
df_final['transaction_sec'] = df_final['transaction_duration'].apply(parse_duration_to_seconds)

# Temporelles
df_final['hour'] = pd.to_datetime(df_final['timestamp']).dt.hour
df_final['day_of_week'] = pd.to_datetime(df_final['timestamp']).dt.dayofweek
df_final['is_weekend'] = df_final['day_of_week'].isin([5, 6]).astype(int)
df_final['is_night'] = df_final['hour'].between(0, 6).astype(int)

# Durées anormales
df_final['duration_very_short'] = (df_final['transaction_sec'] < 60).astype(int)
df_final['duration_very_long'] = (df_final['transaction_sec'] > 3600).astype(int)
df_final['slow_setup'] = (df_final['call_setup_sec'] > 5).astype(int)

# Km
df_final['km_value'] = pd.to_numeric(df_final['km'], errors='coerce')

print("   ✅ Features créées")

# ══════════════════════════════════════════════════════════════
# SAUVEGARDER
# ══════════════════════════════════════════════════════════════

print("\n💾 Sauvegarde...")

import os
os.makedirs('data/processed', exist_ok=True)

output_file = 'data/processed/fusion_ertms_expandium.csv'
df_final.to_csv(output_file, index=False)

print(f"   ✅ {output_file}")

# ══════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("✅ FUSION TERMINÉE !")
print("="*70)

print(f"""
📊 DATASET FUSION ERTMS + EXPANDIUM :
   Total sessions    : {len(df_final):,}
   
   Sources :
   • ERTMS matchées  : {len(matched_sessions):,}
   • Expandium norm. : {len(normal_sessions):,}
   
   Labels :
   • Déconnexion (1) : {(df_final['label']==1).sum():,} ({(df_final['label']==1).sum()/len(df_final)*100:.1f}%)
   • Normal (0)      : {(df_final['label']==0).sum():,} ({(df_final['label']==0).sum()/len(df_final)*100:.1f}%)
   
   Features disponibles :
   • ERTMS : Km, Zone, Cause, RxQual, RxLev
   • Expandium : Durées, End Event, End Cause
   • Combinées : {len(df_final.columns)} colonnes

📁 FICHIER CRÉÉ :
   {output_file}

🚀 PROCHAINE ÉTAPE :
   Entraîner modèle sur ce dataset fusionné
   
   Commande : python train_rf_fusion.py
""")

print("="*70)