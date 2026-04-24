"""
Preparation donnees Expandium pour 1D-CNN
Adapte aux vraies colonnes Expandium
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PREPARATION DONNEES 1D-CNN - EXPANDIUM")
print("="*70)

# ETAPE 1 : CHARGER FICHIERS
print("\nETAPE 1 : Chargement fichiers Expandium...")

dfs = []

for year in [2024, 2025, 2026]:
    filename = f'Expandium_{year}.csv'
    try:
        df_temp = pd.read_csv(filename, sep=';', skiprows=1, low_memory=False)
        print(f"OK {year} : {len(df_temp):,} sessions")
        dfs.append(df_temp)
    except FileNotFoundError:
        print(f"ATTENTION {year} : Fichier non trouve")
    except Exception as e:
        print(f"ERREUR {year} : {e}")

if len(dfs) == 0:
    print("\nERREUR : Aucun fichier charge")
    exit()

df = pd.concat(dfs, ignore_index=True)
print(f"\nFusion : {len(df):,} sessions totales")

avant = len(df)
df = df.drop_duplicates()
print(f"Doublons supprimes : {avant - len(df)}")

# ETAPE 2 : PARSING TIMESTAMPS
print("\nETAPE 2 : Parsing timestamps...")

df['Start Time'] = pd.to_datetime(df['Start Time'], errors='coerce')
avant = len(df)
df = df.dropna(subset=['Start Time'])
print(f"Lignes sans timestamp supprimees : {avant - len(df)}")

# ETAPE 3 : PARSING DUREES (format special "15min 57s 948ms")
print("\nETAPE 3 : Parsing durees...")

def parse_expandium_duration(duration_str):
    """
    Parse format Expandium : "15min 57s 948ms" -> secondes
    """
    if pd.isna(duration_str):
        return np.nan
    
    try:
        duration_str = str(duration_str).strip()
        total_seconds = 0.0
        
        # Extraire minutes
        min_match = re.search(r'(\d+)min', duration_str)
        if min_match:
            total_seconds += int(min_match.group(1)) * 60
        
        # Extraire secondes
        sec_match = re.search(r'(\d+)s', duration_str)
        if sec_match:
            total_seconds += int(sec_match.group(1))
        
        # Extraire millisecondes
        ms_match = re.search(r'(\d+)ms', duration_str)
        if ms_match:
            total_seconds += int(ms_match.group(1)) / 1000.0
        
        return total_seconds
    except:
        return np.nan

df['call_setup_duration_sec'] = df['Call Setup Duration (ms)'].apply(parse_expandium_duration)
df['transaction_duration_sec'] = df['Transaction Duration (ms)'].apply(parse_expandium_duration)

print("Durees parsees")
print(f"  call_setup moyenne : {df['call_setup_duration_sec'].mean():.2f}s")
print(f"  transaction moyenne : {df['transaction_duration_sec'].mean():.2f}s")

# ETAPE 4 : LABELLISATION
print("\nETAPE 4 : Labellisation...")

df['is_disconnect'] = 0

# Critere 1 : End Event = 'Disconnect'
mask1 = df['End Event'].astype(str).str.contains('Disconnect', case=False, na=False)
df.loc[mask1, 'is_disconnect'] = 1
print(f"End Event 'Disconnect' : {mask1.sum()}")

# Critere 2 : ETCS Connected != 'Connected'
mask2 = ~df['ETCS Connected'].astype(str).str.contains('Connected', case=False, na=False)
df.loc[mask2, 'is_disconnect'] = 1
print(f"ETCS non connecte : {mask2.sum()}")

# Critere 3 : GSM-R Connected != 'Connected'
if 'GSM-R Connected' in df.columns:
    mask3 = ~df['GSM-R Connected'].astype(str).str.contains('Connected', case=False, na=False)
    df.loc[mask3, 'is_disconnect'] = 1
    print(f"GSM-R non connecte : {mask3.sum()}")

n_disconnect = df['is_disconnect'].sum()
n_normal = len(df) - n_disconnect

print(f"\nLabels crees :")
print(f"  Normal       : {n_normal:,} ({n_normal/len(df)*100:.1f}%)")
print(f"  Deconnexion  : {n_disconnect:,} ({n_disconnect/len(df)*100:.1f}%)")

if n_disconnect == 0:
    print("\nERREUR : Aucune deconnexion detectee")
    exit()

if n_disconnect < 50:
    print(f"\nATTENTION : Seulement {n_disconnect} deconnexions")
    print("Le modele risque de mal performer")

# ETAPE 5 : FEATURE ENGINEERING
print("\nETAPE 5 : Feature engineering...")

df['hour'] = df['Start Time'].dt.hour
df['day_of_week'] = df['Start Time'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

df['duration_very_short'] = (df['transaction_duration_sec'] < 60).astype(int)
df['duration_very_long'] = (df['transaction_duration_sec'] > 3600).astype(int)
df['slow_setup'] = (df['call_setup_duration_sec'] > 5).astype(int)

FEATURES = [
    'call_setup_duration_sec',
    'transaction_duration_sec',
    'hour',
    'day_of_week',
    'is_weekend',
    'is_night',
    'duration_very_short',
    'duration_very_long',
    'slow_setup'
]

avant = len(df)
df = df.dropna(subset=FEATURES + ['is_disconnect'])
print(f"Lignes avec NaN supprimees : {avant - len(df)}")
print(f"{len(FEATURES)} features creees")
print(f"Dataset final : {len(df):,} sessions")

# ETAPE 6 : EQUILIBRAGE
print("\nETAPE 6 : Equilibrage dataset...")

df_disconnect = df[df['is_disconnect'] == 1].copy()
df_normal = df[df['is_disconnect'] == 0].copy()

n_min = min(len(df_disconnect), len(df_normal))

df_normal_sampled = df_normal.sample(n=n_min, random_state=42)
df_balanced = pd.concat([df_disconnect, df_normal_sampled], ignore_index=True)
df_balanced = df_balanced.sort_values('Start Time').reset_index(drop=True)

print(f"Dataset equilibre :")
print(f"  Deconnexions : {n_min:,}")
print(f"  Normales     : {n_min:,}")
print(f"  Total        : {len(df_balanced):,}")

df_balanced.to_csv('Expandium_BALANCED.csv', index=False)
print("Sauvegarde : Expandium_BALANCED.csv")

# ETAPE 7 : NORMALISATION
print("\nETAPE 7 : Normalisation...")

X = df_balanced[FEATURES].values
y = df_balanced['is_disconnect'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Normalisation OK")

with open('scaler_1dcnn.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler sauvegarde")

# ETAPE 8 : SEQUENCES TEMPORELLES
print("\nETAPE 8 : Creation sequences temporelles...")

WINDOW_SIZE = 15
STEP_SIZE = 3

print(f"  Window size : {WINDOW_SIZE}")
print(f"  Step size   : {STEP_SIZE}")

def create_sequences(X, y, window_size, step_size):
    X_sequences = []
    y_sequences = []
    
    for i in range(0, len(X) - window_size + 1, step_size):
        X_window = X[i:i + window_size]
        y_label = y[i + window_size - 1]
        
        X_sequences.append(X_window)
        y_sequences.append(y_label)
    
    return np.array(X_sequences), np.array(y_sequences)

X_sequences, y_sequences = create_sequences(X_scaled, y, WINDOW_SIZE, STEP_SIZE)

print(f"\nSequences creees :")
print(f"  Nombre : {len(X_sequences):,}")
print(f"  Shape  : {X_sequences.shape}")

unique, counts = np.unique(y_sequences, return_counts=True)
print(f"\nDistribution :")
for label, count in zip(unique, counts):
    classe = "Normal" if label == 0 else "Deconnexion"
    print(f"  {classe} : {count:,} ({count/len(y_sequences)*100:.1f}%)")

# ETAPE 9 : SPLIT
print("\nETAPE 9 : Split train/val/test (temporel)...")

n_total = len(X_sequences)
n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)

X_train = X_sequences[:n_train]
y_train = y_sequences[:n_train]
X_val = X_sequences[n_train:n_train+n_val]
y_val = y_sequences[n_train:n_train+n_val]
X_test = X_sequences[n_train+n_val:]
y_test = y_sequences[n_train+n_val:]

print(f"  Train : {len(X_train):,} ({len(X_train)/n_total*100:.0f}%)")
print(f"  Val   : {len(X_val):,} ({len(X_val)/n_total*100:.0f}%)")
print(f"  Test  : {len(X_test):,} ({len(X_test)/n_total*100:.0f}%)")

# ETAPE 10 : SAUVEGARDER
print("\nETAPE 10 : Sauvegarde fichiers .npy...")

np.save('X_train_sequences.npy', X_train)
np.save('y_train_sequences.npy', y_train)
np.save('X_val_sequences.npy', X_val)
np.save('y_val_sequences.npy', y_val)
np.save('X_test_sequences.npy', X_test)
np.save('y_test_sequences.npy', y_test)

print("OK - Tous les fichiers sauvegardes")

# RESUME
print("\n" + "="*70)
print("PREPARATION TERMINEE - PRET POUR 1D-CNN")
print("="*70)
print(f"""
DONNEES PREPAREES :

Source : Expandium 2026
Sessions totales : {len(df):,}
Dataset equilibre : {len(df_balanced):,} (50/50)

Sequences temporelles :
  Nombre      : {len(X_sequences):,}
  Window size : {WINDOW_SIZE}
  Step size   : {STEP_SIZE}
  Features    : {len(FEATURES)}
  Shape       : {X_sequences.shape}

Split :
  Train : {len(X_train):,} sequences
  Val   : {len(X_val):,} sequences
  Test  : {len(X_test):,} sequences

Fichiers crees :
  - X_train_sequences.npy
  - y_train_sequences.npy
  - X_val_sequences.npy
  - y_val_sequences.npy
  - X_test_sequences.npy
  - y_test_sequences.npy
  - scaler_1dcnn.pkl
  - Expandium_BALANCED.csv

PROCHAINE ETAPE :
python train_1dcnn.py
""")
print("="*70)