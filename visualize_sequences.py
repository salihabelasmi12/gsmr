"""
Visualisation et analyse des sequences temporelles pour 1D-CNN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("="*70)
print("ANALYSE SEQUENCES TEMPORELLES - 1D-CNN")
print("="*70)

# CHARGER LES SEQUENCES
print("\nChargement sequences...")

X_train = np.load('X_train_sequences.npy')
y_train = np.load('y_train_sequences.npy')
X_val = np.load('X_val_sequences.npy')
y_val = np.load('y_val_sequences.npy')
X_test = np.load('X_test_sequences.npy')
y_test = np.load('y_test_sequences.npy')

print(f"\nTrain : {X_train.shape}")
print(f"Val   : {X_val.shape}")
print(f"Test  : {X_test.shape}")

# FEATURES NAMES
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

# ANALYSE QUANTITATIVE
print("\n" + "="*70)
print("ANALYSE QUANTITATIVE")
print("="*70)

n_total = len(X_train) + len(X_val) + len(X_test)
window_size = X_train.shape[1]
n_features = X_train.shape[2]

print(f"\nNombre total sequences : {n_total}")
print(f"  Train : {len(X_train)} ({len(X_train)/n_total*100:.1f}%)")
print(f"  Val   : {len(X_val)} ({len(X_val)/n_total*100:.1f}%)")
print(f"  Test  : {len(X_test)} ({len(X_test)/n_total*100:.1f}%)")

print(f"\nDimensions sequences :")
print(f"  Window size : {window_size} evenements")
print(f"  Features    : {n_features}")
print(f"  Shape       : ({n_total}, {window_size}, {n_features})")

# Distribution labels
n_train_disconnect = y_train.sum()
n_val_disconnect = y_val.sum()
n_test_disconnect = y_test.sum()

print(f"\nDistribution labels (Train) :")
print(f"  Normal       : {len(y_train) - n_train_disconnect} ({(len(y_train)-n_train_disconnect)/len(y_train)*100:.1f}%)")
print(f"  Deconnexion  : {n_train_disconnect} ({n_train_disconnect/len(y_train)*100:.1f}%)")

print(f"\nDistribution labels (Test) :")
print(f"  Normal       : {len(y_test) - n_test_disconnect} ({(len(y_test)-n_test_disconnect)/len(y_test)*100:.1f}%)")
print(f"  Deconnexion  : {n_test_disconnect} ({n_test_disconnect/len(y_test)*100:.1f}%)")

# EVALUATION SUFFISANCE
print("\n" + "="*70)
print("EVALUATION SUFFISANCE POUR 1D-CNN")
print("="*70)

def evaluate_dataset_sufficiency(n_train, n_val, n_test, window_size, n_features):
    """Evalue si le dataset est suffisant pour 1D-CNN"""
    
    print(f"\nCRITERE 1 : Nombre minimum sequences")
    min_required = 500
    status1 = "OK" if n_total >= min_required else "INSUFFISANT"
    print(f"  Requis  : >={min_required} sequences")
    print(f"  Actuel  : {n_total} sequences")
    print(f"  Statut  : {status1}")
    
    print(f"\nCRITERE 2 : Equilibre classes (train)")
    ratio = n_train_disconnect / len(y_train)
    status2 = "OK" if 0.4 <= ratio <= 0.6 else "ATTENTION"
    print(f"  Ideal   : 40-60% deconnexions")
    print(f"  Actuel  : {ratio*100:.1f}% deconnexions")
    print(f"  Statut  : {status2}")
    
    print(f"\nCRITERE 3 : Taille window vs features")
    ratio_window_features = window_size / n_features
    status3 = "OK" if ratio_window_features >= 1.5 else "LIMITE"
    print(f"  Ideal   : window_size >= 1.5 * n_features")
    print(f"  Actuel  : {window_size} / {n_features} = {ratio_window_features:.1f}")
    print(f"  Statut  : {status3}")
    
    print(f"\nCRITERE 4 : Train set size")
    min_train = 300
    status4 = "OK" if n_train >= min_train else "INSUFFISANT"
    print(f"  Requis  : >={min_train} sequences train")
    print(f"  Actuel  : {n_train} sequences train")
    print(f"  Statut  : {status4}")
    
    print(f"\nCRITERE 5 : Test set size")
    min_test = 50
    status5 = "OK" if n_test >= min_test else "INSUFFISANT"
    print(f"  Requis  : >={min_test} sequences test")
    print(f"  Actuel  : {n_test} sequences test")
    print(f"  Statut  : {status5}")
    
    # Score global
    statuses = [status1, status2, status3, status4, status5]
    n_ok = statuses.count("OK")
    
    print(f"\n" + "="*70)
    print(f"SCORE GLOBAL : {n_ok}/5 criteres OK")
    print("="*70)
    
    if n_ok >= 4:
        print("\nCONCLUSION : Dataset SUFFISANT pour 1D-CNN")
        print("Performance attendue : 75-90% accuracy")
    elif n_ok >= 3:
        print("\nCONCLUSION : Dataset ACCEPTABLE pour 1D-CNN")
        print("Performance attendue : 65-80% accuracy")
        print("Recommandation : Collecter plus de donnees si possible")
    else:
        print("\nCONCLUSION : Dataset INSUFFISANT pour 1D-CNN")
        print("Performance attendue : <70% accuracy")
        print("URGENT : Collecter plus de donnees")
    
    return n_ok >= 3

evaluate_dataset_sufficiency(len(X_train), len(X_val), len(X_test), window_size, n_features)

# VISUALISATION EXEMPLES SEQUENCES
print("\n" + "="*70)
print("VISUALISATION SEQUENCES")
print("="*70)

# Trouver 1 sequence normale et 1 sequence deconnexion
idx_normal = np.where(y_train == 0)[0][0]
idx_disconnect = np.where(y_train == 1)[0][0]

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
fig.suptitle('VISUALISATION SEQUENCES TEMPORELLES (Train Set)', 
             fontsize=16, fontweight='bold')

# Selectionner 9 features les plus importantes a visualiser
selected_features = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Toutes

for idx, feature_idx in enumerate(selected_features):
    ax = axes[idx // 3, idx % 3]
    
    # Sequence normale
    ax.plot(X_train[idx_normal, :, feature_idx], 
            'o-', label='Sequence Normale', 
            linewidth=2, markersize=6, color='green', alpha=0.7)
    
    # Sequence deconnexion
    ax.plot(X_train[idx_disconnect, :, feature_idx], 
            's-', label='Sequence Deconnexion', 
            linewidth=2, markersize=6, color='red', alpha=0.7)
    
    ax.set_title(FEATURES[feature_idx], fontweight='bold')
    ax.set_xlabel('Temps (evenement)')
    ax.set_ylabel('Valeur normalisee')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sequences_visualization.png', dpi=300, bbox_inches='tight')
print("\nGraphique sauvegarde : sequences_visualization.png")

# STATISTIQUES PAR FEATURE
print("\n" + "="*70)
print("STATISTIQUES PAR FEATURE")
print("="*70)

print(f"\n{'Feature':<30} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("-"*70)

for i, feature_name in enumerate(FEATURES):
    feature_data = X_train[:, :, i].flatten()
    mean_val = feature_data.mean()
    std_val = feature_data.std()
    min_val = feature_data.min()
    max_val = feature_data.max()
    
    print(f"{feature_name:<30} {mean_val:>9.4f} {std_val:>9.4f} {min_val:>9.4f} {max_val:>9.4f}")

# EXEMPLES DE SEQUENCES
print("\n" + "="*70)
print("EXEMPLES DE SEQUENCES (5 premieres)")
print("="*70)

print("\nSEQUENCE NORMALE (premiere) :")
print(f"Shape : {X_train[idx_normal].shape}")
print("\nEvenement 1 (t=0) :")
for i, feat in enumerate(FEATURES):
    print(f"  {feat:<30} : {X_train[idx_normal, 0, i]:.4f}")

print("\nEvenement 15 (t=14) :")
for i, feat in enumerate(FEATURES):
    print(f"  {feat:<30} : {X_train[idx_normal, 14, i]:.4f}")

print(f"\nLabel : {y_train[idx_normal]} (0=Normal)")

print("\n" + "-"*70)

print("\nSEQUENCE DECONNEXION (premiere) :")
print(f"Shape : {X_train[idx_disconnect].shape}")
print("\nEvenement 1 (t=0) :")
for i, feat in enumerate(FEATURES):
    print(f"  {feat:<30} : {X_train[idx_disconnect, 0, i]:.4f}")

print("\nEvenement 15 (t=14) :")
for i, feat in enumerate(FEATURES):
    print(f"  {feat:<30} : {X_train[idx_disconnect, 14, i]:.4f}")

print(f"\nLabel : {y_train[idx_disconnect]} (1=Deconnexion)")

# SAUVEGARDER ECHANTILLON
print("\n" + "="*70)
print("SAUVEGARDE ECHANTILLON CSV")
sample.to_csv('sample_sequences.csv', index=False)
print("Echantillon sauvegarde : sample_sequences.csv")