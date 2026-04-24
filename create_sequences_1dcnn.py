"""
Création Séquences Temporelles + Entraînement 1D CNN
Pour détection déconnexions GSM-R
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import json

print("="*70)
print("🔄 CRÉATION SÉQUENCES TEMPORELLES + 1D CNN")
print("="*70)

# ══════════════════════════════════════════════════════════════
# CHARGER DONNÉES
# ══════════════════════════════════════════════════════════════

print("\n📥 Chargement données...")

df = pd.read_csv('data/processed/dataset_ml_ready.csv')

print(f"✅ {len(df):,} événements chargés")

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

# ══════════════════════════════════════════════════════════════
# CRÉER SÉQUENCES TEMPORELLES
# ══════════════════════════════════════════════════════════════

print("\n🔄 Création séquences temporelles...")

WINDOW_SIZE = 20  # Chaque séquence = 20 événements consécutifs
STEP_SIZE = 5     # Glissement de 5 événements

def create_sequences(data, labels, window_size, step_size):
    """
    Crée séquences temporelles avec fenêtre glissante
    """
    X_sequences = []
    y_sequences = []
    
    for i in range(0, len(data) - window_size, step_size):
        # Extraire séquence
        sequence = data[i:i+window_size]
        
        # Label = majorité dans la séquence
        label_sequence = labels[i:i+window_size]
        label = int(np.round(np.mean(label_sequence)))
        
        X_sequences.append(sequence)
        y_sequences.append(label)
    
    return np.array(X_sequences), np.array(y_sequences)

# Préparer données
X_data = df[FEATURES].values
y_data = df['is_gsmr_issue'].values

# Créer séquences
X, y = create_sequences(X_data, y_data, WINDOW_SIZE, STEP_SIZE)

print(f"✅ Séquences créées :")
print(f"   Shape X : {X.shape}")
print(f"   Shape y : {y.shape}")
print(f"   ({X.shape[0]} séquences de {X.shape[1]} timesteps × {X.shape[2]} features)")

# Distribution labels
print(f"\n📊 Distribution labels :")
print(f"   BORD (0) : {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
print(f"   GSMR (1) : {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# SPLIT DONNÉES
# ══════════════════════════════════════════════════════════════

print("\n📊 Split données...")

# Split temporel (pas aléatoire pour séries temporelles !)
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

print(f"   Train : {len(X_train):,} séquences")
print(f"   Val   : {len(X_val):,} séquences")
print(f"   Test  : {len(X_test):,} séquences")

# ══════════════════════════════════════════════════════════════
# CONSTRUIRE MODÈLE 1D CNN
# ══════════════════════════════════════════════════════════════

print("\n🏗️  Construction modèle 1D CNN...")

def build_1d_cnn(input_shape):
    model = Sequential([
        # Bloc 1
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Bloc 2
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Bloc 3
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Output (2 classes)
        Dense(2, activation='softmax')
    ])
    
    return model

# Créer modèle
input_shape = (WINDOW_SIZE, len(FEATURES))
model = build_1d_cnn(input_shape)

# Compiler
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Résumé
model.summary()

# ══════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════

print("\n🔥 Entraînement du modèle...")

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Entraîner
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n✅ Entraînement terminé")

# ══════════════════════════════════════════════════════════════
# ÉVALUATION
# ══════════════════════════════════════════════════════════════

print("\n📊 Évaluation sur test set...")

# Prédictions
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Métriques
acc = accuracy_score(y_test, y_pred_classes)
prec = precision_score(y_test, y_pred_classes, zero_division=0)
rec = recall_score(y_test, y_pred_classes, zero_division=0)
f1 = f1_score(y_test, y_pred_classes, zero_division=0)

print(f"\n✅ RÉSULTATS TEST SET :")
print(f"   Accuracy  : {acc*100:.2f}%")
print(f"   Precision : {prec*100:.2f}%")
print(f"   Recall    : {rec*100:.2f}%")
print(f"   F1-Score  : {f1*100:.2f}%")

# Rapport détaillé
print("\n📊 RAPPORT CLASSIFICATION :\n")
print(classification_report(y_test, y_pred_classes, 
                          target_names=['BORD', 'GSMR'],
                          digits=3))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_classes)
print("\n📊 MATRICE DE CONFUSION :")
print(f"                 Prédit BORD  Prédit GSMR")
print(f"Réel BORD             {cm[0,0]:6}       {cm[0,1]:6}")
print(f"Réel GSMR             {cm[1,0]:6}       {cm[1,1]:6}")

# ══════════════════════════════════════════════════════════════
# SAUVEGARDER
# ══════════════════════════════════════════════════════════════

print("\n💾 Sauvegarde...")

os.makedirs('models', exist_ok=True)

# Modèle
model_path = 'models/cnn_1d_model.h5'
model.save(model_path)
print(f"   ✅ {model_path}")

# Configuration
config = {
    'window_size': WINDOW_SIZE,
    'step_size': STEP_SIZE,
    'n_features': len(FEATURES),
    'features': FEATURES,
    'test_accuracy': float(acc),
    'test_precision': float(prec),
    'test_recall': float(rec),
    'test_f1': float(f1)
}

with open('models/cnn_1d_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f"   ✅ models/cnn_1d_config.json")

# ══════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════

print("\n📊 Création visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Courbes d'entraînement - Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_title('Évolution de la Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Courbes d'entraînement - Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0, 1].set_title("Évolution de l'Accuracy", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Matrice de confusion
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['BORD', 'GSMR'],
            yticklabels=['BORD', 'GSMR'],
            ax=axes[1, 0])
axes[1, 0].set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Réel')
axes[1, 0].set_xlabel('Prédit')

# 4. Comparaison métriques
metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Score': [acc, prec, rec, f1]
}
metrics_df = pd.DataFrame(metrics_data)

axes[1, 1].barh(metrics_df['Metric'], metrics_df['Score'], color='steelblue', edgecolor='black')
axes[1, 1].set_title('Métriques de Performance', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Score')
axes[1, 1].set_xlim([0, 1])
axes[1, 1].grid(True, alpha=0.3, axis='x')

for i, v in enumerate(metrics_df['Score']):
    axes[1, 1].text(v + 0.02, i, f'{v*100:.1f}%', va='center', fontweight='bold')

plt.tight_layout()

os.makedirs('results', exist_ok=True)
viz_path = 'results/cnn_1d_results.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"   ✅ {viz_path}")

plt.close()

# ══════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("✅ ENTRAÎNEMENT 1D CNN TERMINÉ !")
print("="*70)

print(f"""
🔄 SÉQUENCES TEMPORELLES :
   Window size    : {WINDOW_SIZE} événements
   Step size      : {STEP_SIZE} événements
   Séquences      : {len(X):,}
   
🏗️  ARCHITECTURE 1D CNN :
   Input          : ({WINDOW_SIZE}, {len(FEATURES)})
   Conv1D layers  : 3 (64, 128, 256 filtres)
   Dense layers   : 2 (128, 64 neurones)
   
📊 PERFORMANCE (Test Set) :
   Accuracy       : {acc*100:.2f}%
   Precision      : {prec*100:.2f}%
   Recall         : {rec*100:.2f}%
   F1-Score       : {f1*100:.2f}%
   
📁 FICHIERS CRÉÉS :
   • models/cnn_1d_model.h5
   • models/cnn_1d_config.json
   • results/cnn_1d_results.png

🎯 COMPARAISON :
   Random Forest  : 84.26% accuracy
   1D CNN         : {acc*100:.2f}% accuracy
   
💡 INTERPRÉTATION :
   Le 1D CNN analyse des séquences de {WINDOW_SIZE} événements
   pour détecter des patterns temporels dans les déconnexions.
   
🚀 VOIR RÉSULTATS :
   start results\\cnn_1d_results.png
""")

print("="*70)