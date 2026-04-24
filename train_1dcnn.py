"""
Entrainement modele 1D-CNN pour detection deconnexions GSM-R.s
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ENTRAINEMENT 1D-CNN - DETECTION DECONNEXIONS GSM-R")
print("="*70)

# ETAPE 1 : CHARGER LES SEQUENCES
print("\nETAPE 1 : Chargement sequences...")

X_train = np.load('X_train_sequences.npy')
y_train = np.load('y_train_sequences.npy')
X_val = np.load('X_val_sequences.npy')
y_val = np.load('y_val_sequences.npy')
X_test = np.load('X_test_sequences.npy')
y_test = np.load('y_test_sequences.npy')

print(f"Train : {X_train.shape} sequences")
print(f"Val   : {X_val.shape} sequences")
print(f"Test  : {X_test.shape} sequences")

# ETAPE 2 : CONSTRUIRE ARCHITECTURE 1D-CNN
print("\nETAPE 2 : Construction architecture 1D-CNN...")

window_size = X_train.shape[1]  # 15
n_features = X_train.shape[2]   # 9

print(f"  Input shape : ({window_size}, {n_features})")

model = Sequential([
    # Block 1
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',
           input_shape=(window_size, n_features)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),
    
    # Block 2
    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),
    
    # Block 3 - SIMPLIFIE (pas de pooling)
    Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.4),
    
    # Global Average Pooling au lieu de Flatten
    GlobalAveragePooling1D(),
    
    # Dense layers
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.4),
    
    # Output
    Dense(1, activation='sigmoid')
])

print("\nArchitecture creee")
print(f"Total parametres : {model.count_params():,}")
# ETAPE 3 : COMPILER
print("\nETAPE 3 : Compilation...")

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Modele compile")

# ETAPE 4 : CALLBACKS
print("\nETAPE 4 : Configuration callbacks...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    ),
  # NOUVEAU (format .keras au lieu de .h5)
ModelCheckpoint(
    'best_1dcnn_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=0
)
]

print("Callbacks OK")

# ETAPE 5 : ENTRAINEMENT
print("\nETAPE 5 : Entrainement...")
print("="*70)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*70)
print("ENTRAINEMENT TERMINE")
print("="*70)

# ETAPE 6 : EVALUATION
print("\nETAPE 6 : Evaluation sur test set...")

y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Metriques
test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy : {test_accuracy*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix :")
print(f"  TN={tn}, FP={fp}")
print(f"  FN={fn}, TP={tp}")

# Metriques
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nMetriques :")
print(f"  Precision : {precision*100:.2f}%")
print(f"  Recall    : {recall*100:.2f}%")
print(f"  F1-Score  : {f1*100:.2f}%")

# ETAPE 7 : VISUALISATIONS
print("\nETAPE 7 : Creation graphiques...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Graph 1 : Loss
axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_title('Evolution Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Graph 2 : Accuracy
axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0, 1].set_title('Evolution Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Graph 3 : Confusion Matrix
im = axes[1, 0].imshow(cm, cmap='Blues')
axes[1, 0].set_title('Confusion Matrix (Test)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xticks([0, 1])
axes[1, 0].set_yticks([0, 1])
axes[1, 0].set_xticklabels(['Normal', 'Deconnexion'])
axes[1, 0].set_yticklabels(['Normal', 'Deconnexion'])

for i in range(2):
    for j in range(2):
        text = axes[1, 0].text(j, i, cm[i, j],
                              ha="center", va="center", 
                              color="white" if cm[i, j] > cm.max()/2 else "black",
                              fontsize=20, fontweight='bold')

plt.colorbar(im, ax=axes[1, 0])

# Graph 4 : Metriques
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [test_accuracy, precision, recall, f1]
colors = ['#2ecc71' if v > 0.75 else '#e74c3c' for v in metrics_values]

axes[1, 1].barh(metrics_names, metrics_values, color=colors)
axes[1, 1].set_xlim([0, 1])
axes[1, 1].set_title('Metriques Test Set', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Score')

for i, v in enumerate(metrics_values):
    axes[1, 1].text(v + 0.02, i, f'{v*100:.1f}%', 
                   va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('1dcnn_results.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegarde : 1dcnn_results.png")

# ETAPE 8 : SAUVEGARDER RESULTATS
print("\nETAPE 8 : Sauvegarde resultats...")

results = {
    'test_accuracy': float(test_accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'confusion_matrix': cm.tolist(),
    'epochs_trained': len(history.history['loss'])
}

with open('1dcnn_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Resultats sauvegardes : 1dcnn_results.json")

# RESUME FINAL
print("\n" + "="*70)
print("ENTRAINEMENT 1D-CNN TERMINE")
print("="*70)

print(f"""
RESULTATS FINAUX :

Test Set Performance :
  Accuracy  : {test_accuracy*100:.2f}%
  Precision : {precision*100:.2f}%
  Recall    : {recall*100:.2f}%
  F1-Score  : {f1*100:.2f}%

Confusion Matrix :
  TN={tn}, FP={fp}
  FN={fn}, TP={tp}

Epochs entraines : {len(history.history['loss'])}

Fichiers crees :
  - best_1dcnn_model.h5 (meilleur modele)
  - 1dcnn_results.png (graphiques)
  - 1dcnn_results.json (metriques)

PROCHAINE ETAPE :
Comparer avec Random Forest
""")

print("="*70)