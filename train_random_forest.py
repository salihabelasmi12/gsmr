"""
Entraînement Random Forest - Détection Déconnexions GSM-R
Approche classique ML (baseline)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

print("="*70)
print("🌲 ENTRAÎNEMENT RANDOM FOREST")
print("="*70)

# ══════════════════════════════════════════════════════════════
# CHARGER DONNÉES
# ══════════════════════════════════════════════════════════════

print("\n📥 Chargement données...")

df = pd.read_csv('data/processed/dataset_ml_ready.csv')

print(f"✅ {len(df):,} lignes chargées")

# Features et labels
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

X = df[FEATURES].values
y = df['is_gsmr_issue'].values  # Prédire si c'est un problème GSMR

print(f"\nTarget : is_gsmr_issue")
print(f"   GSMR (1)  : {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
print(f"   BORD (0)  : {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# SPLIT DONNÉES
# ══════════════════════════════════════════════════════════════

print("\n📊 Split données...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"   Train : {len(X_train):,}")
print(f"   Val   : {len(X_val):,}")
print(f"   Test  : {len(X_test):,}")

# ══════════════════════════════════════════════════════════════
# ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════

print("\n🌲 Entraînement Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

print("✅ Entraînement terminé")

# ══════════════════════════════════════════════════════════════
# ÉVALUATION
# ══════════════════════════════════════════════════════════════

print("\n📊 Évaluation...")

# Prédictions
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# Métriques
def print_metrics(y_true, y_pred, dataset_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n{dataset_name}:")
    print(f"   Accuracy  : {acc*100:.2f}%")
    print(f"   Precision : {prec*100:.2f}%")
    print(f"   Recall    : {rec*100:.2f}%")
    print(f"   F1-Score  : {f1*100:.2f}%")
    
    return acc, prec, rec, f1

train_metrics = print_metrics(y_train, y_pred_train, "TRAIN")
val_metrics = print_metrics(y_val, y_pred_val, "VALIDATION")
test_metrics = print_metrics(y_test, y_pred_test, "TEST")

# ══════════════════════════════════════════════════════════════
# IMPORTANCE FEATURES
# ══════════════════════════════════════════════════════════════

print("\n📊 Importance des features...")

feature_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 features:")
for idx, row in feature_importance.head().iterrows():
    print(f"   {row['feature']:20} : {row['importance']:.4f}")

# ══════════════════════════════════════════════════════════════
# SAUVEGARDER
# ══════════════════════════════════════════════════════════════

print("\n💾 Sauvegarde modèle...")

os.makedirs('models', exist_ok=True)

with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("   ✅ models/random_forest_model.pkl")

# Sauvegarder métriques
metrics = {
    'test_accuracy': test_metrics[0],
    'test_precision': test_metrics[1],
    'test_recall': test_metrics[2],
    'test_f1': test_metrics[3]
}

import json
with open('models/random_forest_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ══════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════

print("\n📊 Création visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Matrice de confusion
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['BORD', 'GSMR'],
            yticklabels=['BORD', 'GSMR'],
            ax=axes[0, 0])
axes[0, 0].set_title('Matrice de Confusion (Test)', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Réel')
axes[0, 0].set_xlabel('Prédit')

# 2. Importance features
feature_importance.plot(x='feature', y='importance', kind='barh', ax=axes[0, 1], color='steelblue')
axes[0, 1].set_title('Importance des Features', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Importance')

# 3. Comparaison métriques
metrics_df = pd.DataFrame({
    'Train': train_metrics,
    'Val': val_metrics,
    'Test': test_metrics
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

metrics_df.plot(kind='bar', ax=axes[1, 0], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1, 0].set_title('Comparaison Métriques', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].legend(loc='lower right')
axes[1, 0].grid(True, alpha=0.3)

# 4. Distribution prédictions
axes[1, 1].hist([y_test, y_pred_test], label=['Réel', 'Prédit'], bins=2, alpha=0.7)
axes[1, 1].set_title('Distribution Prédictions', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Classe (0=BORD, 1=GSMR)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend()

plt.tight_layout()

os.makedirs('results', exist_ok=True)
plt.savefig('results/random_forest_results.png', dpi=300, bbox_inches='tight')
print("   ✅ results/random_forest_results.png")

plt.close()

# ══════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("✅ ENTRAÎNEMENT TERMINÉ !")
print("="*70)

print(f"""
🌲 RANDOM FOREST :
   Arbres        : 100
   Profondeur max: 10
   
📊 PERFORMANCE (Test Set) :
   Accuracy      : {test_metrics[0]*100:.2f}%
   Precision     : {test_metrics[1]*100:.2f}%
   Recall        : {test_metrics[2]*100:.2f}%
   F1-Score      : {test_metrics[3]*100:.2f}%

📁 FICHIERS CRÉÉS :
   • models/random_forest_model.pkl
   • models/random_forest_metrics.json
   • results/random_forest_results.png

🎯 INTERPRÉTATION :
   Le modèle peut prédire si une déconnexion
   est due au réseau GSMR ou aux équipements BORD
   avec {test_metrics[0]*100:.1f}% de précision.

🚀 PROCHAINE ÉTAPE :
   Voir les résultats : start results\\random_forest_results.png
""")

print("="*70)