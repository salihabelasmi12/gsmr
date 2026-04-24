"""
Entraînement Random Forest - Données Expandium
Prédiction Déconnexions vs Normales
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
import json

print("="*70)
print("🌲 ENTRAÎNEMENT RANDOM FOREST - EXPANDIUM")
print("="*70)

# ══════════════════════════════════════════════════════════════
# CHARGER DONNÉES
# ══════════════════════════════════════════════════════════════

print("\n📥 Chargement données...")

df = pd.read_csv('data/processed/expandium_ml_balanced.csv')

print(f"✅ {len(df):,} lignes chargées")

# Features et labels
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

X = df[FEATURES].values
y = df['is_disconnect'].values

print(f"\n📊 Distribution :")
print(f"   NORMAL (0)      : {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
print(f"   DÉCONNEXION (1) : {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")

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
    n_estimators=200,
    max_depth=15,
    min_samples_split=3,
    min_samples_leaf=1,
    class_weight='balanced',
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

# Rapport détaillé
print("\n📊 RAPPORT CLASSIFICATION (Test Set) :\n")
print(classification_report(y_test, y_pred_test, 
                          target_names=['NORMAL', 'DÉCONNEXION'],
                          digits=3))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_test)
print("\n📊 MATRICE DE CONFUSION (Test Set) :")
print(f"                 Prédit NORMAL  Prédit DÉCONNEXION")
print(f"Réel NORMAL           {cm[0,0]:6}            {cm[0,1]:6}")
print(f"Réel DÉCONNEXION      {cm[1,0]:6}            {cm[1,1]:6}")

# Taux fausses alertes
far = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
print(f"\n🚨 Taux fausses alertes : {far*100:.2f}%")

# ══════════════════════════════════════════════════════════════
# IMPORTANCE FEATURES
# ══════════════════════════════════════════════════════════════

print("\n📊 Importance des features...")

feature_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 features :")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:35} : {row['importance']:.4f}")

# ══════════════════════════════════════════════════════════════
# SAUVEGARDER
# ══════════════════════════════════════════════════════════════

print("\n💾 Sauvegarde modèle...")

os.makedirs('models', exist_ok=True)

with open('models/rf_expandium_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("   ✅ models/rf_expandium_model.pkl")

# Sauvegarder métriques
metrics = {
    'test_accuracy': float(test_metrics[0]),
    'test_precision': float(test_metrics[1]),
    'test_recall': float(test_metrics[2]),
    'test_f1': float(test_metrics[3]),
    'false_alert_rate': float(far),
    'features': FEATURES
}

with open('models/rf_expandium_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("   ✅ models/rf_expandium_metrics.json")

# ══════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════

print("\n📊 Création visualisations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Random Forest - Prédiction Déconnexions Expandium', fontsize=16, fontweight='bold')

# 1. Matrice de confusion
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NORMAL', 'DÉCONNEXION'],
            yticklabels=['NORMAL', 'DÉCONNEXION'],
            ax=axes[0, 0], cbar_kws={'label': 'Count'})
axes[0, 0].set_title('Matrice de Confusion (Test)', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Réel')
axes[0, 0].set_xlabel('Prédit')

# 2. Importance features (Top 10)
top_features = feature_importance.head(10)
axes[0, 1].barh(range(len(top_features)), top_features['importance'], color='steelblue', edgecolor='black')
axes[0, 1].set_yticks(range(len(top_features)))
axes[0, 1].set_yticklabels(top_features['feature'], fontsize=9)
axes[0, 1].set_title('Top 10 Features Importance', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Importance')
axes[0, 1].grid(True, alpha=0.3, axis='x')
axes[0, 1].invert_yaxis()

# 3. Comparaison métriques
metrics_df = pd.DataFrame({
    'Train': train_metrics,
    'Val': val_metrics,
    'Test': test_metrics
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

metrics_df.plot(kind='bar', ax=axes[1, 0], color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
axes[1, 0].set_title('Comparaison Métriques', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_ylim([0, 1.05])
axes[1, 0].legend(loc='lower right')
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=0)

# 4. Métriques finales (barres)
final_metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Score': [test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3]]
})

bars = axes[1, 1].barh(final_metrics['Metric'], final_metrics['Score'], 
                       color=['#2ecc71' if s >= 0.8 else '#f39c12' if s >= 0.7 else '#e74c3c' for s in final_metrics['Score']],
                       edgecolor='black')
axes[1, 1].set_title('Performance Finale (Test Set)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Score')
axes[1, 1].set_xlim([0, 1.05])
axes[1, 1].grid(True, alpha=0.3, axis='x')

for i, v in enumerate(final_metrics['Score']):
    axes[1, 1].text(v + 0.02, i, f'{v*100:.1f}%', va='center', fontweight='bold', fontsize=11)

plt.tight_layout()

os.makedirs('results', exist_ok=True)
viz_path = 'results/rf_expandium_results.png'
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"   ✅ {viz_path}")

plt.close()

# ══════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("✅ ENTRAÎNEMENT TERMINÉ !")
print("="*70)

print(f"""
🌲 RANDOM FOREST - EXPANDIUM :
   Arbres         : 200
   Profondeur max : 15
   Dataset        : 1,938 sessions (équilibré)
   
📊 PERFORMANCE (Test Set) :
   Accuracy       : {test_metrics[0]*100:.2f}%  ← Précision globale
   Precision      : {test_metrics[1]*100:.2f}%  ← Fiabilité alertes
   Recall         : {test_metrics[2]*100:.2f}%  ← Détection déconnexions
   F1-Score       : {test_metrics[3]*100:.2f}%  ← Score équilibré
   Fausses alertes: {far*100:.2f}%

📁 FICHIERS CRÉÉS :
   • models/rf_expandium_model.pkl
   • models/rf_expandium_metrics.json
   • results/rf_expandium_results.png

🎯 INTERPRÉTATION :
   Le modèle peut prédire si une session va déconnecter
   avec {test_metrics[0]*100:.1f}% de précision.
   
   Il détecte {test_metrics[2]*100:.1f}% des déconnexions réelles.
   
   Fausses alertes : {far*100:.1f}% (acceptable si <10%)

📊 TOP 3 FEATURES IMPORTANTES :
   1. {feature_importance.iloc[0]['feature']}
   2. {feature_importance.iloc[1]['feature']}
   3. {feature_importance.iloc[2]['feature']}

🚀 VOIR RÉSULTATS :
   start results\\rf_expandium_results.png
   
💡 COMPARAISON :
   Random Forest Excel  : 84.26% accuracy
   Random Forest Expandium : {test_metrics[0]*100:.2f}% accuracy
""")

print("="*70)