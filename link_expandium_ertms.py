import pandas as pd
import os

print("="*70)
print("🔗 LIAISON EXPANDIUM + INCIDENTS ERTMS")
print("="*70)

# -----------------------------
# 📥 Chemins des fichiers
# -----------------------------
expandium_file = "data/raw/ETCS-Call-tracing_.csv"  # ton fichier ETCS
ertms_file = "data/raw/incidents_ertms.xlsx"       # fichier incidents ERTMS

# Vérifier que les fichiers existent
if not os.path.exists(expandium_file):
    raise FileNotFoundError(f"Le fichier ETCS n'existe pas : {expandium_file}")
if not os.path.exists(ertms_file):
    raise FileNotFoundError(f"Le fichier ERTMS n'existe pas : {ertms_file}")

# -----------------------------
# 📥 Chargement Expandium (ETCS)
# -----------------------------
print("📥 Chargement Expandium...")
exp = pd.read_csv(expandium_file, sep=";", encoding="utf-8", low_memory=False)

# Nettoyer les noms de colonnes
exp.columns = exp.columns.str.strip()

# Afficher les colonnes pour vérifier le nom de la date
print("Colonnes ETCS disponibles :", list(exp.columns))

# Détecter automatiquement la colonne de date
possible_date_cols = [c for c in exp.columns if "start" in c.lower() or "time" in c.lower() or "date" in c.lower()]
if not possible_date_cols:
    raise KeyError("Aucune colonne de date trouvée dans le fichier ETCS.")
date_col = possible_date_cols[0]
print(f"→ Colonne utilisée pour la date : {date_col}")

# Convertir en datetime
exp["Start Time"] = pd.to_datetime(exp[date_col], errors="coerce")  # coerce transforme les erreurs en NaT
print(exp[["Start Time"]].head())

# -----------------------------
# 📥 Chargement incidents ERTMS
# -----------------------------
print("📥 Chargement incidents ERTMS...")
ertms = pd.read_excel(ertms_file)
ertms.columns = ertms.columns.str.strip()
print("Colonnes ERTMS disponibles :", list(ertms.columns))

# -----------------------------
# 🔗 Liaison ETCS ↔ ERTMS
# Exemple : fusion par ID ou timestamp
# -----------------------------
# Attention : adaptez 'Call ID' ou 'Incident ID' selon tes fichiers
if "Call ID" in exp.columns and "Incident ID" in ertms.columns:
    merged = pd.merge(exp, ertms, left_on="Call ID", right_on="Incident ID", how="left")
    print("Fusion réalisée :", merged.shape)
else:
    merged = exp.copy()
    print("Fusion non réalisée : vérifier les colonnes d'identifiants")

# -----------------------------
# 📤 Sauvegarde résultat
# -----------------------------
output_file = "data/processed/merged_etcs_ertms.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
merged.to_csv(output_file, index=False, sep=";")
print(f"✅ Fichier fusionné sauvegardé : {output_file}")