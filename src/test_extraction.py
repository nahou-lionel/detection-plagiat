"""
Test rapide de l'extraction de features avec les données existantes
"""

import pandas as pd
import os
import sys
from preprocessing import TextPreprocessor
from feature_extraction import extract_features_from_pairs

print("="*70)
print("TEST DE L'EXTRACTION DE FEATURES")
print("="*70)

# === CONFIGURATION ===
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)
root_dir    = os.path.dirname(src_dir)
LABELS_PATH = os.path.join(root_dir, 'data', 'labels.csv')
DATA_DIR    = os.path.join(root_dir, 'data-plagiarism')
# DATA_DIR = '../data-plagiarism'          # Répertoire avec vos documents
# LABELS_PATH = '../data/labels.csv'  # Votre fichier labels

# === 1. VÉRIFIER LES FICHIERS ===
print("\n[1/4] Vérification des fichiers...")

if not os.path.exists(LABELS_PATH):
    print(f" Fichier {LABELS_PATH} introuvable !")
    exit(1)

if not os.path.exists(DATA_DIR):
    print(f"Répertoire {DATA_DIR} introuvable !")
    exit(1)

print(f" Fichier labels trouvé: {LABELS_PATH}")
print(f" Répertoire data trouvé: {DATA_DIR}")

# === 2. CHARGER UN ÉCHANTILLON ===
print("\n[2/4] Chargement des labels...")

# Charger toutes les paires
all_pairs = pd.read_csv(LABELS_PATH)
print(f"✓ Total: {len(all_pairs)} paires dans le fichier")

# Distribution des classes
print("\n📊 Distribution des classes:")
print(all_pairs['label'].value_counts())

# Prendre un PETIT échantillon (2 paires par classe)
sample_pairs = pd.DataFrame()

for label in all_pairs['label'].unique():
    label_pairs = all_pairs[all_pairs['label'] == label]
    n_samples = min(100, len(label_pairs))  # Max 2 par classe
    sample_pairs = pd.concat([sample_pairs, label_pairs.head(n_samples)])

print(f"\n🔍 Test sur {len(sample_pairs)} paires échantillonnées:")
print(sample_pairs.to_string(index=False))

# === 3. VÉRIFIER QUE LES FICHIERS EXISTENT ===
print("\n[3/4] Vérification des fichiers de documents...")

all_files = set(sample_pairs['doc_source'].tolist() + 
                sample_pairs['doc_suspect'].tolist())

missing_files = []
for filename in all_files:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        missing_files.append(filename)

if missing_files:
    print(f"{len(missing_files)} fichiers manquants:")
    for f in missing_files:
        print(f"   - {f}")
    exit(1)
else:
    print(f"Tous les {len(all_files)} fichiers sont présents")

# === 4. EXTRACTION DES FEATURES ===
print("\n[4/4] Extraction des features...")

try:
    X, y, feature_names = extract_features_from_pairs(
        sample_pairs,
        DATA_DIR,
        preprocessor=None  # Changez en TextPreprocessor() si vous voulez tester avec prétraitement
    )
    
    print(f"\nEXTRACTION RÉUSSIE !")
    print(f"   Shape: {X.shape}")
    print(f"   Nombre de features: {len(feature_names)}")
    print(f"   Labels: {list(y)}")
    
except Exception as e:
    print(f"\nERREUR lors de l'extraction:")
    print(f"   {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# === 5. AFFICHER LES RÉSULTATS ===
print("\n" + "="*70)
print("RÉSULTATS DE L'EXTRACTION")
print("="*70)

# Créer un DataFrame pour affichage
df_results = pd.DataFrame(X, columns=feature_names)
df_results['label'] = y

# Ajouter les noms de fichiers pour référence
df_results.insert(0, 'doc_source', sample_pairs['doc_source'].values)
df_results.insert(1, 'doc_suspect', sample_pairs['doc_suspect'].values)

print("\nFeatures extraites (5 premières colonnes):")
print(df_results[['doc_source', 'doc_suspect', 'label', 
                   'tfidf_cos', 'jaccard_sim', 
                   'dice_sim']].to_string(index=False))

# === 6. ANALYSE DES RÉSULTATS ===
print("\n" + "="*70)
print("ANALYSE DES RÉSULTATS")
print("="*70)

print("\n🔍 Analyse par paire:")
for idx, row in df_results.iterrows():
    label = row['label']
    tfidf = row['tfidf_cos']
    jaccard = row['jaccard_sim']
    char_sim = row['dice_sim']
    
    print(f"\nPaire {idx+1}: {row['doc_suspect'][:30]}... ({label})")
    print(f"  TF-IDF 1-gram:  {tfidf:.4f}")
    print(f"  Jaccard:        {jaccard:.4f}")
    print(f"  Dice:    {char_sim:.4f}")
    
    # Vérifications de cohérence
    if label == 'heavy':
        if tfidf > 0.8:
            print(f"  ✅ Cohérent (heavy devrait avoir tfidf > 0.8)")
        else:
            print(f"  ⚠️  Inattendu pour heavy (tfidf devrait être > 0.8)")
    
    elif label == 'light':
        if 0.4 < tfidf < 0.85:
            print(f"  ✅ Cohérent (light devrait avoir 0.4 < tfidf < 0.85)")
        else:
            print(f"  ⚠️  Inattendu pour light (tfidf devrait être entre 0.4 et 0.85)")
    
    elif label == 'non':
        if tfidf < 0.5:
            print(f"  ✅ Cohérent (non devrait avoir tfidf < 0.5)")
        else:
            print(f"  ⚠️  Inattendu pour non (tfidf devrait être < 0.5)")
    
    elif label == 'cut':
        if 0.3 < tfidf < 0.8:
            print(f"  ✅ Cohérent (cut devrait avoir 0.3 < tfidf < 0.8)")
        else:
            print(f"  ⚠️  Inattendu pour cut (tfidf devrait être entre 0.3 et 0.8)")

# === 7. STATISTIQUES GLOBALES ===
print("\n" + "="*70)
print("STATISTIQUES PAR CLASSE")
print("="*70)

print("\n📈 Moyennes des features par classe:")
stats = df_results.groupby('label')[['tfidf_cos', 'jaccard_sim', 
                                       'dice_sim',]].mean()
print(stats.round(4).to_string())

# === 8. SAUVEGARDER LES RÉSULTATS DU TEST ===
output_path = 'data/processed/test_features.csv'
os.makedirs('data/processed', exist_ok=True)
df_results.to_csv(output_path, index=False)
print(f"\n💾 Résultats sauvegardés dans: {output_path}")

# === RÉSUMÉ FINAL ===
print("\n" + "="*70)
print("✅ TEST TERMINÉ AVEC SUCCÈS")
print("="*70)

print(f"""
Résumé:
  - Paires testées:     {len(sample_pairs)}
  - Features extraites: {len(feature_names)}
  - Classes présentes:  {list(df_results['label'].unique())}
  - Résultats dans:     {output_path}
  
Prochaines étapes:
  1. Vérifier les résultats dans {output_path}
  2. Si tout est OK, lancer l'extraction complète avec main.py
  3. Entraîner les modèles
""")