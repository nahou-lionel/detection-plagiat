# Détection de Plagiat

Système de détection de plagiat par apprentissage automatique. Classifie des paires de documents texte selon leur degré de similarité : `heavy`, `light`, `cut`, `non`.

## Fonctionnalités

- Prétraitement NLP : nettoyage, tokenisation, suppression des stopwords, lemmatisation (spaCy)
- Extraction de features : TF-IDF, similarité cosinus, embeddings, n-grams
- 5 classificateurs interchangeables : Random Forest (défaut), Gradient Boosting, Naive Bayes, SVM, Régression logistique
- Évaluation complète : accuracy, cross-validation, rapport de classification, feature importance (SHAP)

## Prérequis

```bash
pip install -r requirements.txt

# Modèles spaCy (une seule fois)
python -m spacy download en_core_web_md
python -m spacy download fr_core_news_sm
```

## Données

Le corpus (95 paires de documents `.txt`) est versionné dans `data-plagiarism/`. `data/labels.csv` contient les paires annotées :

```
doc_source,doc_suspect,label
orig_taska.txt,g0pC_taska.txt,heavy
orig_taska.txt,g1pD_taska.txt,light
...
```

## Utilisation

```bash
cd src

# Test rapide (train/test split, affiche l'accuracy)
python test_quick.py

# Évaluation complète (cross-validation, rapport, SHAP)
python evaluation.py
```

Les graphiques et rapports générés par `evaluation.py` sont sauvegardés dans `data/processed/` (matrices de confusion, courbes SHAP, résultats de la grille exhaustive).

## Architecture

```
src/
├── preprocessing.py      # TextPreprocessor : nettoyage, tokenisation, lemmatisation
├── feature_extraction.py # Extraction de features sur les paires de documents
├── models.py             # PlagiarismClassifier : wrapper sklearn (5 algorithmes)
├── evaluation.py         # Pipeline complet : entraînement, cross-val, rapport
└── test_quick.py         # Test rapide train/test split

data/
└── labels.csv            # Paires annotées (source, suspect, label)

data-plagiarism/          # Fichiers .txt du corpus (non versionnés)
```

## Classificateurs disponibles

Le projet implémente 5 algorithmes de classification interchangeables. Chacun a des performances différentes selon la taille du corpus et la nature des features :

| Clé                   | Algorithme             | Caractéristiques                                   |
| --------------------- | ---------------------- | -------------------------------------------------- |
| `random_forest`       | Random Forest (défaut) | Bon équilibre précision/rapidité, robuste au bruit |
| `gradient_boosting`   | Gradient Boosting      | Plus précis mais plus lent à entraîner             |
| `svm`                 | SVM (noyau RBF)        | Efficace sur des petits corpus                     |
| `naive_bayes`         | Naive Bayes            | Très rapide, moins précis                          |
| `logistic_regression` | Régression logistique  | Résultats interprétables                           |

Pour changer de classificateur, modifier la ligne suivante dans `src/test_quick.py` (test rapide) :

```python
clf = PlagiarismClassifier('random_forest')  # remplacer par l'une des clés ci-dessus
```
