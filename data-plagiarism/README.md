# Corpus de plagiat (PAN)

Ce dossier doit contenir les fichiers texte du corpus PAN utilisés pour l'entraînement et l'évaluation.

## Téléchargement

1. Télécharger le corpus depuis la compétition PAN :
   https://pan.webis.de/clef09/pan09-web/plagiarism-detection.html

2. Extraire l'archive et copier **tous les fichiers `.txt`** dans ce dossier (`data-plagiarism/`).

## Structure attendue

```
data-plagiarism/
├── orig_taska.txt
├── orig_taskb.txt
├── orig_taskc.txt
├── orig_taskd.txt
├── g0pA_taska.txt
├── g0pA_taskb.txt
├── ...
└── file_information.csv
```

## Note

Les fichiers du corpus ne sont pas versionnés dans ce dépôt pour des raisons de licence.
La liste des paires annotées se trouve dans `data/labels.csv`.
