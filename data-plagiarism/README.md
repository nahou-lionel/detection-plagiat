# Corpus de plagiat

Ce dossier doit contenir les fichiers texte du corpus utilisés pour l'entraînement et l'évaluation.

## Téléchargement

1. Télécharger le corpus :
   https://ecampus-vert.unicaen.fr/mod/resource/view.php?id=278104

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
