import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from feature_extraction import extract_features_from_pairs
from models import PlagiarismClassifier

# Charger
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)
root_dir    = os.path.dirname(src_dir)
LABELS_PATH = os.path.join(root_dir, 'data', 'labels.csv')
DATA_DIR    = os.path.join(root_dir, 'data-plagiarism')

pairs_df = pd.read_csv(LABELS_PATH)
print(f"Paires: {len(pairs_df)}")

# Extraire
X, y, names = extract_features_from_pairs(pairs_df, DATA_DIR)
print(f"Features: {X.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Entraîner
clf = PlagiarismClassifier('random_forest')
clf.train(X_train, y_train)

# Évaluer
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")