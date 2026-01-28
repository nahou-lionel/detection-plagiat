from pathlib import Path

from bow import texts_to_bow, display_bow
from similarity import BowCosineSimilarity

# Répertoires / chemins
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "data-plagiarism"

# --- BoW par phrase (optionnel, juste pour inspection) --------------------
# Charge chaque ligne non vide comme un document distinct
data_lines_a = (DATA_DIR / "orig_taska.txt").read_text(encoding="utf-8").splitlines()
data = [line.strip() for line in data_lines_a if line.strip()]
bow = texts_to_bow(data, stop_words="english")
#display_bow(bow)

# --- Score de similarité global entre les deux fichiers complets ----------
full_a = (DATA_DIR / "orig_taska.txt").read_text(encoding="utf-8")
full_b = (DATA_DIR / "orig_taskb.txt").read_text(encoding="utf-8")

cosine = BowCosineSimilarity(stop_words="english")
cosine.fit([full_a, full_b])
score = cosine.pair_similarity(0, 1)

print(f"Cosine similarity (document complet orig_taska vs orig_taskb) : {score:.3f}")

