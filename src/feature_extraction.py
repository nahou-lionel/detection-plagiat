# Extract similarity measures between pairs of documents
import os
import re
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
# from .preprocessing import load_document
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import load_document
class FeatureExtractor:
    def __init__(self, language="english"):
        self.language = language


    @staticmethod
    def _char_ngrams(text: str, n: int):
        # char n-grams
        text = re.sub(r"\s+", " ", text.lower()).strip()
        if len(text) < n:
            return {text} if text else set()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    def cosine_sim(
        self,
        doc1,
        doc2,
        *,
        
        preprocessor=None,
        use_char_ngrams=False,
        ngram_range=(1, 1),
        char_ngram_range=(3, 5),
        language=None,
    ) -> float:
        """Calcul de la similarité du cosinus entre deux documents en utilisant  TF-IDF."""
        t1 = doc1
        t2 = doc2

        if preprocessor is not None:
            t1 = preprocessor.preprocess(t1, remove_stopwords=True, lemmatize=False)
            t2 = preprocessor.preprocess(t2, remove_stopwords=True, lemmatize=False)

        lang = language or self.language
        if use_char_ngrams:
            vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=char_ngram_range,
                lowercase=True,
            )
        else:
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=ngram_range,
                lowercase=True,
                # scikit-learn ne fournit que la liste de stopwords anglaise en standard
                stop_words="english" if lang == "english" else None,
            )

        tfidf = vectorizer.fit_transform([t1 or "", t2 or ""])
        return float(cosine_similarity(tfidf[0], tfidf[1])[0, 0])

    def jaccard_sim(
        self,
        doc1,
        doc2,
        *,
        
        use_char_ngrams=False,
        n=4,
        preprocessor=None,
    ) -> float:
        """Calcul de la similarité de Jaccard en utilsant des tokens de mots ou des caractères en n-grams."""
        t1 = doc1
        t2 = doc2

        if preprocessor is not None:
            t1 = preprocessor.preprocess(t1, remove_stopwords=True, lemmatize=False)
            t2 = preprocessor.preprocess(t2, remove_stopwords=True, lemmatize=False)

        if use_char_ngrams:
            s1, s2 = self._char_ngrams(t1 or "", n), self._char_ngrams(t2 or "", n)
        else:
            tokens1 = re.findall(r"\b\w+\b", (t1 or "").lower())
            tokens2 = re.findall(r"\b\w+\b", (t2 or "").lower())
            s1, s2 = set(tokens1), set(tokens2)

        if not s1 and not s2:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)

    def dice_sim(
        self,
        doc1,
        doc2,
        *,
        
        use_char_ngrams=False,
        n=4,
        preprocessor=None,
    ) -> float:
        """Calcul de la similarité de Dice en utilsant des tokens de mots ou des caractères en n-grams."""
        t1 = doc1
        t2 = doc2

        if preprocessor is not None:
            t1 = preprocessor.preprocess(t1, remove_stopwords=True, lemmatize=False)
            t2 = preprocessor.preprocess(t2, remove_stopwords=True, lemmatize=False)

        if use_char_ngrams:
            s1, s2 = self._char_ngrams(t1 or "", n), self._char_ngrams(t2 or "", n)
        else:
            tokens1 = re.findall(r"\b\w+\b", (t1 or "").lower())
            tokens2 = re.findall(r"\b\w+\b", (t2 or "").lower())
            s1, s2 = set(tokens1), set(tokens2)

        if not s1 and not s2:
            return 0.0
        return (2 * len(s1 & s2)) / (len(s1) + len(s2))

    def extract_all_features(self, doc1, doc2):
        """
            Extrait toutes les features pour une paire de documents
            features = { 
                'tfidf-cos' : xxxx,
                'jaccard_words' : xxxx,
                ....
            }
        """
        features = {}
        features["tfidf_cos"] = self.cosine_sim(doc1, doc2, language=self.language)
        features["jaccard_sim"] = self.jaccard_sim(doc1,doc2,use_char_ngrams=True,n=1)
        features["dice_sim"] = self.dice_sim(doc1, doc2)
        return features

def extract_features_from_pairs(pairs_df, doc_dir, preprocessor=None):
    """
        Extrait toutes les features pour toutes les paires de documents
        pairs_df : DataFrame avec les colonnes [doc_orig, doc_susp, label]
        data_dir : Répertoire contenant les documents

        retourne
        (X,y, feature_names) avac X la matrice de features

        {
            'mesure1' : [val_pair1, val_pair2, ....]
            'mesure2' : [val_pair1, val_pair2, ....],
            ...,
            'label' : ['heavy',...]
        }


        FLOW
            Charger le doc original et suspect pour chaque oair
            Pretaitement(opt)
            
            Extraire les features
            Ajouter a X les valeurs de les features
            Ajouter  a y la valueur de la colonne label pour étiqueter

            REturn np.array(X), np.array(y), feature_names    
    """
    feature_extractor = FeatureExtractor()
    X = []
    y = []
    feature_names = None
    
    # print(f"Extraction des features pour {len(pairs_df)} paires...")
    
    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
        
        # Charger les documents
        doc1_path = os.path.join(doc_dir, row['doc_source'])
        doc2_path = os.path.join(doc_dir, row['doc_suspect'])
        
        doc1 = load_document(doc1_path)
        doc2 = load_document(doc2_path)
        
        # Prétraitement optionnel
        if preprocessor:
            doc1 = preprocessor.preprocess(doc1)
            doc2 = preprocessor.preprocess(doc2)
        
        # Extraction features
        features = feature_extractor.extract_all_features(doc1, doc2)
        
        if feature_names is None:
            feature_names = list(features.keys())
        
        X.append(list(features.values()))
        y.append(row['label'])
    
    return np.array(X), np.array(y), feature_names
       

if __name__ == "__main__":
    base_dir = Path(__file__).resolve()
    # print(base_dir)
    
    # extractor = FeatureExtractor()
    # doc1 = data_dir / "g0pC_taskd.txt"
    # doc2 = data_dir / "orig_taskd.txt"
    # pairs_df = data_dir / "labels.csv"
    # df = pd.read_csv(pairs_df)

    # features = extractor.extract_all_features(str(doc1), str(doc2))
    # print("Features extraites : ")
    # for name, value in features.items():
    #     print(f"{name} : {value:.4f}")

    src_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, src_dir)
    root_dir    = os.path.dirname(src_dir)
    LABELS_PATH = os.path.join(root_dir, 'data', 'labels.csv')
    DATA_DIR    = os.path.join(root_dir, 'data-plagiarism')

    pairs_df = pd.read_csv(LABELS_PATH)
    

    X, y, feature_names = extract_features_from_pairs(pairs_df,DATA_DIR,preprocessor=None)
    
    # Résultat
    print(f"\nExtraction réussie !")
    print(f"  Label: {y[0]}")
    print(f"\n  Features:")
    for name, value in zip(feature_names, X[0]):
        print(f"    {name:25s}: {value:.4f}")