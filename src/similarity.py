"""Similarité cosine basée sur un sac de mots (CountVectorizer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from bow import BowResult, texts_to_bow


@dataclass
class BowCosineSimilarity:
    """Calcule une matrice de similarité cosine sur un corpus BoW."""

    stop_words: Optional[Union[str, List[str]]] = "english"
    max_df: Optional[Union[int, float]] = 1.0
    min_df: Optional[Union[int, float]] = 1
    max_features: Optional[int] = None
    lowercase: bool = True

    bow: Optional[BowResult] = None
    sim_matrix: Optional[pd.DataFrame] = None

    def fit(self, texts: Union[pd.Series, Sequence[str]]) -> pd.DataFrame:
        """Vectorise les textes puis calcule la matrice de similarité cosine."""
        self.bow = texts_to_bow(
            texts,
            stop_words=self.stop_words,
            max_df=self.max_df,
            min_df=self.min_df,
            max_features=self.max_features,
            lowercase=self.lowercase,
        )

        matrix = self.bow.matrix  # scipy.sparse matrix
        sims = cosine_similarity(matrix)
        self.sim_matrix = pd.DataFrame(sims)
        return self.sim_matrix

    def similarity_df(
        self, labels: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """Retourne la matrice de similarité avec index/colonnes optionnels."""
        if self.sim_matrix is None:
            raise RuntimeError("Appelle fit() avant d'obtenir la matrice.")

        if labels is None:
            return self.sim_matrix

        if len(labels) != len(self.sim_matrix):
            raise ValueError("labels doit avoir la même longueur que le corpus.")

        df = self.sim_matrix.copy()
        df.index = labels
        df.columns = labels
        return df

    def pair_similarity(self, i: int, j: int) -> float:
        """Score cosine entre deux textes (indices i, j)."""
        if self.sim_matrix is None:
            raise RuntimeError("Appelle fit() avant d'obtenir un score.")
        return float(self.sim_matrix.iat[i, j])
