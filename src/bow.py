# bow.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Union, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


@dataclass
class BowResult:
    """Résultat complet d'une vectorisation Bag-of-Words."""
    vectorizer: CountVectorizer
    matrix  : "scipy.sparse.spmatrix"  # type: ignore[name-defined]
    df: pd.DataFrame


def texts_to_bow(
    texts: Union[pd.Series, List[str], Tuple[str, ...]],
    *,
    stop_words: Optional[Union[str, List[str]]] = "english",
    max_df: Optional[float] = 1.0,
    min_df: Optional[Union[int, float]] = 1,
    max_features: Optional[int] = None,
    lowercase: bool = True,
) -> BowResult:
    """
    Convertit une collection de textes en Bag-of-Words.

    - texts: liste/tuple/Series de strings
    - stop_words: "english" ou liste de stop words, ou None
    - max_df / min_df: filtrage fréquentiel (comme sklearn)
    - max_features: limite du vocabulaire
    """
    if isinstance(texts, pd.Series):
        texts_list = texts.astype(str).tolist()
    else:
        texts_list = [str(t) for t in texts]

    if len(texts_list) == 0:
        raise ValueError("La liste de textes est vide.")

    if max_df is None:
        max_df = 1.0
    if min_df is None:
        min_df = 1

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        max_df=max_df,
        min_df=min_df,
        max_features=max_features,
        lowercase=lowercase,
    )

    matrix = vectorizer.fit_transform(texts_list)
    vocab = vectorizer.get_feature_names_out()
    df = pd.DataFrame(matrix.toarray(), columns=vocab)

    return BowResult(vectorizer=vectorizer, matrix=matrix, df=df)


def display_bow(
    bow: BowResult,
    *,
    head: Optional[int] = None,
    show_vocab: bool = True,
) -> None:
    """Affiche le vocabulaire et la matrice BoW (via DataFrame)."""
    if show_vocab:
        print(f"Vocabulary ({len(bow.vectorizer.get_feature_names_out())} mots) :")
        print(bow.vectorizer.get_feature_names_out())

    print("BoW :")
    if head is None:
        print(bow.df)
    else:
        print(bow.df.head(head))
