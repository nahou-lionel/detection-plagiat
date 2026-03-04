#Nettoyer et normaliser les textes
import re
import unicodedata
import spacy
import nltk
from nltk.corpus import stopwords
import os
# Télécharger les ressources NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


class TextPreprocessor:
    def __init__(self, language='english', use_spacy=True):
        self.language = language
        self.use_spacy = use_spacy
        
        # Stop words
        if language == 'french':
            self.stop_words = set(stopwords.words('french'))
            if use_spacy:
                self.nlp = spacy.load('fr_core_news_sm')
        else:
            self.stop_words = set(stopwords.words('english'))
            if use_spacy:
                self.nlp = spacy.load('en_core_web_sm')

    def clean_text(self, text):
        """
            Normalisation unicode, supprimer urls, mails, chiffres(opt), caractères spéciaux, espaces multiples
        """
        # Normalisation unicode
        text = unicodedata.normalize("NFKD", text)
        # Minuscules
        text = text.lower()
        # Supprimer URLs
        text = re.sub(r"http\S+|www\S+", "", text)
        # Supprimer emails
        text = re.sub(r"\S+@\S+", "", text)
        # Supprimer chiffres
        text = re.sub(r"\d+", "", text)
        # Supprimer caractères spéciaux
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        # Supprimer espaces multiples
        text = re.sub(r"\s+", " ", text).strip()
        return text


    def tokenize(self, text):
        return nltk.word_tokenize(text.lower(), language=self.language)

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize(self, text):
        """
            Si spacy n'est pas utilisé retourner le texte tel quel sinon le lemmatiser
        """
        if not self.use_spacy:
            return text
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_punct])
        
    def preprocess(self, text, remove_stopwords=True, lemmatize=False):
        """
        Process complet de pretaitement
        Nettoyage -> Lemmatisation -> Tokenisation -> Suppression des stop words
        """
        text = self.clean_text(text)
        if lemmatize:
            text = self.lemmatize(text)
        tokens = self.tokenize(text)
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        return ' '.join(tokens)
        
def load_document(input_or_path, encoding='utf-8'):
    # Accept either a file path or raw text. If it's a valid path, read the file; otherwise, return the string.
    if isinstance(input_or_path, str) and os.path.exists(input_or_path):
        try:
            with open(input_or_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            with open(input_or_path, 'r', encoding='latin-1') as f:
                return f.read()
    elif isinstance(input_or_path, str):
        # Treat as raw text
        return input_or_path
    else:
        raise TypeError("load_document expects a string path or raw text.")

if __name__ == "__main__":
    # Test
    preprocessor = TextPreprocessor(language='english', use_spacy=True)
    test_text = """
    This is a simple example! Contact us at example@email.com
    or visit https://example.com.
    """
    print("Texte original :")
    print(test_text)
    print("\nTexte nettoyé :")
    print(preprocessor.clean_text(test_text))
    print("\nTexte prétraité (avec lemmatisation) :")
    print(preprocessor.preprocess(test_text, remove_stopwords=True, lemmatize=True))
    print("\nTexte prétraité (sans lemmatisation) :")
    print(preprocessor.preprocess(test_text, remove_stopwords=True, lemmatize=False))
