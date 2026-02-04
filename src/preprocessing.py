#Nettoyer et normaliser les textes
import re
import unicodedata
import spacy
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


class TextPreprocessor:
    def __init__(self, language='english', use_spacy=True):
        self.language = language
        self.use_spacy = use_spacy
        # Stopwords
        self.stopwords = set(stopwords.words(language))
        # Chargement de spaCy si demandé
        if self.use_spacy:
            if language == 'english':
                self.nlp = spacy.load("en_core_web_sm")
            elif language == 'french':
                self.nlp = spacy.load("fr_core_news_sm")
            else:
                raise ValueError("Langue non supportée par spaCy")
        else:
            self.nlp = None

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
        #Là je fais juste la tokenisation simple par espace
        return text.split()

    def remove_stopwords(self, text):
        
        if isinstance(tokens, str):
            tokens = tokens.split()
        return [token for token in tokens if token not in self.stopwords]

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
        return tokens
        
    @staticmethod
    def load_document(filepath, encoding='utf-8'):
        """
            Charger un document texte 
        """
        with open(filepath, "r", encoding=encoding) as f:
            return f.read()

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
