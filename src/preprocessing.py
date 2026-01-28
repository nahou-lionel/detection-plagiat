#Nettoyer et normaliser les textes

class TextPreprocessor:
    def __init__(self, language='english',use_spacy=true):
        pass
        """
            Selon la langue et si spacy est utilisé définir l'attribut nlp avec spacy.load
        """
    def clean_text(self, text):
        """
            Normalisation unicode, supprimer urls, mails, chiffres(opt), caractères spéciaux, espaces multiples
        """
        pass
    def tokenize(self, text):
        pass
    def remove_stopwords(self, text):
        pass
    def lemmatize(self, text):
        """
            Si spacy n'est pas utilisé retourner le texte tel quel sinon le lemmatiser
        """
        pass
    def preprocess(self, text, remove_stopwords=True, lemmatize=False):
        """
        Process complet de pretaitement
        Nettoyage -> Lemmatisation -> Tokenisation -> Suppression des stop words
        """
        pass
def load_document(filepath, encoding='utf-8'):
    """
        Charger un document texte 
    """
    pass

if __name__ == "__main__":
    #Test 
    preprocessor = TextPreprocessor()

    test_text= """
    """

    print("Texte original : ")
    print(test_text)
    print("\n Texte nettoyé :")
    print(preprocessor.clean_text(test_text))
    print("\n Texte prétraité (avec lemmatisation) :")
    print(preprocessor.preprocess(test_text, remove_stopwords=True, lemmatize=True))
