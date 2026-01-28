#Entrainer et gerer les modèles de classification
class PlagiarismClassifier:
    def __init__(self, classifier_type=''):
        self.classifier_type = classifier_type
        self.model = self._init_classifier(classifier_type)
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False

    def _init_classifier(self, classifier_type):
        classifiers = {
            '...': Classifier(...)
        }
        return classifiers.get(classifier_type)

    def train(self, X, y, feature_names=None):
        """
            Encoder les labels
            Sauvegarder les noms des features 
            Entrainement 
        """
        pass

    def predict(self, X):
        """
        Predit les labels
        X matrice de features
        return 
            np.array : Labels prédits
        """

    def predict_proba(self, X):
        """
        Prédit les probabilités
        X matrice de features
        return
            np.array : Probabilités pour chaque classe
        """

    def get_feature_importance(self, top_n=10):
        """
        Retourne l'importance des features (RandomForest et GB)
        top_n : Nombre de features à retouner
        Return : liste de tuple (feature_name, importance )
        """

    def save(self, model_path, encoder_path):
        pass

    def load(self, model_path, encoder_path):
        pass

def train_and_compare_models(X,y,feature_names=None):
    """
    Entraine et compares plusieurs models 

    X: matrice de features
    y : labels 
    feature_names: Noms des features

    REturn : 
    dict : Resultats pour chaque modèle
    """