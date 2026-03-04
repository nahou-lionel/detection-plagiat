#Entrainer et gerer les modèles de classification
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib
import json

class PlagiarismClassifier:
    def __init__(self, classifier_type='random_forest'):
        self.classifier_type = classifier_type
        self.model = self._init_classifier(classifier_type)
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False

    def _init_classifier(self, classifier_type):
        """Initialise le classificateur"""
        classifiers = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'naive_bayes': GaussianNB(),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        }
        
        return classifiers.get(classifier_type, classifiers['random_forest'])

    def train(self, X, y, feature_names=None):
        """
            Encoder les labels
            Sauvegarder les noms des features 
            Entrainement 
        """

        #Encoder les labels 
        y_encoded = self.label_encoder.fit_transform(y)
        self.feature_names = feature_names

        print(f"Entrainement du modele {self.classifier_type}...")
        self.model.fit(X,y_encoded)
        self.is_trained = True

    def predict(self, X):
        """
        Predit les labels
        X matrice de features
        return 
            np.array : Labels prédits
        """

        if not self.is_trained :
            raise ValueError("Le modèle n'a pas été entraîné")


        y_pred_encoded = self.model.predict(X)

        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X):
        """
        Prédit les probabilités
        X matrice de features
        return
            np.array : Probabilités pour chaque classe
        """
        if not self.is_trained :
            raise ValueError("Le modèle n'a pas été entraîné")



        return self.model.predict_proba(X)


    def get_feature_importance(self, top_n=10):
        """
        Retourne l'importance des features (RandomForest et GB)
        top_n : Nombre de features à retouner
        Return : liste de tuple (feature_name, importance )
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("Ce modèle ne supporte pas l'importance des features")
            return None 

        importances = self.model.feature_importances_
        if self.feature_names : 
            features_with_importance = list(zip(self.feature_names, importances))
        else:
            features_with_importance = list(zip(range(len(importances)), importances))

        # Trierrrrrr

        features_with_importance.sort(key=lambda x : x[1], reverse=True)
        return features_with_importance[:top_n]

    def save(self, model_path, encoder_path):
        """Sauvegarde le modèle et l'encodeur"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        # Sauvegarder les métadonnées
        metadata = {
            'classifier_type': self.classifier_type,
            'feature_names': self.feature_names,
            'classes': self.label_encoder.classes_.tolist()
        }
        
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Modèle sauvegardé dans {model_path}")
    
    def load(self, model_path, encoder_path):
        """Charge un modèle sauvegardé"""
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)
        
        # Charger les métadonnées
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.classifier_type = metadata['classifier_type']
                self.feature_names = metadata['feature_names']
        except:
            print("Métadonnées non trouvées, chargeant sans...")
        
        self.is_trained = True
        print(f"Modèle chargé depuis {model_path}")

def train_and_compare_models(X,y,feature_names=None):
    """
    Entraine et compares plusieurs models 

    X: matrice de features
    y : labels 
    feature_names: Noms des features

    REturn : 
    dict : Resultats pour chaque modèle
    """

if __name__ == "__main__":
    # Test
    from sklearn.datasets import make_classification
    
    # Données fictives
    X, y = make_classification(n_samples=100, n_features=10, n_classes=4, random_state=42, n_clusters_per_class=1)
    y = np.array(['non', 'light', 'cut', 'heavy'])[y]
    
    clf = PlagiarismClassifier('random_forest')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf.train(X_train, y_train)
    predictions = clf.predict(X_test)
    
    print("Prédictions:", predictions)