#Extraire les mesures de similarités entre paires de documents
class FeatureExtractor:
    def __init__(self, language='english'):
        pass
        

    def extract_all_features(self,doc1, doc2):
        """
            Extrait toutes les features pour une paire de documents
            features = { 
                'tfidf-cos' : xxxx,
                'jaccard_words' : xxxx,
                ....
            }
        """
        features={}
        #features.update()

        return features

    def extract_features_from_pairs(pairs_df, doc_dir, preprocessor=True):
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


        pass

if __name__== "___main___":
    extractor = FeatureExtractor(language='french')
    doc1 = ""
    doc2 = ""

    features = extractor.extract_all_features(doc1, doc2)

    print("Features extraites : ")
    for name, value in features.items():
        print(f"{name} : {value:.4f}")