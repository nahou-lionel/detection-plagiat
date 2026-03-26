# Evaluer les performances et créer des visualisations
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations, chain

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder

from models import PlagiarismClassifier


# ─────────────────────────────────────────────
# 1. Évaluation d'un classificateur
# ─────────────────────────────────────────────

def evaluate_classifier(y_true, y_pred, classes=None):
    """
    Évalue les performances d'un classificateur.

    Paramètres :
        y_true  : vrais labels
        y_pred  : labels prédits
        classes : liste ordonnée des classes (optionnel)

    Retourne :
        dict avec accuracy, précision, rappel, F1 (macro & weighted),
        rapport de classification complet, matrice de confusion.
    """
    if classes is None:
        classes = sorted(list(set(y_true) | set(y_pred)))

    return {
        'accuracy':            accuracy_score(y_true, y_pred),
        'precision_macro':     precision_score(y_true, y_pred, average='macro',    zero_division=0),
        'recall_macro':        recall_score   (y_true, y_pred, average='macro',    zero_division=0),
        'f1_macro':            f1_score       (y_true, y_pred, average='macro',    zero_division=0),
        'precision_weighted':  precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted':     recall_score   (y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted':         f1_score       (y_true, y_pred, average='weighted', zero_division=0),
        'report':              classification_report(y_true, y_pred, labels=classes, zero_division=0),
        'confusion_matrix':    confusion_matrix(y_true, y_pred, labels=classes),
        'classes':             classes,
    }


def print_evaluation(metrics, classifier_name="Classificateur"):
    """Affiche les métriques d'évaluation de manière lisible."""
    print(f"\n{'='*55}")
    print(f"  Évaluation : {classifier_name}")
    print(f"{'='*55}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Précision (macro) : {metrics['precision_macro']:.4f}")
    print(f"  Rappel    (macro) : {metrics['recall_macro']:.4f}")
    print(f"  F1-score  (macro) : {metrics['f1_macro']:.4f}")
    print(f"\nRapport de classification :\n")
    print(metrics['report'])


# ─────────────────────────────────────────────
# 2. Visualisation – Matrice de confusion
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, classes, title="Matrice de Confusion", save_path=None):
    """
    Calcule et affiche la matrice de confusion.

    Retourne :
        matplotlib.figure.Figure
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=12)

    ax.set_ylabel('Vrai label')
    ax.set_xlabel('Label prédit')
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Matrice de confusion sauvegardée : {save_path}")

    return fig


# ─────────────────────────────────────────────
# 3. Comparaison de plusieurs modèles
# ─────────────────────────────────────────────

def compare_models(X, y, feature_names=None, test_size=0.2, random_state=42):
    """
    Entraîne et compare les cinq classificateurs disponibles.

    Retourne :
        dict  {classifier_type -> {classifier, metrics, y_pred, y_test}}
    """
    classifier_types = [
        'random_forest', 'gradient_boosting',
        'naive_bayes', 'svm', 'logistic_regression'
    ]
    classes = sorted(list(set(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"\n{'='*60}")
    print(f"  Comparaison de {len(classifier_types)} classificateurs")
    print(f"  Train : {len(X_train)} | Test : {len(X_test)} | Classes : {classes}")
    print(f"{'='*60}")

    results = {}
    for clf_type in classifier_types:
        clf = PlagiarismClassifier(clf_type)
        clf.train(X_train, y_train, feature_names=feature_names)

        y_pred = clf.predict(X_test)
        metrics = evaluate_classifier(y_test, y_pred, classes=classes)

        results[clf_type] = {
            'classifier': clf,
            'metrics':    metrics,
            'y_pred':     y_pred,
            'y_test':     y_test,
        }
        print_evaluation(metrics, clf_type)

    return results, X_test


def plot_model_comparison(results, save_path=None):
    """
    Graphique en barres groupées comparant accuracy, précision, rappel, F1
    pour chaque modèle.

    Retourne :
        matplotlib.figure.Figure
    """
    short_names = {
        'random_forest':      'RF',
        'gradient_boosting':  'GB',
        'naive_bayes':        'NB',
        'svm':                'SVM',
        'logistic_regression':'LR',
    }
    model_names   = list(results.keys())
    metric_keys   = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    metric_labels = ['Accuracy', 'Précision', 'Rappel', 'F1-score']

    x = np.arange(len(model_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        values = [results[m]['metrics'][key] for m in model_names]
        bars = ax.bar(x + i * width, values, width, label=label)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([short_names.get(m, m) for m in model_names], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Comparaison des classificateurs')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Graphique comparaison sauvegardé : {save_path}")

    return fig


# ─────────────────────────────────────────────
# 4. Ablation study (question scientifique)
# ─────────────────────────────────────────────

def ablation_study(X, y, feature_names, test_size=0.2, random_state=42, classifier_type='random_forest'):
    """
    Teste chaque mesure de similarité seule puis en combinaisons
    afin de répondre à :
        « Quelles mesures de similarité sont les plus performantes pour détecter le plagiat ? »

    Retourne :
        dict  {label -> {f1_macro, accuracy, features}}
        trié par F1 décroissant.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"\n{'='*60}")
    print(f"  Ablation Study – contribution de chaque mesure de similarité")
    print(f"  Classificateur : {classifier_type}")
    print(f"{'='*60}")

    results = {}
    n = X.shape[1]

    def _eval(indices, label):
        Xtr = X_train[:, indices]
        Xte = X_test[:, indices]
        clf = PlagiarismClassifier(classifier_type)
        clf.train(Xtr, y_train)
        y_pred = clf.predict(Xte)
        f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        results[label] = {
            'f1_macro':  f1,
            'accuracy':  acc,
            'features':  [feature_names[i] for i in indices],
        }
        print(f"  [{label:<35s}]  F1={f1:.4f}  Acc={acc:.4f}")

    # Features individuelles
    for i, name in enumerate(feature_names):
        _eval([i], name)

    # Combinaisons de 2 features
    for combo in combinations(range(n), 2):
        label = " + ".join(feature_names[i] for i in combo)
        _eval(list(combo), label)

    # Toutes les features
    all_label = " + ".join(feature_names)
    _eval(list(range(n)), all_label)

    # Trier par F1 décroissant
    return dict(sorted(results.items(), key=lambda kv: kv[1]['f1_macro'], reverse=True))


def plot_ablation_study(ablation_results, save_path=None):
    """
    Graphique horizontal (barh) des F1-scores de l'ablation study.

    Retourne :
        matplotlib.figure.Figure
    """
    labels    = list(ablation_results.keys())
    f1_scores = [ablation_results[k]['f1_macro'] for k in labels]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.55)))
    bars = ax.barh(labels, f1_scores, color='steelblue')

    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)

    ax.set_xlim(0, 1.12)
    ax.set_xlabel('F1-score (macro)')
    ax.set_title('Ablation Study : contribution des mesures de similarité')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Ablation study sauvegardée : {save_path}")

    return fig


# ─────────────────────────────────────────────
# 5. Validation croisée
# ─────────────────────────────────────────────

def cross_validate_model(X, y, classifier_type='random_forest', n_splits=5):
    """
    Effectue une validation croisée stratifiée sur le modèle choisi.

    Retourne :
        dict {metric -> {mean, std, scores}}
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    clf = PlagiarismClassifier(classifier_type)
    model = clf.model

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scorers = {
        'accuracy':        'accuracy',
        'f1_macro':        make_scorer(f1_score,        average='macro', zero_division=0),
        'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
        'recall_macro':    make_scorer(recall_score,    average='macro', zero_division=0),
    }

    print(f"\n{'='*60}")
    print(f"  Validation croisée – {classifier_type} ({n_splits} folds)")
    print(f"{'='*60}")

    cv_results = {}
    for metric, scorer in scorers.items():
        scores = cross_val_score(model, X, y_encoded, cv=cv, scoring=scorer)
        mean, std = scores.mean(), scores.std()
        cv_results[metric] = {'mean': mean, 'std': std, 'scores': scores.tolist()}
        print(f"  {metric:<20s}: {mean:.4f} ± {std:.4f}")

    return cv_results


# ─────────────────────────────────────────────
# 6. Grille complète features × modèles
# ─────────────────────────────────────────────

def full_grid_search(X, y, feature_names, test_size=0.2, random_state=42):
    """
    Teste toutes les combinaisons possibles de features × modèles.
    Avec 7 features → 127 sous-ensembles × 5 modèles = 635 entraînements.

    Retourne :
        (best_clf_type, best_features, results_df)
    """
    classifier_types = [
        'random_forest', 'gradient_boosting',
        'naive_bayes', 'svm', 'logistic_regression'
    ]

    n = X.shape[1]
    all_subsets = list(chain.from_iterable(
        combinations(range(n), r) for r in range(1, n + 1)
    ))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    total = len(all_subsets) * len(classifier_types)
    print(f"\n{'='*60}")
    print(f"  Grid Search : {len(all_subsets)} sous-ensembles × {len(classifier_types)} modèles = {total} entraînements")
    print(f"{'='*60}")

    records = []
    count = 0
    for subset in all_subsets:
        feat_names = [feature_names[i] for i in subset]
        feat_label = " + ".join(feat_names)
        Xtr = X_train[:, list(subset)]
        Xte = X_test[:, list(subset)]

        for clf_type in classifier_types:
            clf = PlagiarismClassifier(clf_type)
            clf.train(Xtr, y_train)
            y_pred = clf.predict(Xte)
            f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            records.append({
                'model':    clf_type,
                'features': feat_label,
                'n_feats':  len(subset),
                'f1_macro': f1,
                'accuracy': acc,
            })
            count += 1
            if count % 50 == 0:
                print(f"  ... {count}/{total} entraînements effectués")

    results_df = pd.DataFrame(records).sort_values('f1_macro', ascending=False).reset_index(drop=True)

    best = results_df.iloc[0]
    print(f"\n  ✓ Meilleure combinaison trouvée :")
    print(f"    Modèle   : {best['model']}")
    print(f"    Features : {best['features']}")
    print(f"    F1-score : {best['f1_macro']:.4f}")
    print(f"    Accuracy : {best['accuracy']:.4f}")

    return best['model'], best['features'], results_df


def plot_grid_search(results_df, top_n=15, save_path=None):
    """
    Graphique barh des top_n meilleures combinaisons modèle × features.

    Retourne :
        matplotlib.figure.Figure
    """
    short_names = {
        'random_forest':      'RF',
        'gradient_boosting':  'GB',
        'naive_bayes':        'NB',
        'svm':                'SVM',
        'logistic_regression':'LR',
    }

    top_df = results_df.head(top_n).copy()
    # Trier croissant pour barh (meilleur en haut)
    top_df = top_df.sort_values('f1_macro', ascending=True)

    labels = [
        f"{short_names.get(row['model'], row['model'])} | {row['features']}"
        for _, row in top_df.iterrows()
    ]
    f1_values = top_df['f1_macro'].tolist()

    fig, ax = plt.subplots(figsize=(12, max(5, top_n * 0.5)))
    bars = ax.barh(labels, f1_values, color='steelblue')

    for bar, val in zip(bars, f1_values):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    ax.set_xlim(0, 1.12)
    ax.set_xlabel('F1-score (macro)')
    ax.set_title(f'Grid Search – Top {top_n} combinaisons modèle × features')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Grid search sauvegardée : {save_path}")

    return fig


# ─────────────────────────────────────────────
# 7. Analyse SHAP
# ─────────────────────────────────────────────

def shap_analysis(classifier, X_test, feature_names, class_names, save_path=None):
    """
    Calcule et affiche les valeurs SHAP pour expliquer les prédictions du modèle.

    Paramètres :
        classifier    : PlagiarismClassifier entraîné
        X_test        : matrice de features du jeu de test
        feature_names : liste des noms de features
        class_names   : liste des classes
        save_path     : chemin optionnel pour sauvegarder le graphique

    Retourne :
        shap_values
    """
    import shap

    # TreeExplainer : RandomForest uniquement 
    if classifier.classifier_type == 'random_forest':
        explainer   = shap.TreeExplainer(classifier.model)
        shap_values = explainer.shap_values(X_test)
    else:
        explainer   = shap.KernelExplainer(classifier.model.predict_proba, X_test)
        shap_values = explainer.shap_values(X_test)

    # Summary plot : importance moyenne par feature (toutes classes confondues)
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        class_names=class_names,
        plot_type="bar",
        show=False,
    )
    plt.title("SHAP – Importance des features par classe")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  SHAP summary sauvegardé : {save_path}")

    plt.close()
    return shap_values


# ─────────────────────────────────────────────
# Point d'entrée – pipeline d'évaluation complet
# ─────────────────────────────────────────────

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, src_dir)
    from feature_extraction import extract_features_from_pairs

    root_dir    = os.path.dirname(src_dir)
    labels_path = os.path.join(root_dir, 'data', 'labels.csv')
    docs_dir    = os.path.join(root_dir, 'data-plagiarism')
    output_dir  = os.path.join(root_dir, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    # Chargement
    pairs_df = pd.read_csv(labels_path)
    print(f"Paires chargées : {len(pairs_df)}")

    # Extraction des features
    X, y, feature_names = extract_features_from_pairs(pairs_df, docs_dir)
    print(f"Shape : {X.shape} | Classes : {sorted(set(y))}")

    # 1. Comparaison des modèles (toutes les features)
    results, X_test = compare_models(X, y, feature_names=feature_names)
    plot_model_comparison(results, save_path=os.path.join(output_dir, 'model_comparison.png'))

    # 2. Identifier le meilleur modèle
    best_model = max(results, key=lambda m: results[m]['metrics']['f1_macro'])
    best_f1    = results[best_model]['metrics']['f1_macro']
    print(f"\n  Meilleur modèle : {best_model} (F1={best_f1:.4f})")

    # 3. Grille complète features × modèles
    best_model, best_feats, grid_df = full_grid_search(X, y, feature_names)
    plot_grid_search(grid_df, save_path=os.path.join(output_dir, 'grid_search.png'))
    grid_df.to_csv(os.path.join(output_dir, 'grid_search.csv'), index=False)

    # 4. Ablation study avec le meilleur modèle
    print(f"\n  => Ablation study lancée avec : {best_model}")
    ablation_results = ablation_study(X, y, feature_names, classifier_type=best_model)
    plot_ablation_study(ablation_results, save_path=os.path.join(output_dir, 'ablation_study.png'))

    # 5. Matrice de confusion
    plot_confusion_matrix(
        results[best_model]['y_test'],
        results[best_model]['y_pred'],
        classes=sorted(set(y)),
        title=f"Matrice de Confusion – {best_model}",
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )

    # 6. Validation croisée
    cross_validate_model(X, y, classifier_type=best_model)

    # 7. Analyse SHAP
    shap_analysis(
        results[best_model]['classifier'],
        X_test,
        feature_names=feature_names,
        class_names=sorted(set(y)),
        save_path=os.path.join(output_dir, 'shap_summary.png')
    )

    print("\nÉvaluation terminée. Résultats dans", output_dir)