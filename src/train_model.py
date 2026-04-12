"""
=============================================================
 Etape 5 - Modelisation
 Fichier : src/train_model.py
 Projet  : Analyse Comportementale Clientele Retail
=============================================================
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# ── Chemins ───────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_TEST_DIR = os.path.join(BASE_DIR, 'data', 'train_test')
PROCESSED      = os.path.join(BASE_DIR, 'data', 'processed', 'data_clean.csv')
MODELS_DIR     = os.path.join(BASE_DIR, 'models')
REPORTS_DIR    = os.path.join(BASE_DIR, 'reports')
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# CHARGEMENT DES DONNEES
# ─────────────────────────────────────────────

def load_train_test():
    """Charge les fichiers X_train, X_test, y_train, y_test."""
    X_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, 'X_train.csv'))
    X_test  = pd.read_csv(os.path.join(TRAIN_TEST_DIR, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, 'y_train.csv')).squeeze()
    y_test  = pd.read_csv(os.path.join(TRAIN_TEST_DIR, 'y_test.csv')).squeeze()
    print(f"[OK] X_train : {X_train.shape} | X_test : {X_test.shape}")
    print(f"[OK] Taux Churn train : {y_train.mean():.3f} | test : {y_test.mean():.3f}\n")
    return X_train, X_test, y_train, y_test


# ═════════════════════════════════════════════
# A) CLUSTERING (non supervise)
# ═════════════════════════════════════════════

def run_clustering(X_train, n_clusters=4):
    """
    Segmente les clients avec K-Means.

    PRINCIPE : K-Means regroupe les clients en k groupes
    de facon a minimiser la distance de chaque client
    au centre (centroide) de son groupe.

    METHODE DU COUDE :
    On teste k de 2 a 10 et on trace l'inertie.
    L'inertie diminue toujours quand k augmente.
    Le "coude" = le k ou le gain devient faible.

    SILHOUETTE SCORE :
    Mesure la qualite des clusters.
    Proche de 1 = clusters bien separes.
    Proche de 0 = clusters qui se chevauchent.
    """
    print("=" * 55)
    print("  A) CLUSTERING K-MEANS")
    print("=" * 55 + "\n")

    # ── Normalisation pour le clustering ───────────────────────
    # KMeans est sensible aux échelles. On utilise le scaler global.
    try:
        scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            # S'assurer que les colonnes matchent
            cols_scaler = list(scaler.feature_names_in_)
            X_train_sc = X_train[cols_scaler]
            X_train_sc = scaler.transform(X_train_sc)
            print("[OK] Données normalisées pour le clustering")
        else:
            print(f"[WARN] Fichier {scaler_path} non trouvé")
            X_train_sc = X_train
    except Exception as e:
        print(f"[WARN] Erreur lors de la normalisation : {e}")
        X_train_sc = X_train

    # ── Methode du coude ──────────────────────────────────────
    print("[...] Methode du coude (k = 2 a 10)...")
    inertias    = []
    sil_scores  = []
    k_range     = range(2, 11)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_train_sc)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_train_sc, km.labels_))

    # Affichage
    print(f"\n  {'k':<5} {'Inertie':>12}  {'Silhouette':>12}")
    print("  " + "-" * 35)
    for k, iner, sil in zip(k_range, inertias, sil_scores):
        marker = " <-- optimal" if k == n_clusters else ""
        print(f"  {k:<5} {iner:>12.1f}  {sil:>12.4f}{marker}")

    # ── Graphiques ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(x=n_clusters, color='red', linestyle='--',
                    label=f'k={n_clusters} choisi')
    axes[0].set_xlabel('Nombre de clusters (k)')
    axes[0].set_ylabel('Inertie')
    axes[0].set_title('Methode du Coude', fontweight='bold')
    axes[0].legend()

    axes[1].plot(k_range, sil_scores, 'gs-', linewidth=2, markersize=8)
    axes[1].axvline(x=n_clusters, color='red', linestyle='--',
                    label=f'k={n_clusters} choisi')
    axes[1].set_xlabel('Nombre de clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score par k', fontweight='bold')
    axes[1].legend()

    plt.suptitle('Choix du nombre optimal de clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'clustering_elbow.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # ── KMeans final ──────────────────────────────────────────
    print(f"\n[...] KMeans final avec k={n_clusters}...")
    kmeans   = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_train_sc)

    sil_final = silhouette_score(X_train_sc, clusters)
    print(f"  Silhouette Score final : {sil_final:.4f}")

    # ── Interpretation des clusters ───────────────────────────
    df_cluster = X_train.copy()
    df_cluster['Cluster'] = clusters

    print(f"\n  Taille de chaque cluster :")
    for i in range(n_clusters):
        n = (clusters == i).sum()
        pct = n / len(clusters) * 100
        print(f"    Cluster {i} : {n} clients ({pct:.1f}%)")

    # Profil moyen par cluster (top features)
    key_features = [c for c in ['Recency', 'Frequency', 'MonetaryTotal',
                                  'Age', 'SatisfactionScore'] if c in X_train.columns]
    if key_features:
        print(f"\n  Profil moyen par cluster (features cles) :")
        profile = df_cluster.groupby('Cluster')[key_features].mean().round(3)
        print(profile.to_string())

    # Sauvegarde du modele
    joblib.dump(kmeans, os.path.join(MODELS_DIR, 'kmeans.joblib'))
    print(f"\n[OK] KMeans sauvegarde : models/kmeans.joblib\n")

    return kmeans, clusters


# ═════════════════════════════════════════════
# B) CLASSIFICATION — Predire Churn
# ═════════════════════════════════════════════

def evaluate_classifier(name, model, X_test, y_test):
    """Evalue un modele de classification et affiche les metriques."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n  --- {name} ---")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}" if auc else "  AUC-ROC   : N/A")
    print(f"  Precision : {report['1']['precision']:.4f}")
    print(f"  Recall    : {report['1']['recall']:.4f}")
    print(f"  F1-Score  : {report['1']['f1-score']:.4f}")

    return {
        'name': name, 'model': model,
        'accuracy': acc, 'auc': auc,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score'],
        'y_pred': y_pred, 'y_proba': y_proba
    }


def plot_classification_results(results, X_test, y_test):
    """Trace les graphiques de comparaison des modeles de classification."""

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    colors = ['#3498db', '#2ecc71', '#e74c3c']

    # ── Graphique 1 : Comparaison des metriques ───────────────
    metrics = ['accuracy', 'auc', 'precision', 'recall', 'f1']
    labels  = ['Accuracy', 'AUC-ROC', 'Precision', 'Recall', 'F1']
    x       = np.arange(len(metrics))
    width   = 0.25

    for i, res in enumerate(results):
        vals = [res[m] if res[m] is not None else 0 for m in metrics]
        axes[0].bar(x + i*width, vals, width, label=res['name'],
                    color=colors[i], alpha=0.85, edgecolor='white')

    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title('Comparaison des metriques', fontweight='bold')
    axes[0].legend()
    axes[0].axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Seuil 0.8')

    # ── Graphiques 2-4 : Matrices de confusion ─────────────────
    for i, res in enumerate(results):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Fidele', 'Parti'],
                    yticklabels=['Fidele', 'Parti'],
                    ax=axes[i+1])
        axes[i+1].set_title(f'Matrice confusion — {res["name"]}', fontweight='bold')
        axes[i+1].set_ylabel('Reel')
        axes[i+1].set_xlabel('Predit')

    # ── Graphique 5 : Courbes ROC ─────────────────────────────
    for i, res in enumerate(results):
        if res['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
            axes[4].plot(fpr, tpr, color=colors[i], linewidth=2,
                        label=f"{res['name']} (AUC={res['auc']:.3f})")

    axes[4].plot([0,1], [0,1], 'k--', alpha=0.5, label='Aleatoire')
    axes[4].set_xlabel('Taux Faux Positifs')
    axes[4].set_ylabel('Taux Vrais Positifs')
    axes[4].set_title('Courbes ROC', fontweight='bold')
    axes[4].legend(fontsize=9)

    # ── Graphique 6 : Feature Importance (Random Forest) ──────
    rf_res = next((r for r in results if 'Forest' in r['name']), None)
    if rf_res and hasattr(rf_res['model'], 'feature_importances_'):
        importances = rf_res['model'].feature_importances_
        # Top 15 features
        indices = np.argsort(importances)[::-1][:15]
        
        feature_names = list(X_test.columns)
        feat_names = [feature_names[i] for i in indices]
        axes[5].barh(feat_names[::-1], importances[indices][::-1],
                     color='#3498db', alpha=0.8)
        axes[5].tick_params(axis='y', labelsize=8)
        axes[5].set_title('Feature Importance — Random Forest (Top 15)', fontweight='bold')
        axes[5].set_xlabel('Importance')

    plt.suptitle('Resultats Classification — Prediction Churn', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'classification_results.png'),
                dpi=150, bbox_inches='tight')
    plt.show()


def run_classification(X_train, X_test, y_train, y_test):
    """
    Entraine 3 modeles de classification pour predire le Churn.

    MODELES :
    1. Regression Logistique : modele lineaire simple et interpretable
    2. Random Forest         : ensemble d'arbres de decision
    3. XGBoost               : boosting gradient, tres performant

    DESEQUILIBRE DES CLASSES :
    67% fideles vs 33% partis. Sans correction, le modele predit
    toujours "fidele" et obtient 67% d'accuracy sans rien apprendre.
    Solution : class_weight='balanced' penalise plus les erreurs
    sur la classe minoritaire (les churners).
    """
    print("=" * 55)
    print("  B) CLASSIFICATION — PREDICTION CHURN")
    print("=" * 55 + "\n")

    print(f"  Desequilibre : {y_train.mean():.1%} churners dans le train")
    print(f"  Solution     : SMOTE (Oversampling) + GridSearch\n")

    # Appliquer SMOTE pour équilibrer les classes
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  [SMOTE] Nouvelles dimensions : {X_train_res.shape}\n")

    results = []

    # ── 1. Regression Logistique ──────────────────────────────
    print("[...] Entrainement Regression Logistique...")
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train_res, y_train_res)
    results.append(evaluate_classifier('Reg. Logistique', lr, X_test, y_test))
    joblib.dump(lr, os.path.join(MODELS_DIR, 'logistic_regression.joblib'))

    # ── 2. Random Forest (OPTIMISÉ) ───────────────────────────
    print("\n[...] Optimisation Random Forest...")
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid_rf,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )
    grid_rf.fit(X_train_res, y_train_res)
    rf = grid_rf.best_estimator_
    
    print(f"  Meilleurs params RF : {grid_rf.best_params_}")
    results.append(evaluate_classifier('Random Forest (opt)', rf, X_test, y_test))
    joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest.joblib'))
    # ── 3. XGBoost (OPTIMISÉ) ─────────────────────────────────
    print("\n[...] Optimisation XGBoost...")

    try:
     from xgboost import XGBClassifier

     scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

     param_grid_xgb = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.05, 0.1],
     }

     grid_xgb = GridSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        ),
        param_grid_xgb,
        cv=3,
        scoring='f1',
        n_jobs=-1
     )

     grid_xgb.fit(X_train_res, y_train_res)

     xgb = grid_xgb.best_estimator_

     print(f"  Meilleurs params XGB : {grid_xgb.best_params_}")

     results.append(evaluate_classifier('XGBoost (opt)', xgb, X_test, y_test))

     joblib.dump(xgb, os.path.join(MODELS_DIR, 'xgboost_optimized.joblib'))

    except ImportError:
     print("  [WARN] XGBoost non installe. Faites : pip install xgboost")
    # ── Meilleur modele ───────────────────────────────────────
    best = max(results, key=lambda r: r['f1'])
    print(f"\n  Meilleur modele (F1-Score) : {best['name']} → {best['f1']:.4f}")
    joblib.dump(best['model'], os.path.join(MODELS_DIR, 'best_classifier.joblib'))
    print(f"  Sauvegarde : models/best_classifier.joblib")

    # ── Graphiques ────────────────────────────────────────────
    plot_classification_results(results, X_test, y_test)

    print(f"\n[OK] Classification terminee\n")
    return results, best


# ═════════════════════════════════════════════
# C) REGRESSION — Predire MonetaryTotal
# ═════════════════════════════════════════════

def evaluate_regressor(name, model, X_test, y_test):
    """Evalue un modele de regression et affiche les metriques."""
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print(f"\n  --- {name} ---")
    print(f"  MAE  (erreur absolue moyenne)  : {mae:.2f} £")
    print(f"  RMSE (racine erreur quadratique): {rmse:.2f} £")
    print(f"  R²   (variance expliquee)       : {r2:.4f}")

    return {
        'name': name, 'model': model,
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'y_pred': y_pred
    }


def plot_regression_results(results, y_test):
    """Trace les graphiques de comparaison des modeles de regression."""

    fig, axes = plt.subplots(2, len(results), figsize=(7*len(results), 12))
    if len(results) == 1:
        axes = axes.reshape(-1, 1)

    colors = ['#3498db', '#2ecc71']

    for i, res in enumerate(results):
        y_pred = res['y_pred']

        # Graphique 1 : Reel vs Predit
        axes[0][i].scatter(y_test, y_pred, alpha=0.3, s=15, color=colors[i])
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0][i].plot([min_val, max_val], [min_val, max_val],
                        'r--', linewidth=2, label='Prediction parfaite')
        axes[0][i].set_xlabel('Valeur Reelle (£)')
        axes[0][i].set_ylabel('Valeur Predite (£)')
        axes[0][i].set_title(f'{res["name"]}\nR²={res["r2"]:.4f}', fontweight='bold')
        axes[0][i].legend()

        # Graphique 2 : Distribution des residus
        residus = y_test.values - y_pred
        axes[1][i].hist(residus, bins=50, color=colors[i], alpha=0.7, edgecolor='white')
        axes[1][i].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1][i].set_xlabel('Residus (Reel - Predit)')
        axes[1][i].set_ylabel('Frequence')
        axes[1][i].set_title(f'Distribution des residus — {res["name"]}', fontweight='bold')

    plt.suptitle('Resultats Regression — Prediction MonetaryTotal',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'regression_results.png'),
                dpi=150, bbox_inches='tight')
    plt.show()


def run_regression(X_train, X_test, y_train, y_test):
    """
    Entraine 2 modeles de regression pour predire MonetaryTotal.

    MODELES :
    1. Regression Lineaire  : suppose une relation lineaire entre X et y
    2. Random Forest Regressor : capture les relations non lineaires

    METRIQUES :
    - MAE  : erreur moyenne en livres sterling (comprehensible)
    - RMSE : penalise plus les grandes erreurs que le MAE
    - R²   : proportion de variance expliquee (1 = parfait, 0 = nul)
    """
    print("=" * 55)
    print("  C) REGRESSION — PREDICTION MONETARYTOTAL")
    print("=" * 55 + "\n")

    # Charger le dataset pour recuperer MonetaryTotal
    df_full = pd.read_csv(PROCESSED)

    if 'MonetaryTotal' not in df_full.columns:
        print("  [WARN] MonetaryTotal absent du dataset. Regression ignoree.")
        return [], None

    # Aligner les indices avec X_train / X_test
    y_reg_train = df_full.loc[X_train.index, 'MonetaryTotal'] \
        if len(df_full) == len(X_train) + len(X_test) \
        else df_full['MonetaryTotal'].iloc[:len(X_train)]

    # Reconstruction propre
    df_full = df_full.reset_index(drop=True)
    train_idx = X_train.index if hasattr(X_train, 'index') else range(len(X_train))
    test_idx  = X_test.index  if hasattr(X_test,  'index') else range(len(X_test))

    # Target regression : MonetaryTotal
    target_reg = df_full['MonetaryTotal']

    # Re-split pour regression
    from sklearn.model_selection import train_test_split
    X_r = df_full.drop(columns=['MonetaryTotal', 'Churn'], errors='ignore')
    X_r = X_r.select_dtypes(include=['int64', 'float64']).fillna(0)
    y_r = df_full['MonetaryTotal']

    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
        X_r, y_r, test_size=0.2, random_state=42
    )

    # Normalisation
    scaler_reg = StandardScaler()
    X_r_train_sc = scaler_reg.fit_transform(X_r_train)
    X_r_test_sc  = scaler_reg.transform(X_r_test)
    joblib.dump(scaler_reg, os.path.join(MODELS_DIR, 'scaler_regression.joblib'))

    results = []

    # ── 1. Regression Lineaire ────────────────────────────────
    print("[...] Entrainement Regression Lineaire...")
    lr = LinearRegression()
    lr.fit(X_r_train_sc, y_r_train)
    results.append(evaluate_regressor('Reg. Lineaire', lr,
                                       X_r_test_sc, y_r_test))
    joblib.dump(lr, os.path.join(MODELS_DIR, 'linear_regression.joblib'))

    # ── 2. Random Forest Regressor (OPTIMISÉ étape 6) ────────────────────────────
    print("\n[...] Optimisation Random Forest Regressor (GridSearch)...")

    param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

    rf_base = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
    ) 

    grid_rf = GridSearchCV(
     rf_base,
     param_grid,
     cv=3,
     scoring='r2',
     n_jobs=-1,
     verbose=1
    )

    grid_rf.fit(X_r_train_sc, y_r_train)

    rf = grid_rf.best_estimator_

    print(f"  Meilleurs params RF : {grid_rf.best_params_}")

    results.append(evaluate_regressor('Random Forest (opt)', rf, X_r_test_sc, y_r_test))

    joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest_optimized.joblib'))
    

    # ── Meilleur modele ───────────────────────────────────────
    best = max(results, key=lambda r: r['r2'])
    print(f"\n  Meilleur modele (R²) : {best['name']} → {best['r2']:.4f}")
    joblib.dump(best['model'], os.path.join(MODELS_DIR, 'best_regressor.joblib'))

    # ── Graphiques ────────────────────────────────────────────
    plot_regression_results(results, y_r_test)

    print(f"\n[OK] Regression terminee\n")
    return results, best


# ═════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═════════════════════════════════════════════

def run_all_models():
    """Lance les 3 types de modelisation dans l'ordre."""

    print("\n" + "=" * 55)
    print("  ETAPE 5 - MODELISATION - DEBUT")
    print("=" * 55 + "\n")

    # Chargement
    X_train, X_test, y_train, y_test = load_train_test()

    # A) Clustering
    kmeans, clusters = run_clustering(X_train, n_clusters=4)

    # B) Classification
    clf_results, best_clf = run_classification(
        X_train, X_test, y_train, y_test
    )

    # C) Regression
    reg_results, best_reg = run_regression(
        X_train, X_test, y_train, y_test
    )

    # ── Recap final ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RECAP FINAL")
    print("=" * 55)
    print(f"\n  [CLUSTERING]")
    print(f"    Modele    : KMeans k=4")
    print(f"    Sauvegarde: models/kmeans.joblib")

    print(f"\n  [CLASSIFICATION]")
    for r in clf_results:
        print(f"    {r['name']:<20} F1={r['f1']:.4f}  AUC={r['auc']:.4f}")
    if best_clf:
        print(f"    Meilleur  : {best_clf['name']}")

    print(f"\n  [REGRESSION]")
    for r in reg_results:
        print(f"    {r['name']:<22} R²={r['r2']:.4f}  MAE={r['mae']:.2f}")
    if best_reg:
        print(f"    Meilleur  : {best_reg['name']}")

    print(f"\n  Graphiques sauvegardes dans : reports/")
    print(f"  Modeles sauvegardes dans    : models/")

    print("\n" + "=" * 55)
    print("  ETAPE 5 - TERMINEE [OK]")
    print("=" * 55)

    return kmeans, clf_results, reg_results


# ─────────────────────────────────────────────
# EXECUTION DIRECTE
# ─────────────────────────────────────────────

if __name__ == '__main__':
    kmeans, clf_results, reg_results = run_all_models()