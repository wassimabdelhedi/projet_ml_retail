import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# CHEMINS

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED    = os.path.join(BASE_DIR, 'data', 'processed', 'data_clean.csv')
REPORTS_DIR  = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. FEATURE ENGINEERING (nouvelles features)

def feature_engineering(df):
    """
    Cree de nouvelles features a partir des existantes.
    Ces features capturent des relations non visibles dans les donnees brutes.
    """
    print("[...] Feature Engineering...")

    created = []

    # Depenses par jour d'inactivite
    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
        created.append('MonetaryPerDay')

    # Valeur moyenne du panier
    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)
        created.append('AvgBasketValue')

    # Ratio recence / anciennete (client actif ou endormi ?)
    if 'Recency' in df.columns and 'CustomerTenureDays' in df.columns:
        df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)
        created.append('TenureRatio')

    # Taux d'annulation
    if 'CancelledTransactions' in df.columns and 'TotalTransactions' in df.columns:
        df['CancelRate'] = df['CancelledTransactions'] / (df['TotalTransactions'] + 1)
        created.append('CancelRate')

    # Valeur par produit unique
    if 'MonetaryTotal' in df.columns and 'UniqueProducts' in df.columns:
        df['ValuePerProduct'] = df['MonetaryTotal'] / (df['UniqueProducts'] + 1)
        created.append('ValuePerProduct')

    # Engagement : frequence × produits distincts
    if 'Frequency' in df.columns and 'UniqueProducts' in df.columns:
        df['EngagementScore'] = df['Frequency'] * df['UniqueProducts']
        created.append('EngagementScore')

    print(f"[OK] {len(created)} features creees : {created}\n")
    return df, created

# 2. SUPPRESSION MULTICOLINEARITE (corr > seuil)

def remove_correlated_features(df, target_col='Churn', threshold=0.85):
    """
    Supprime les features numeriques trop correlees entre elles (|corr| > threshold).
    
    POURQUOI : Des features tres correlees portent la meme information.
    Garder les deux perturbe les modeles lineaires et ralentit les calculs.
    
    METHODE :
    - Calcule la matrice de correlation
    - Parcourt la matrice triangulaire superieure
    - Supprime la colonne la moins correlée avec Churn (garde la plus informative)
    """
    print(f"[...] Suppression features correlees (seuil = {threshold})...")

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    corr_matrix = df[num_cols].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = []
    for col in upper.columns:
        if any(upper[col] > threshold):
            # Garder celle la plus correlee avec Churn
            corr_with_churn = df[num_cols + [target_col]].corr()[target_col].abs()
            paired = [c for c in upper.index if upper.loc[c, col] > threshold]
            paired.append(col)
            least_useful = min(paired, key=lambda c: corr_with_churn.get(c, 0))
            if least_useful not in to_drop:
                to_drop.append(least_useful)

    to_drop = [c for c in to_drop if c in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"  {len(to_drop)} colonnes supprimees :")
        for col in to_drop:
            print(f"    - {col}")
    else:
        print("  Aucune colonne redondante detectee")

    print(f"[OK] Shape apres suppression : {df.shape}\n")
    return df, to_drop

# 3. CALCUL DU VIF

def compute_vif(df, target_col='Churn', max_vif=10.0):
    """
    Calcule le VIF (Variance Inflation Factor) pour chaque feature numerique.

    POURQUOI : La correlation detecte les paires de variables similaires.
    Le VIF detecte la multicolinearite multiple : une variable expliquee
    par PLUSIEURS autres en meme temps.

    INTERPRETATION :
    - VIF = 1        : aucune multicolinearite
    - VIF entre 1-5  : acceptable
    - VIF entre 5-10 : moderee, surveiller
    - VIF > 10       : severe → supprimer la feature

    METHODE : Pour chaque feature X_i, on fait une regression de X_i
    sur toutes les autres. VIF = 1 / (1 - R²).
    """
    print(f"[...] Calcul du VIF (seuil severe = {max_vif})...")

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    # Supprimer les colonnes avec variance nulle
    df_vif = df[num_cols].dropna()
    df_vif = df_vif.loc[:, df_vif.var() > 0]

    vif_data = []
    for i, col in enumerate(df_vif.columns):
        try:
            vif_val = variance_inflation_factor(df_vif.values, i)
            vif_data.append({'Feature': col, 'VIF': round(vif_val, 2)})
        except Exception:
            vif_data.append({'Feature': col, 'VIF': np.nan})

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

    # Affichage
    print(f"\n  {'Feature':<35} {'VIF':>8}  {'Niveau'}")
    print("  " + "-" * 60)
    for _, row in vif_df.iterrows():
        v = row['VIF']
        if pd.isna(v):
            niveau = "N/A"
        elif v > 10:
            niveau = "SEVERE   ← supprimer"
        elif v > 5:
            niveau = "Moderee"
        else:
            niveau = "OK"
        print(f"  {row['Feature']:<35} {v:>8.2f}  {niveau}")

    severe = vif_df[vif_df['VIF'] > max_vif]['Feature'].tolist()
    print(f"\n  Features VIF severe (>{max_vif}) : {len(severe)}")
    if severe:
        print(f"  A envisager : {severe[:5]}")

    print(f"[OK] VIF calcule\n")
    return vif_df

# 4. ANALYSE EN COMPOSANTES PRINCIPALES (ACP)

def run_pca(df, target_col='Churn', n_components=10, save_dir=None):
    """
    Applique l'ACP (Analyse en Composantes Principales).

    POURQUOI :
    - Reduit 70 features → 2-10 composantes
    - Elimine le bruit et la redondance
    - Permet la visualisation en 2D/3D
    - Accelere l'entrainement des modeles
    - Evite la malediction de la dimensionnalite

    COMMENT CA MARCHE :
    L'ACP cherche les directions (axes) dans lesquelles les donnees
    varient le plus. La 1ere composante capture le maximum de variance,
    la 2eme le maximum restant, etc.
    On garde suffisamment de composantes pour expliquer 80-90% de la variance.

    ATTENTION : L'ACP necessite des donnees normalisees au prealable.
    """
    print(f"[...] ACP sur les features numeriques (n_components={n_components})...")

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    X = df[num_cols].fillna(0)
    y = df[target_col] if target_col in df.columns else None

    # Normalisation avant ACP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ACP
    pca = PCA(n_components=min(n_components, len(num_cols)))
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print(f"\n  Variance expliquee par composante :")
    print(f"  {'CP':<6} {'Variance':>10}  {'Cumulee':>10}")
    print("  " + "-" * 32)
    for i, (exp, cum) in enumerate(zip(explained, cumulative)):
        marker = " <-- 80%" if abs(cum - 0.8) == min(abs(cumulative - 0.8)) else ""
        print(f"  CP{i+1:<4} {exp*100:>9.2f}%  {cum*100:>9.2f}%{marker}")

    n_80 = np.argmax(cumulative >= 0.80) + 1
    n_90 = np.argmax(cumulative >= 0.90) + 1
    print(f"\n  Composantes pour 80% de variance : {n_80}")
    print(f"  Composantes pour 90% de variance : {n_90}")

    #  Graphique 1 : Variance expliquee 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(1, len(explained)+1), explained*100,
                color='#3498db', alpha=0.8, edgecolor='white')
    axes[0].plot(range(1, len(explained)+1), cumulative*100,
                 'ro-', linewidth=2, markersize=6, label='Variance cumulee')
    axes[0].axhline(y=80, color='green', linestyle='--', alpha=0.7, label='Seuil 80%')
    axes[0].axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='Seuil 90%')
    axes[0].set_xlabel('Composante Principale')
    axes[0].set_ylabel('Variance expliquee (%)')
    axes[0].set_title('Variance expliquee par composante (Scree Plot)', fontweight='bold')
    axes[0].legend()

    #  Graphique 2 : Projection 2D 
    colors = ['#2ecc71', '#e74c3c']
    labels = ['Fidele (0)', 'Parti (1)']

    if y is not None:
        for val, color, label in zip([0, 1], colors, labels):
            mask = y == val
            axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                           c=color, alpha=0.4, s=15, label=label)
        axes[1].legend(fontsize=10)
    else:
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                       c='#3498db', alpha=0.4, s=15)

    axes[1].set_xlabel(f'CP1 ({explained[0]*100:.1f}% variance)')
    axes[1].set_ylabel(f'CP2 ({explained[1]*100:.1f}% variance)')
    axes[1].set_title('Projection 2D — CP1 vs CP2', fontweight='bold')

    plt.suptitle('Analyse en Composantes Principales (ACP)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, 'pca_analysis.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Graphique sauvegarde : {path}")
    plt.show()

    print(f"[OK] ACP terminee\n")
    return pca, X_pca, explained

# 5. VISUALISATION CLUSTERS EN 2D (ACP + KMeans)

def visualize_clusters_2d(df, target_col='Churn', n_clusters=4, save_dir=None):
    """
    Combine ACP (reduction 2D) + KMeans (clustering) pour visualiser
    les groupes de clients dans un espace 2D.

    POURQUOI :
    On ne peut pas visualiser 70 dimensions. L'ACP projette les clients
    sur 2 axes principaux, puis KMeans groupe les clients similaires.
    C'est une technique exploratoire pour comprendre la structure des donnees.

    METHODE COUDE (Elbow Method) :
    Pour choisir le bon nombre de clusters k, on trace l'inertie
    (distance des points a leur centroide) en fonction de k.
    Le "coude" de la courbe indique le k optimal.
    """
    print(f"[...] Visualisation clusters 2D (k={n_clusters})...")

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)

    X = df[num_cols].fillna(0)
    y = df[target_col] if target_col in df.columns else None

    # Normalisation + ACP 2D
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca2    = PCA(n_components=2)
    X_2d    = pca2.fit_transform(X_scaled)

    #  Methode du coude 
    inertias = []
    k_range  = range(2, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_2d)
        inertias.append(km.inertia_)

    #  KMeans avec k choisi 
    kmeans   = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_2d)

    # Calcul Silhouette Score
    from sklearn.metrics import silhouette_score
    sil_score = silhouette_score(X_2d, clusters)
    print(f"  Silhouette Score (k={n_clusters}) : {sil_score:.4f}")
    print(f"  (0=mauvais | 0.5=moyen | 1=parfait)")

    #  Figure : 3 graphiques 
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Graphique 1 : Methode du coude
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(x=n_clusters, color='red', linestyle='--',
                    label=f'k={n_clusters} choisi')
    axes[0].set_xlabel('Nombre de clusters (k)')
    axes[0].set_ylabel('Inertie')
    axes[0].set_title('Methode du Coude', fontweight='bold')
    axes[0].legend()

    # Graphique 2 : Clusters KMeans
    colors_clusters = ['#e74c3c','#3498db','#2ecc71','#f39c12',
                       '#9b59b6','#1abc9c','#e67e22','#34495e']
    for i in range(n_clusters):
        mask = clusters == i
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=colors_clusters[i], alpha=0.5, s=15,
                        label=f'Cluster {i}')
    centers = kmeans.cluster_centers_
    axes[1].scatter(centers[:, 0], centers[:, 1],
                    c='black', marker='X', s=200, zorder=5, label='Centroides')
    axes[1].set_xlabel(f'CP1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)')
    axes[1].set_ylabel(f'CP2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)')
    axes[1].set_title(f'Clusters KMeans (k={n_clusters})', fontweight='bold')
    axes[1].legend(fontsize=8)

    # Graphique 3 : Churn par cluster
    if y is not None:
        df_viz = pd.DataFrame({'CP1': X_2d[:,0], 'CP2': X_2d[:,1],
                               'Cluster': clusters, 'Churn': y.values})
        churn_rate = df_viz.groupby('Cluster')['Churn'].mean()
        bar_colors = ['#e74c3c' if r > 0.5 else '#f39c12' if r > 0.3 else '#2ecc71'
                      for r in churn_rate.values]
        bars = axes[2].bar([f'Cluster {i}' for i in churn_rate.index],
                           churn_rate.values * 100, color=bar_colors, edgecolor='white')
        axes[2].set_ylabel('Taux de Churn (%)')
        axes[2].set_title('Taux de Churn par Cluster', fontweight='bold')
        axes[2].axhline(y=df[target_col].mean()*100, color='blue',
                        linestyle='--', alpha=0.7, label='Moyenne globale')
        axes[2].legend()
        for bar, val in zip(bars, churn_rate.values):
            axes[2].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.5,
                        f'{val*100:.1f}%', ha='center', fontweight='bold')

        print(f"\n  Taux de Churn par cluster :")
        for i, rate in churn_rate.items():
            profil = "RISQUE ELEVE" if rate > 0.5 else "Risque moyen" if rate > 0.3 else "Fidele"
            print(f"    Cluster {i} : {rate*100:.1f}%  → {profil}")

    plt.suptitle('Segmentation Clients — ACP + KMeans', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, 'clusters_2d.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\n  Graphique sauvegarde : {path}")
    plt.show()

    print(f"[OK] Visualisation clusters terminee\n")
    return clusters, kmeans, X_2d

# PIPELINE PRINCIPAL

def run_feature_selection(input_path=None, save_dir=None):
    """
    Lance toute l'etape 3 dans l'ordre :
    1. Chargement data_clean.csv
    2. Feature Engineering (nouvelles features)
    3. Suppression multicolinearite (corr > 0.85)
    4. Calcul VIF
    5. ACP (scree plot + projection 2D)
    6. Visualisation clusters KMeans
    """
    print("\n" + "=" * 55)
    print("  ETAPE 3 - FEATURE ENGINEERING & SELECTION")
    print("=" * 55 + "\n")

    if input_path is None:
        input_path = PROCESSED
    if save_dir is None:
        save_dir = REPORTS_DIR

    # Chargement
    df = pd.read_csv(input_path)
    print(f"[OK] Dataset charge : {df.shape}\n")

    # 1. Feature Engineering
    df, new_features = feature_engineering(df)

    # 2. Suppression multicolinearite
    df, dropped = remove_correlated_features(df, threshold=0.85)

    # 3. VIF
    vif_df = compute_vif(df)

    # 4. ACP
    pca, X_pca, explained = run_pca(df, n_components=10, save_dir=save_dir)

    # 5. Clusters 2D
    clusters, kmeans, X_2d = visualize_clusters_2d(df, n_clusters=4, save_dir=save_dir)

    print("=" * 55)
    print("  ETAPE 3 - TERMINEE [OK]")
    print("=" * 55)
    print(f"\n  Shape finale        : {df.shape}")
    print(f"  Features creees     : {new_features}")
    print(f"  Features supprimees : {len(dropped)}")
    print(f"  Graphiques          : {save_dir}")

    return df, pca, X_pca, clusters


# EXECUTION DIRECTE

if __name__ == '__main__':
    df, pca, X_pca, clusters = run_feature_selection()