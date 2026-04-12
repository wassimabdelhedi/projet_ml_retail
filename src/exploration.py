import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

def run_exploration():
    print("\n" + "="*60)
    print("  ÉTAPES D'EXPLORATION ET TRANSFORMATION (ACP)")
    print("="*60 + "\n")

    # 1. Chargement des données
    data_path = os.path.join(DATA_DIR, 'data_clean.csv')
    if not os.path.exists(data_path):
        print(f"❌ Erreur: Le fichier {data_path} n'existe pas. Veuillez lancer le preprocessing d'abord.")
        return

    df = pd.read_csv(data_path)
    print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes\n")

    # 2. Matrice de Corrélation (Heatmap)
    print("[...] Génération de la Heatmap de corrélation...")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'Churn' in num_cols:
        # On place Churn au début pour mieux voir les corrélations
        num_cols = ['Churn'] + [c for c in num_cols if c != 'Churn']
    
    corr_matrix = df[num_cols].corr()
    
    plt.figure(figsize=(24, 18))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matrice de Corrélation des Features (52+ features)', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'correlation_heatmap_detailed.png'), dpi=150)
    print("✅ Heatmap sauvegardée: reports/correlation_heatmap_detailed.png\n")

    # 3. Calcul du VIF (Multicolinéarité)
    print("[...] Calcul du VIF (Variance Inflation Factor)...")
    # On enlève la cible
    X_vif = df[num_cols].drop(columns=['Churn'], errors='ignore').dropna()
    X_vif = X_vif.loc[:, X_vif.var() > 0] # Enlever variance nulle
    
    # Échelle pour VIF
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
    vif_data = vif_data.sort_values(by="VIF", ascending=False)
    
    print("\nTop 10 Features avec plus haut VIF (Risque de multicolinéarité):")
    print(vif_data.head(10).to_string(index=False))
    
    with open(os.path.join(REPORTS_DIR, 'vif_report.txt'), 'w') as f:
        f.write("RAPPORT VIF (MAX RECOMMANDE = 10)\n")
        f.write("="*40 + "\n")
        f.write(vif_data.to_string(index=False))
    print("\n✅ Rapport VIF sauvegardé: reports/vif_report.txt\n")

    # 4. ACP - Analyse en Composantes Principales
    print("[...] Analyse en Composantes Principales (ACP)...")
    X = df.drop(columns=['Churn'], errors='ignore').select_dtypes(include=[np.number])
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=0.90) # Garder 90% de la variance
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"✅ ACP terminée: {pca.n_components_} composantes pour expliquer 90% de la variance.")
    
    # Graphique Scree Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Variance Individuelle')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Variance Cumulée')
    plt.axhline(y=0.9, color='r', linestyle='--', label='Seuil 90%')
    plt.xlabel('Nombre de Composantes')
    plt.ylabel('Ratio de Variance Expliquée')
    plt.title('Graphique d\'Eboulis (Scree Plot) - ACP', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORTS_DIR, 'pca_scree_plot.png'))
    
    # Visualisation 2D
    plt.figure(figsize=(10, 8))
    if 'Churn' in df.columns:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Churn'], cmap='coolwarm', alpha=0.6)
        plt.colorbar(scatter, label='Churn (0=No, 1=Yes)')
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Projection ACP en 2D', fontweight='bold')
    plt.savefig(os.path.join(REPORTS_DIR, 'pca_2d_projection.png'))
    print("✅ Graphiques ACP sauvegardés: reports/pca_scree_plot.png et reports/pca_2d_projection.png\n")

    print("="*60)
    print("  EXPLORATION TERMINÉE")
    print("="*60)

if __name__ == "__main__":
    run_exploration()
