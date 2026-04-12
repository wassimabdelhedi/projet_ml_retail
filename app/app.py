"""
=============================================================
 Etape 8 - Flask CORRIGE
 Fichier : app/app.py
=============================================================
"""
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from flask import Flask, render_template, request, jsonify
from predict import predict_client
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ─────────────────────────────────────────────
# CALCUL AUTOMATIQUE DU VRAI MAPPING CLUSTER
# ─────────────────────────────────────────────

def compute_cluster_mapping():
    """
    Calcule le vrai mapping cluster → nom en utilisant
    les CENTROIDES du KMeans (dans l'espace normalisé).

    POURQUOI LES CENTROIDES :
    X_train est normalisé (StandardScaler → moyenne=0, std=1)
    Les centroides sont stockés dans ce même espace normalisé
    On peut donc les comparer directement sans charger X_train.

    SCORE RFM = -Recency + Frequency
    Plus Recency est basse (achat récent) et Frequency est haute
    (beaucoup d'achats), meilleur est le client.

    Champions  → score RFM le plus élevé
    Fidèles    → score RFM élevé
    Potentiels → score RFM moyen
    Dormants   → score RFM le plus bas
    """
    MODELS_DIR     = os.path.join(BASE_DIR, 'models')
    TRAIN_TEST_DIR = os.path.join(BASE_DIR, 'data', 'train_test')

    default_mapping = {
        0: {'name': 'Champions',  'emoji': '🏆'},
        1: {'name': 'Fidèles',    'emoji': '⭐'},
        2: {'name': 'Potentiels', 'emoji': '🌱'},
        3: {'name': 'Dormants',   'emoji': '💤'},
    }

    try:
        kmeans  = joblib.load(os.path.join(MODELS_DIR, 'kmeans.joblib'))
        X_train = pd.read_csv(os.path.join(TRAIN_TEST_DIR, 'X_train.csv'))

        # ── Trouver les index de Recency et Frequency ─────────
        cols = list(X_train.columns)

        if 'Recency' not in cols or 'Frequency' not in cols:
            print("[WARN] Recency/Frequency absents, mapping par défaut")
            return default_mapping

        idx_rec = cols.index('Recency')
        idx_frq = cols.index('Frequency')

        # ── Lire les centroides dans l'espace normalisé ───────
        # centroides shape : (n_clusters, n_features)
        centroides = kmeans.cluster_centers_

        # ── Calculer le score RFM pour chaque centroide ───────
        # Score = -Recency + Frequency
        # (Recency basse = bon) + (Frequency haute = bon)
        scores = {}
        for i in range(kmeans.n_clusters):
            rec_norm = centroides[i][idx_rec]
            frq_norm = centroides[i][idx_frq]
            scores[i] = -rec_norm + frq_norm   # score RFM normalisé

        # ── Trier par score décroissant ───────────────────────
        # Rang 0 (meilleur score) = Champions
        # Rang 1                  = Fidèles
        # Rang 2                  = Potentiels
        # Rang 3 (pire score)     = Dormants
        sorted_clusters = sorted(scores.keys(),
                                 key=lambda i: scores[i],
                                 reverse=True)

        noms   = ['Champions', 'Fidèles', 'Potentiels', 'Dormants']
        emojis = ['🏆', '⭐', '🌱', '💤']

        mapping = {}
        for rang, cluster_id in enumerate(sorted_clusters):
            mapping[cluster_id] = {
                'name':  noms[rang],
                'emoji': emojis[rang],
            }

        # ── Afficher le mapping calculé ───────────────────────
        print("\n[OK] Mapping clusters calculé (via centroides) :")
        for cluster_id in sorted(mapping.keys()):
            info    = mapping[cluster_id]
            rec_n   = centroides[cluster_id][idx_rec]
            frq_n   = centroides[cluster_id][idx_frq]
            score   = scores[cluster_id]
            print(f"  Cluster {cluster_id} → {info['name']} {info['emoji']}"
                  f"  (Recency={rec_n:+.3f}, "
                  f"Freq={frq_n:+.3f}, "
                  f"Score={score:+.3f})")

        return mapping

    except Exception as e:
        print(f"[WARN] Mapping auto impossible ({e}), mapping par défaut")
        return default_mapping


def get_spend_category(spend):
    """Catégorie de dépense selon le montant."""
    if spend < 100:    return 'Très petit panier'
    elif spend < 300:  return 'Petit panier'
    elif spend < 500:  return 'Panier moyen'
    elif spend < 2000: return 'Bon panier'
    elif spend < 5000: return 'Gros panier'
    else:              return 'VIP'


# Calcul du mapping au démarrage (une seule fois)
SEGMENTS = compute_cluster_mapping()


# ─────────────────────────────────────────────
# ROUTES FLASK
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        client_data = {
            'Recency':               float(data.get('Recency', 30)),
            'Frequency':             float(data.get('Frequency', 5)),
            'MonetaryTotal':         float(data.get('MonetaryTotal', 500)),
            'Age':                   float(data.get('Age', 35)),
            'SatisfactionScore':     float(data.get('SatisfactionScore', 3.5)),
            'SupportTicketsCount':   float(data.get('SupportTicketsCount', 0)),
            'AvgDaysBetweenPurchases': float(data.get('AvgDaysBetweenPurchases', 30)),
            'CustomerTenureDays':    float(data.get('CustomerTenureDays', 365)),
            'CancelledTransactions': float(data.get('CancelledTransactions', 0)),
            'TotalTransactions':     float(data.get('TotalTransactions', 5)),
        }

        results    = predict_client(client_data)
        churn      = int(results.get('churn', 0))
        proba      = float(results.get('churn_proba', 0) or 0)
        segment_id = int(results.get('segment', 0))
        spend      = float(results.get('predicted_spend', 0) or 0)
        churn_pct  = round(proba * 100, 1)

        # Mapping corrigé automatiquement (IA Pure)
        seg_info = SEGMENTS.get(segment_id, {
            'name': f'Cluster {segment_id}', 'emoji': '👤'
        })

        # FORMAT EXACT pour main.js
        return jsonify({
            'success': True,
            'predictions': {
                'churn':             churn,
                'churn_probability': churn_pct,
                'predicted_spend':   round(spend, 2),
            },
            'interpretations': {
                'segment_profile': {
                    'name':  seg_info['name'],
                    'emoji': seg_info['emoji'],
                },
                'spend_category': get_spend_category(spend),
            }
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n🌐 Flask sur http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)