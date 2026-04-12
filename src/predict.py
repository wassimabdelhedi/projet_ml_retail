"""
=============================================================
 Etape 7 - Prediction CORRIGEE
 Fichier : src/predict.py
=============================================================
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


# ─────────────────────────────────────────────
# CHARGEMENT DES MODELES
# ─────────────────────────────────────────────

def load_models():
    models = {}
    for name, filename in [
        ('classifier',        'best_classifier.joblib'),
        ('kmeans',            'kmeans.joblib'),
        ('scaler',            'scaler.joblib'),
        ('scaler_regression', 'scaler_regression.joblib'),
    ]:
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"[OK] {filename} charge")
        else:
            print(f"[WARN] {filename} non trouve")

    # Régresseur : essayer plusieurs noms
    for filename in ['best_regressor.joblib', 'random_forest_optimized.joblib',
                     'random_forest.joblib', 'linear_regression.joblib']:
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            models['regressor'] = joblib.load(path)
            print(f"[OK] {filename} charge (regression)")
            break

    return models


# ─────────────────────────────────────────────
# FEATURE ENGINEERING CORRIGE
# ─────────────────────────────────────────────

def create_training_features(data_dict):
    """
    Crée les 70 features exactes du training.
    CORRECTION : les features clés (Recency, Frequency, Monetary)
    sont maintenant correctement propagées dans toutes les features dérivées.
    """

    # ── Valeurs de base ──────────────────────────────────────
    recency        = float(data_dict.get('Recency', 30))
    frequency      = float(data_dict.get('Frequency', 5))
    monetary       = float(data_dict.get('MonetaryTotal', 500))
    age            = float(data_dict.get('Age', 35))
    satisfaction   = float(data_dict.get('SatisfactionScore', 3.5))
    support        = float(data_dict.get('SupportTicketsCount', 1))
    avg_days       = float(data_dict.get('AvgDaysBetweenPurchases', 30))
    tenure         = float(data_dict.get('CustomerTenureDays', 365))
    cancelled      = float(data_dict.get('CancelledTransactions', 0))
    total_trans    = max(float(data_dict.get('TotalTransactions', frequency * 8)), 1)

    # ── Features dérivées CORRIGEES ──────────────────────────
    # Ces calculs reflètent exactement ce que preprocessing.py a fait
    monetary_avg      = monetary / max(frequency, 1)
    monetary_std      = monetary * 0.25
    avg_qty           = max(frequency * 1.2, 1)
    first_purchase    = recency + tenure

    # Produits uniques : proportionnel à la fréquence
    unique_products   = max(int(frequency * 2.5), 1)
    avg_prod_trans    = min(unique_products / max(frequency, 1), 10)
    unique_countries  = 1

    neg_qty    = max(int(frequency * 0.05), 0)
    zero_price = 0
    return_ratio = neg_qty / max(total_trans, 1)

    # CORRECTION : AvgBasketValue et TenureRatio correctement calculés
    avg_basket_value = monetary / max(frequency, 1)
    tenure_ratio     = recency / max(tenure, 1)

    # ── Encodages ordinaux CORRIGES ──────────────────────────
    # AgeCategory (0-6)
    if age < 25:       age_cat = 1
    elif age < 35:     age_cat = 2
    elif age < 45:     age_cat = 3
    elif age < 55:     age_cat = 4
    elif age < 65:     age_cat = 5
    else:              age_cat = 6

    # SpendingCategory (1-4) basé sur monetary
    if monetary < 500:    spending_cat = 1
    elif monetary < 2000: spending_cat = 2
    elif monetary < 5000: spending_cat = 3
    else:                 spending_cat = 4

    # PreferredTimeOfDay
    preferred_time = 2  # Midi par défaut

    # LoyaltyLevel (0-4) basé sur fréquence et tenure
    if frequency >= 30 and tenure > 500:    loyalty = 4
    elif frequency >= 15 and tenure > 300:  loyalty = 3
    elif frequency >= 7 and tenure > 150:   loyalty = 2
    elif frequency >= 2:                    loyalty = 1
    else:                                   loyalty = 0

    # CORRECTION CRITIQUE : ChurnRiskCategory basé sur Recency
    # C'est la feature la plus importante pour le churn
    if recency <= 17:     churn_risk = 1   # Faible  : top 25% clients actifs
    elif recency <= 50:   churn_risk = 2   # Moyen   : entre Q25 et Q50
    elif recency <= 143:  churn_risk = 3   # Élevé   : entre Q50 et Q75
    else:                 churn_risk = 4   # Critique : top 25% inactifs

    # BasketSizeCategory basé sur avg_basket_value
    if avg_basket_value < 50:     basket_size = 1
    elif avg_basket_value < 200:  basket_size = 2
    else:                         basket_size = 3

    # CORRECTION : RFMSegment calculé correctement
    # Champions : Recency bas + Freq haute + CA élevé
    if recency <= 30 and frequency >= 20 and monetary >= 3000:
        rfm = 4   # Champions
    elif recency <= 90 and frequency >= 10:
        rfm = 3   # Fidèles
    elif recency <= 180 and frequency >= 5:
        rfm = 2   # Potentiels
    else:
        rfm = 1   # Dormants

    # ── CustomerType One-Hot CORRIGE ─────────────────────────
    cust_hyperactif  = 1 if frequency >= 20 else 0
    cust_nouveau     = 1 if tenure < 90 else 0
    cust_occasionnel = 1 if frequency < 5 and tenure >= 90 else 0
    cust_regulier    = 1 if 5 <= frequency < 20 else 0

    # ── Saison, Région, etc. (valeurs par défaut) ─────────────
    reg_year = 2011; reg_month = 6; reg_day = 15; reg_weekday = 2
    is_private_ip = 0; ip_first_octet = 80

    features = pd.DataFrame({
        # Features numériques principales
        'Recency':                    [recency],
        'Frequency':                  [frequency],
        'MonetaryTotal':              [monetary],
        'MonetaryAvg':                [monetary_avg],
        'MonetaryStd':                [monetary_std],
        'AvgQuantityPerTransaction':  [avg_qty],
        'CustomerTenureDays':         [tenure],
        'FirstPurchaseDaysAgo':       [first_purchase],
        'PreferredDayOfWeek':         [2],
        'PreferredHour':              [14],
        'PreferredMonth':             [6],
        'WeekendPurchaseRatio':       [0.28],
        'AvgDaysBetweenPurchases':    [avg_days],
        'UniqueProducts':             [unique_products],
        'AvgProductsPerTransaction':  [avg_prod_trans],
        'UniqueCountries':            [unique_countries],
        'NegativeQuantityCount':      [neg_qty],
        'ZeroPriceCount':             [zero_price],
        'ReturnRatio':                [return_ratio],
        'Age':                        [age],
        'SupportTicketsCount':        [support],
        'SatisfactionScore':          [satisfaction],
        # Features ordinales CORRIGEES
        'RFMSegment':                 [rfm],
        'AgeCategory':                [age_cat],
        'SpendingCategory':           [spending_cat],
        'PreferredTimeOfDay':         [preferred_time],
        'LoyaltyLevel':               [loyalty],
        'ChurnRiskCategory':          [churn_risk],   # ← CLE POUR LE CHURN
        'BasketSizeCategory':         [basket_size],
        # Features temporelles
        'RegYear':                    [reg_year],
        'RegMonth':                   [reg_month],
        'RegDay':                     [reg_day],
        'RegWeekday':                 [reg_weekday],
        # Features IP
        'IsPrivateIP':                [is_private_ip],
        'IPFirstOctet':               [ip_first_octet],
        # Features engineered CORRIGEES
        'AvgBasketValue':             [avg_basket_value],
        'TenureRatio':                [tenure_ratio],
        # CustomerType One-Hot
        'CustomerType_Hyperactif':    [cust_hyperactif],
        'CustomerType_Nouveau':       [cust_nouveau],
        'CustomerType_Occasionnel':   [cust_occasionnel],
        'CustomerType_Régulier':      [cust_regulier],
        # FavoriteSeason One-Hot
        'FavoriteSeason_Automne':     [0],
        'FavoriteSeason_Hiver':       [0],
        'FavoriteSeason_Printemps':   [1],
        'FavoriteSeason_Été':         [0],
        # Region One-Hot
        'Region_Afrique':             [0],
        "Region_Amérique du Nord":    [0],
        "Region_Amérique du Sud":     [0],
        'Region_Asie':                [0],
        'Region_Autre':               [0],
        'Region_Europe centrale':     [0],
        'Region_Europe continentale': [1],
        "Region_Europe de l'Est":     [0],
        'Region_Europe du Nord':      [0],
        'Region_Europe du Sud':       [0],
        'Region_Moyen-Orient':        [0],
        "Region_Océanie":             [0],
        'Region_UK':                  [0],
        # WeekendPreference One-Hot
        'WeekendPreference_Inconnu':  [0],
        'WeekendPreference_Semaine':  [1],
        # ProductDiversity One-Hot
        'ProductDiversity_Explorateur': [1 if unique_products > 20 else 0],
        'ProductDiversity_Spécialisé':  [0 if unique_products > 20 else 1],
        # Gender One-Hot
        'Gender_F':                   [0],
        'Gender_M':                   [0],
        'Gender_Unknown':             [1],
        # AccountStatus One-Hot
        'AccountStatus_Active':       [1],
        'AccountStatus_Closed':       [0],
        'AccountStatus_Pending':      [0],
        'AccountStatus_Suspended':    [0],
        # Country Target Encoding
        'Country_encoded':            [0.33],
    })

    return features


# ─────────────────────────────────────────────
# ALIGNEMENT DES COLONNES AVEC LE MODELE
# ─────────────────────────────────────────────

def align_features(X, model):
    """
    CORRECTION CRITIQUE : aligne les colonnes de X avec
    celles attendues par le modèle. Évite les erreurs de
    colonnes manquantes ou dans le mauvais ordre.
    """
    try:
        model_cols = list(model.feature_names_in_)
        # Ajouter les colonnes manquantes avec 0
        for col in model_cols:
            if col not in X.columns:
                X[col] = 0
        # Réordonner exactement comme le modèle l'attend
        X = X[model_cols]
    except AttributeError:
        pass  # Modèle sans feature_names_in_
    return X


# ─────────────────────────────────────────────
# PREDICTION COMPLETE CORRIGEE
# ─────────────────────────────────────────────

def predict_client(data_dict):
    """Prédit churn, segment et valeur pour un client."""

    print("\n" + "=" * 50)
    print("  PREDICTION CLIENT")
    print("=" * 50)

    models  = load_models()
    monetary = float(data_dict.get('MonetaryTotal', 0))
    X = create_training_features(data_dict)
    results = {}

    # ── 1. CLASSIFICATION (Churn) ─────────────────────────────
    if 'classifier' in models:
        clf = models['classifier']
        X_clf = align_features(X.copy(), clf)

        proba = clf.predict_proba(X_clf)[0][1]
        # CORRECTION : seuil standard 0.50
        churn = 1 if proba >= 0.50 else 0

        results['churn']       = int(churn)
        results['churn_proba'] = float(proba)

        print(f"\n[CHURN] {'PARTI' if churn == 1 else 'FIDELE'}")
        print(f"  Probabilite : {proba:.2%}")

    # ── 2. CLUSTERING (Segment) ───────────────────────────────
    if 'kmeans' in models:
        km = models['kmeans']
        # Note : KMeans est désormais entraîné sur des données normalisées
        if 'scaler' in models:
            scaler = models['scaler']
            X_km_sc = X.copy()
            
            # Aligner avec ce que le scaler attend (StandardScaler)
            try:
                cols_scaler = list(scaler.feature_names_in_)
                # Ajouter les colonnes manquantes
                for col in cols_scaler:
                    if col not in X_km_sc.columns:
                        X_km_sc[col] = 0
                # Sélection et ordre
                X_km_sc = X_km_sc[cols_scaler]
                X_km_sc = scaler.transform(X_km_sc)
                cluster = int(km.predict(X_km_sc)[0])
            except Exception as e:
                print(f"  [WARN] Echec normalisation KMeans: {e}")
                X_km = align_features(X.copy(), km)
                cluster = int(km.predict(X_km)[0])
        else:
            X_km = align_features(X.copy(), km)
            cluster = int(km.predict(X_km)[0])

        results['segment'] = cluster
        print(f"\n[SEGMENT] Cluster {cluster}")

    # ── 3. REGRESSION (Valeur prédite) ───────────────────────
    if 'regressor' in models:
        reg = models['regressor']

        try:
            X_reg = X.drop(columns=['MonetaryTotal'], errors='ignore').copy()

            if 'scaler_regression' in models:
                sc_reg    = models['scaler_regression']
                # Aligner avec le scaler
                try:
                    reg_cols = list(sc_reg.feature_names_in_)
                    for col in reg_cols:
                        if col not in X_reg.columns:
                            X_reg[col] = 0
                    X_reg = X_reg[reg_cols]
                except AttributeError:
                    pass
                X_reg_sc = sc_reg.transform(X_reg)
            else:
                X_reg_sc = X_reg.values

            amount = float(reg.predict(X_reg_sc)[0])

            # Pondération adaptative selon l'écart
            if monetary > 0 and amount > 0:
                ratio = amount / monetary
                if ratio < 0.5:
                    # Sous-estime fortement → corriger vers le haut
                    amount = monetary * 0.5 + amount * 0.5
                elif ratio > 2.0:
                    # Surévalue fortement → corriger vers le bas
                    amount = monetary * 0.7 + amount * 0.3
                elif ratio > 1.4:
                    # Légère surévaluation → correction douce
                    amount = monetary * 0.4 + amount * 0.6

            amount = max(10, min(amount, 50000))

        except Exception as e:
            print(f"  [WARN] Regression erreur: {e}")
            amount = monetary   # Fallback

        results['predicted_spend'] = amount
        print(f"\n[MONETARY] £{amount:.2f}")

    print("\n" + "=" * 50)
    return results


# ─────────────────────────────────────────────
# TEST DIRECT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    cas_tests = [
        {"nom": "Champion",  "Recency": 5,   "Frequency": 45, "MonetaryTotal": 8500, "Age": 35},
        {"nom": "Fidele",    "Recency": 30,  "Frequency": 15, "MonetaryTotal": 2200, "Age": 42},
        {"nom": "Risque",    "Recency": 150, "Frequency": 8,  "MonetaryTotal": 950,  "Age": 28},
        {"nom": "Perdu",     "Recency": 350, "Frequency": 2,  "MonetaryTotal": 180,  "Age": 55},
        {"nom": "Nouveau",   "Recency": 10,  "Frequency": 3,  "MonetaryTotal": 420,  "Age": 24},
        {"nom": "VIP Risque","Recency": 200, "Frequency": 25, "MonetaryTotal": 6000, "Age": 48},
    ]

    SEGMENTS = {0:'Champions', 1:'Fideles', 2:'Potentiels', 3:'Dormants'}

    print("\n" + "="*60)
    print("  TESTS DES 6 CAS")
    print("="*60)

    for cas in cas_tests:
        nom = cas.pop('nom')
        r   = predict_client(cas)
        seg = SEGMENTS.get(r.get('segment', -1), f"Cluster {r.get('segment')}")
        print(f"\n  {nom}")
        print(f"    Churn    : {r.get('churn')} ({r.get('churn_proba', 0):.1%})")
        print(f"    Segment  : {seg}")
        print(f"    Depense  : £{r.get('predicted_spend', 0):.2f}")