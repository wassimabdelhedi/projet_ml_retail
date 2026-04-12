
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import ipaddress
import os
import joblib
import warnings
warnings.filterwarnings('ignore')
import sys
sys.stdout.reconfigure(encoding='utf-8')


#  0. CHARGEMENT DES DONNEES

def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"[OK] Dataset charge : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    return df


#  1. SUPPRESSION FEATURES INUTILES 

def drop_useless_features(df):
    cols_to_drop = []
    if 'NewsletterSubscribed' in df.columns:
        if df['NewsletterSubscribed'].nunique() == 1:
            cols_to_drop.append('NewsletterSubscribed')
            print("  [DROP] NewsletterSubscribed (variance nulle)")
    if 'CustomerID' in df.columns:
        cols_to_drop.append('CustomerID')
        print("  [DROP] CustomerID (identifiant non predictif)")
    df = df.drop(columns=cols_to_drop)
    print(f"[OK] {len(cols_to_drop)} colonnes supprimees\n")
    return df


#  2. PARSING REGISTRATIONDATE 

def parse_registration_date(df):
    if 'RegistrationDate' not in df.columns:
        return df
    df['RegistrationDate'] = pd.to_datetime(
        df['RegistrationDate'], dayfirst=True, errors='coerce'
    )
    df['RegYear']    = df['RegistrationDate'].dt.year
    df['RegMonth']   = df['RegistrationDate'].dt.month
    df['RegDay']     = df['RegistrationDate'].dt.day
    df['RegWeekday'] = df['RegistrationDate'].dt.weekday
    df = df.drop(columns=['RegistrationDate'])
    print("[OK] RegistrationDate parsee → RegYear, RegMonth, RegDay, RegWeekday\n")
    return df


#  3. TRAITEMENT LASTLOGINIP 

def process_last_login_ip(df):
    if 'LastLoginIP' not in df.columns:
        return df

    def is_private(ip):
        try: return int(ipaddress.ip_address(str(ip).strip()).is_private)
        except: return -1

    def first_octet(ip):
        try: return int(str(ip).strip().split('.')[0])
        except: return -1

    df['IsPrivateIP']  = df['LastLoginIP'].apply(is_private)
    df['IPFirstOctet'] = df['LastLoginIP'].apply(first_octet)
    df = df.drop(columns=['LastLoginIP'])
    print("[OK] LastLoginIP → IsPrivateIP + IPFirstOctet\n")
    return df


#  4. CORRECTION VALEURS ABERRANTES 

def fix_outliers(df):
    if 'SatisfactionScore' in df.columns:
        df['SatisfactionScore'] = df['SatisfactionScore'].replace(-1, np.nan)
        df['SatisfactionScore'] = df['SatisfactionScore'].replace(99, np.nan)
        df.loc[~df['SatisfactionScore'].between(0, 5), 'SatisfactionScore'] = np.nan
        print("  [FIX] SatisfactionScore : -1 et 99 → NaN")

    if 'SupportTicketsCount' in df.columns:
        df['SupportTicketsCount'] = df['SupportTicketsCount'].replace(-1, np.nan)
        df['SupportTicketsCount'] = df['SupportTicketsCount'].replace(999, np.nan)
        print("  [FIX] SupportTicketsCount : -1 et 999 → NaN")

    print("[OK] Valeurs aberrantes corrigees\n")
    return df


#  5. IMPUTATION VALEURS MANQUANTES 

def impute_missing_values(df):
    # Mediane pour colonnes simples
    for col in ['AvgDaysBetweenPurchases', 'SatisfactionScore', 'SupportTicketsCount']:
        if col in df.columns and df[col].isna().sum() > 0:
            med = df[col].median()
            n   = df[col].isna().sum()
            df[col] = df[col].fillna(med)
            print(f"  [MEDIAN] {col} : {n} NaN → {med:.2f}")

    # KNN pour Age (30% manquant)
    if 'Age' in df.columns and df['Age'].isna().sum() > 0:
        n = df['Age'].isna().sum()
        num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        if 'Churn' in num_cols: num_cols.remove('Churn')
        knn = KNNImputer(n_neighbors=5)
        df_imp = pd.DataFrame(knn.fit_transform(df[num_cols]),
                              columns=num_cols, index=df.index)
        df['Age'] = df_imp['Age'].round(1)
        print(f"  [KNN]    Age : {n} NaN → KNN k=5")

    print(f"[OK] Imputation terminee. NaN restants : {df.isnull().sum().sum()}\n")
    return df


#  6. FEATURE ENGINEERING 

def feature_engineering(df):
    if 'MonetaryTotal' in df.columns and 'Recency' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    if 'MonetaryTotal' in df.columns and 'Frequency' in df.columns:
        df['AvgBasketValue'] = df['MonetaryTotal'] / (df['Frequency'] + 1)
    if 'Recency' in df.columns and 'CustomerTenureDays' in df.columns:
        df['TenureRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)
    if 'CancelledTransactions' in df.columns and 'TotalTransactions' in df.columns:
        df['CancelRate'] = df['CancelledTransactions'] / (df['TotalTransactions'] + 1)
    print("[OK] Feature Engineering : MonetaryPerDay, AvgBasketValue, TenureRatio, CancelRate\n")
    return df


#  7. ENCODAGE CATEGORIEL 

def encode_categorical(df):
    # Ordinal
    ordinal_mappings = {
        'AgeCategory':       {'Inconnu':0,'18-24':1,'25-34':2,'35-44':3,'45-54':4,'55-64':5,'65+':6},
        'SpendingCategory':  {'Low':1,'Medium':2,'High':3,'VIP':4},
        'LoyaltyLevel':      {'Inconnu': 0, 'Nouveau': 1, 'Jeune': 2, 'Établi': 3, 'Etabli': 3, 'Ancien': 4},
        'ChurnRiskCategory': {'Faible': 1, 'Moyen': 2, 'Élevé': 3, 'Critique': 4, 'Eleve': 3, 'Elevé': 3},
        'BasketSizeCategory':{'Inconnu':0,'Petit':1,'Moyen':2,'Grand':3},
        'PreferredTimeOfDay':{'Nuit':0,'Matin':1,'Midi':2,'Apres-midi':3,'Soir':4},
        'RFMSegment':        {'Dormants':1,'Potentiels':2,'Fideles':3,'Champions':4},
    }
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # One-Hot
    onehot_cols = [c for c in ['CustomerType','FavoriteSeason','Region',
                                'WeekendPreference','ProductDiversity',
                                'Gender','AccountStatus'] if c in df.columns]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=False, dtype=int)

    # Target Encoding pour Country
    if 'Country' in df.columns and 'Churn' in df.columns:
        rate = df.groupby('Country')['Churn'].mean()
        df['Country_encoded'] = df['Country'].map(rate).fillna(df['Churn'].mean())
        df = df.drop(columns=['Country'])

    print(f"[OK] Encodage termine. Shape : {df.shape}\n")
    return df


#  8. SUPPRESSION MULTICOLLINEARITE 

def remove_multicollinearity(df, threshold=0.85):
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if 'Churn' in num_cols: num_cols.remove('Churn')
    corr  = df[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"[OK] {len(to_drop)} colonnes supprimees (corr > {threshold}) : {to_drop}\n")
    return df


#  9. NORMALISATION 

def normalize_features(X_train, X_test, save_scaler_path=None):
    """
    REGLE : fit_transform sur X_train UNIQUEMENT
             transform   sur X_test  (evite le data leakage)
    """
    num_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
    scaler   = StandardScaler()
    X_train  = X_train.copy()
    X_test   = X_test.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])
    if save_scaler_path:
        os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)
        joblib.dump(scaler, save_scaler_path)
    print(f"[OK] Normalisation terminee ({len(num_cols)} features)\n")
    return X_train, X_test, scaler


#  10. SPLIT TRAIN/TEST 

def split_and_save(df, target_col='Churn', test_size=0.2,
                   random_state=42, output_dir='../data/train_test/'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f'{output_dir}X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}X_test.csv',   index=False)
    y_train.to_csv(f'{output_dir}y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}y_test.csv',   index=False)
    print(f"[OK] Split 80/20 : X_train {X_train.shape} | X_test {X_test.shape}\n")
    return X_train, X_test, y_train, y_test


#  PIPELINE PRINCIPAL 

def run_preprocessing(input_path,
                      processed_path='../data/processed/data_clean.csv',
                      train_test_dir='../data/train_test/',
                      scaler_path='../models/scaler.joblib'):
    print("\n" + "="*55)
    print("  PREPROCESSING - DEBUT")
    print("="*55 + "\n")

    df = load_data(input_path)
    df = drop_useless_features(df)
    df = parse_registration_date(df)
    df = process_last_login_ip(df)
    df = fix_outliers(df)
    df = impute_missing_values(df)
    df = feature_engineering(df)
    df = encode_categorical(df)
    df = remove_multicollinearity(df, threshold=0.85)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Donnees nettoyees → {processed_path} | Shape : {df.shape}\n")

    X_train, X_test, y_train, y_test = split_and_save(
        df, target_col='Churn', test_size=0.2,
        random_state=42, output_dir=train_test_dir
    )
    X_train, X_test, scaler = normalize_features(
        X_train, X_test, save_scaler_path=scaler_path
    )

    print("="*55)
    print("  PREPROCESSING - TERMINE [OK]")
    print("="*55)
    return X_train, X_test, y_train, y_test, scaler


#  EXECUTION DIRECTE 

if __name__ == '__main__':
    import os
    
    # Chemin absolu basé sur l'emplacement du script
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    X_train, X_test, y_train, y_test, scaler = run_preprocessing(
        input_path     = os.path.join(BASE_DIR, 'data', 'raw', 'retail_customers_COMPLETE_CATEGORICAL.csv'),
        processed_path = os.path.join(BASE_DIR, 'data', 'processed', 'data_clean.csv'),
        train_test_dir = os.path.join(BASE_DIR, 'data', 'train_test') + os.sep,
        scaler_path    = os.path.join(BASE_DIR, 'models', 'scaler.joblib')
    )
    print(f"X_train : {X_train.shape} | X_test : {X_test.shape}")