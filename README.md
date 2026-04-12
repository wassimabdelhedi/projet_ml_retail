# ML Retail : Analyse du Comportement Client

Une pipeline de machine learning complète conçue pour analyser et prédire le comportement des clients dans un contexte de commerce de détail et de e-commerce. Ce projet implémente un flux de travail professionnel, de la transformation des données brutes à une application web interactive.

## Table des Matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités du Projet](#fonctionnalités-du-projet)
- [Architecture Technique](#architecture-technique)
- [Structure du Répertoire](#structure-du-répertoire)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Méthodologie](#méthodologie)
- [Métriques de Performance](#métriques-de-performance)
- [Améliorations Futures](#améliorations-futures)

---

## Vue d'ensemble

L'objectif principal de ce projet est de fournir des informations stratégiques sur le comportement des clients. En utilisant des données historiques, le système génère des prédictions précises pour optimiser la rétention client et les performances commerciales.

## Fonctionnalités du Projet

Le projet repose sur trois piliers analytiques :

1. **Prédiction du Churn (Attrition)** : Identification des clients à risque de départ via un modèle XGBoost optimisé.
2. **Segmentation Client** : Regroupement automatique des clients en segments homogènes via K-Means pour un ciblage marketing précis.
3. **Prévision des Dépenses** : Estimation du montant futur des achats à l'aide d'une régression Random Forest.

---

## Architecture Technique

Le projet adopte une structure modulaire favorisant la maintenabilité :

- **Traitement des Données** : Nettoyage, ingénierie des caractéristiques et normalisation.
- **Modélisation** : Entraînement avec optimisation des hyperparamètres (GridSearchCV).
- **Inférence** : Module dédié pour les prédictions en temps réel.
- **Interface Utilisateur** : Application web Flask interactive avec visualisation des résultats.

### Stack Technique

- **Langage** : Python 3.8+
- **Machine Learning** : Scikit-learn, XGBoost
- **Analyse de Données** : Pandas, NumPy
- **Visualisation** : Matplotlib, Seaborn
- **Framework Web** : Flask

---

## Structure du Répertoire

```text
projet_ml_retail/
├── data/                          # Données (brutes, traitées et splits)
├── src/                           # Code source principal
│   ├── preprocessing.py          # Nettoyage et ingénierie des données
│   ├── train_model.py            # Entraînement des modèles ML
│   ├── predict.py                # Pipeline d'inférence
│   ├── exploration.py            # Scripts d'analyse exploratoire
│   └── utils.py                  # Fonctions utilitaires
├── models/                        # Modèles et scalers sauvegardés (.joblib)
├── app/                           # Application Flask (routes, templates, static)
├── reports/                       # Graphiques d'analyse et rapports VIF
├── notebooks/                     # Expérimentations et prototypage
├── requirements.txt               # Liste des dépendances
└── README.md                      # Documentation du projet
```

---

## Installation

### Prérequis

- Python 3.8+
- pip (gestionnaire de paquets)

### Étapes d'Installation

1. **Clonage du projet**
   ```bash
   git clone https://github.com/wassimabdelhedi/projet_ml_retail
   cd projet_ml_retail
   ```

2. **Configuration de l'environnement virtuel**
   ```bash
   python -m venv venv
   # Activation (Windows)
   venv\Scripts\activate
   # Activation (Linux/macOS)
   source venv/bin/activate
   ```

3. **Installation des bibliothèques**
   ```bash
   pip install -r requirements.txt
   ```

---

## Utilisation

### 1. Entraînement des Modèles
Pour traiter les données et entraîner l'ensemble des modèles de la pipeline :
```bash
python src/train_model.py
```

### 2. Lancement de l'Application Web
Pour démarrer l'interface interactive :
```bash
cd app
python app.py
```
L'interface sera accessible à l'adresse suivante : `http://localhost:5000`

---

## Méthodologie

### Analyse et Prétraitement
- Traitement rigoureux des valeurs manquantes et aberrantes.
- Feature engineering : création de variables temporelles et comportementales.
- Analyse de la colinéarité via le facteur d'inflation de la variance (VIF).

### Développement des Modèles
- **Clustering** : Détermination du nombre optimal de clusters par la méthode du coude.
- **Classification** : Optimisation du F1-Score pour compenser le déséquilibre des classes (churn).
- **Régression** : Optimisation du R-carré pour la précision des dépenses prévues.

---

## Métriques de Performance

| Modèle | Métrique | Performance |
|--------|----------|-------------|
| Classification (Churn) | F1-Score | 0,84 |
| Classification (Churn) | AUC-ROC | 0,91 |
| Régression (Dépenses) | R-carré | 0,72 |
| Clustering (Segments) | Silhouette | 0,48 |

---

## Améliorations Futures

- Intégration d'une base de données SQL pour l'historique des prédictions.
- Mise en place d'une pipeline CI/CD pour l'automatisation des déploiements.
- Développement d'un tableau de bord de monitoring des modèles en temps réel.
- Ajout d'une couche d'authentification sécurisée pour l'application.


