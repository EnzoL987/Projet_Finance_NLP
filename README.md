# 📈 Analyseur de Signaux Mixtes (Finance & NLP)

## 🎯 Contexte et Objectifs
Ce projet est un pipeline de Machine Learning de bout-en-bout (Data Engineering & Data Science) conçu pour prédire la tendance à court terme d'un actif financier (hausse ou baisse à un horizon de 5 jours). 

L'innovation de ce projet repose sur l'hybridation des données : il combine des séries temporelles classiques (historiques de prix, indicateurs techniques) avec le Traitement du Langage Naturel (NLP) appliqué à l'actualité financière quotidienne.

## 🏗️ Architecture du Pipeline (ETL)

Le projet est structuré selon les meilleures pratiques d'ingénierie logicielle avec des scripts modulaires :

1. **Extract (`Src/extract.py`) :** 
   - Récupération des prix via l'API **yfinance**.
   - Collecte des articles de presse via **NewsAPI**.
2. **Transform (`Src/transform.py`) :** 
   - Analyse de sentiment NLP via **FinBERT** (ProsusAI/finbert).
   - Feature Engineering : Moyennes mobiles (SMA, EMA), Volatilité, RSI.
3. **Load (`Src/load.py`) :** 
   - Stockage intermédiaire dans une base de données relationnelle **SQLite** structurée en schéma en étoile (`dim_assets`, `fact_market_data`, `fact_news_sentiment`).
4. **Train (`Src/train.py`) :** 
   - Entraînement et comparaison de modèles de classification binaire (**XGBoost** vs **Random Forest**).

## 🚀 Installation et Exécution

### Prérequis
Assurez-vous d'avoir Python 3.9+ installé. Clonez ce dépôt, puis installez les dépendances :

il vous faudra aussi lancez ça : pip install -r requirements.txt

Lancer le Pipeline Automatisé

Un script chef d'orchestre permet de lancer la chaîne complète d'extraction, de transformation, de chargement en base SQL et de modélisation prédictive avec une seule commande dans le terminal : python pipeline_run.py

## 📊 Résultats et Modélisation

Le modèle évalue la performance du NLP couplé à la finance en utilisant la classification binaire. Le projet inclut l'extraction de l'importance des variables (Feature Importance) afin de quantifier la valeur ajoutée du sentiment des actualités face aux indicateurs de prix purs.
