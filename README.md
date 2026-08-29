# Hybrid Stock Market Forecasting : Integrating NLP and Technical Analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-171515?style=for-the-badge&logo=xgboost&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000)

## 🎯 Project Background

This project implements an end-to-end machine learning pipeline designed to predict short-term stock market trends. Its innovation lies in data hybridisation, combining traditional technical price analysis (quantitative data) with sentiment analysis of financial news using artificial intelligence (textual data processed by FinBERT).

## 🏗️ Pipeline Architecture

*(Overview of the data structure and ETL flow)*

![Overall Diagram](Images/Schéma_global.png)

## 📂 Project Structure

The code is structured in accordance with data engineering standards, clearly separating the Extraction, Transformation and Loading (ETL) stages.

Here’s how to navigate the project:

```text
Projet_Finance_NLP/ 
├── Data/                   # Storage of generated artefacts, such as the final model (champion_model.pkl)
│
├── Images/                 # Saving analytical charts (ROC, matrices, sentiment, etc.)
│
├── Notebooks/              
│   ├── Data/                     # Historical synthetic data for exploration
│   │
│   ├── 01_exploration...         # Research and initial testing log where the functions have been created
│   │
│   └── 02_Rapport_de_Synthèse    # The final report containing all the analyses and visualisations
│
│
├── SQL/                    # Storage of the SQLite relational database (finance_nlp.db)
│
├── Src/                    # Source
│   ├── extract.py                # Data extraction functions via APIs (yfinance & NewsAPI)
│   │
│   ├── transform.py              # Feature engineering (calculation of RSI, SMA) and NLP (FinBERT)
│   │
│   ├── load.py                   # Creating the star schema and SQL insertion
│   │
│   └── train.py                  # Machine Learning Modelling (XGBoost and Random Forest)
│
├── .env                    # Environment variables file (not tracked by Git but required for production)
│
├── pipeline_run.py         # The executable file
│
└── requirements.txt        # List of the project’s Python dependencies
```

Why this architecture?

* Modularity (Src/): Each stage of data processing is isolated in a dedicated script, making the code clean, maintainable and testable.

* Separation of storage and code (SQL/ & Data/): Databases and resource-intensive models are isolated from the execution scripts to comply with deployment best practices[cite: 2].

* Automation (pipeline_run.py): This script imports the functions from Src/ to launch the entire process automatically.

## ⚙️ Installation and Execution

* Installing dependencies:

Ensure you have Python installed, then run the following command from the project root directory:

Bash

```text
pip install -r requirements.txt
```
* API configuration (optional):
The project uses the NewsAPI to download news articles. To run the extraction yourself, create a file called .env in the root directory and add your key to it:

Plaintext

```text
NEWS_API_KEY=votre_cle_api_ici
```

* Run the entire pipeline:
A single command is all it takes to trigger the ETL, the NLP analysis, the SQL insertion and the model training:

Bash

```text
python pipeline_run.py
```

## 📊 Results and Detailed Analysis

Does incorporating text analysis offer a genuine predictive advantage over a traditional quantitative model? Delve into the summary report to discover the answer through a detailed analysis, supported by performance metrics (Accuracy, F1-Score) and our interactive visualisations :

> 👉 [**Open the Summary Report (Notebook 02)**](Notebooks/02_Rapport_de_Synthese.ipynb)


##
