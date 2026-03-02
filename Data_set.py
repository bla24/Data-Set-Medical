# import des librairies pour notre projet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import os
import openpyxl

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Déclaration des variables
# base contenant les variables suivantes
# -	Variables de 1 à 40
# -	PATHOLOGIE : 0 : Absence / 1 : Présence

noms_variables = ["V1","V2","V3","V4","sexe","degre_rougeur_nez"]

# Nom du fichier Excel à ouvrir
# Lire le fichier Excel (première feuille par défaut)
fichier_excel = "dataset_maladie_40vars.xlsx"
df = pd.read_excel(fichier_excel, engine="openpyxl")

# Afficher les lignes
print("Aperçu du fichier Excel :")
print(df.head())


# Diviser le dataset en ensembles d'entraînement et de test:
# 80% de train set et 20% de test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)