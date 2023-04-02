import warnings
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import pickle

warnings.simplefilter("ignore")

df_comb = pd.read_csv("Dataset/dis_sym_dataset_comb.csv") # Disease combination
df_norm = pd.read_csv("Dataset/dis_sym_dataset_norm.csv") # Individual Disease

X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0:1]

dataset_symptoms = list(X.columns)

lr = LogisticRegression()
lr = lr.fit(X, Y)
scores = cross_val_score(lr, X, Y, cv=5)

pickle.dump(lr, open('flsk_api\model.pkl','wb'))