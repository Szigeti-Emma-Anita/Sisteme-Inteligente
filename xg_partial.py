import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

df = pd.read_csv('mxmh_cleaned.csv')

def categorii_scor(scor):
    if scor <= 2: return 0 
    elif scor <= 6: return 1
    else: return 2
df['Target'] = df['Depression'].apply(categorii_scor)

# matricea pearson
# extragem lista de coloane unice pentru genuri si ne asiguram ca nu avem duplicate
genuri_gasite = []
for x in ['Rock', 'Metal', 'Lofi', 'Pop', 'Classical', 'Hip_hop']:
    for col in df.columns:
        if x.lower() in col.lower() and col not in genuri_gasite:
            genuri_gasite.append(col)

factori_baza = ['Age', 'Hours per day', 'BPM', 'Depression']
cols_pearson = factori_baza + genuri_gasite

# cream dataframe-ul pentru corelatie si eliminam coloanele identice
df_corr = df[cols_pearson].apply(pd.to_numeric, errors='coerce').fillna(0)
df_corr = df_corr.loc[:,~df_corr.columns.duplicated()]

plt.figure(figsize=(12, 8))
correlation_matrix = df_corr.corr(method='pearson')

# heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Matricea Pearson: Corelații Unice Genuri vs. Depresie')
plt.tight_layout()
plt.show()

# pregatim datele pentru model
music_features = ['Age', 'Hours per day', 'BPM', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages']
genre_features = [col for col in df.columns if 'Frequency' in col]

X = df[music_features + genre_features].copy()
X.columns = [col.replace('[', '').replace(']', '').replace(' ', '_') for col in X.columns]
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['Target']

# SMOTE si split
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.15, random_state=42)

# XGBOOST
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"\n ACURATEȚE: {acc:.2f} ({acc*100:.1f}%)")

# importanta factorilor
plt.figure(figsize=(10, 8))
pd.Series(model.feature_importances_, index=X.columns).nlargest(15).plot(kind='barh', color='green')
plt.title('Top 15 Factori Muzicali Determinanți')
plt.show()