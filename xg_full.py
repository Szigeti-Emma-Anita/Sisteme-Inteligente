import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv('mxmh_cleaned.csv')

# categorii
def categorii_scor(scor):
    if scor <= 2: return 0
    elif scor <= 6: return 1
    else: return 2
df['Target'] = df['Depression'].apply(categorii_scor)

# matricea pearson
cols_for_corr = ['Age', 'Hours per day', 'BPM', 'Anxiety', 'Insomnia', 'OCD', 'Depression']

music_freq = [col for col in df.columns if 'Frequency' in col][:5]
corr_df = df[cols_for_corr + music_freq]

plt.figure(figsize=(12, 10))
correlation_matrix = corr_df.corr(method='pearson')

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap='RdBu_r',
    vmin=-1,
    vmax=1,
    center=0
)

plt.title('Matricea de Corelație Pearson')
plt.tight_layout()
plt.show()

# selectia variabilelor
top_features = ['Age', 'Hours per day', 'BPM', 'Anxiety', 'Insomnia', 'Music effects']
top_genres = [col for col in df.columns if any(x in col for x in ['Rock', 'Metal', 'Lofi'])]

X = df[top_features + top_genres]
X.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X.columns]
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['Target']

# SMOTE si split
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.15, random_state=42)

# XGBoost
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    gamma=0.2,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Acuratețe: {accuracy_score(y_test, y_pred):.2f}")