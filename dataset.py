import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# incarcarea datelor
try:
    df = pd.read_csv('mxmh_survey_results.csv')
    print("Datasetul a fost încărcat cu succes.")
except FileNotFoundError:
    print("Eroare: Asigură-te că fișierul 'mxmh_survey_results.csv' este în același folder cu scriptul.")

# eliminarea coloanelor irelevante
# timestamp si permissions nu au valoare predictiva
df.drop(['Timestamp', 'Permissions'], axis=1, inplace=True, errors='ignore')

# curatarea datelor fara sens si lipsa
# eliminam varstele nerealiste (peste 100 ani) si completam valorile lipsa la varsta cu mediana
df = df[df['Age'] < 100]
df['Age'] = df['Age'].fillna(df['Age'].median())

# reparam coloana BPM: completam lipsurile si eliminam valorile exagerate (ex: erori de tastare 9999)
df['BPM'] = df['BPM'].fillna(df['BPM'].median())
df = df[df['BPM'] < 300]

# completam restul valorilor lipsa cu cea mai frecventa valoare de pe fiecare coloană
df = df.apply(lambda x: x.fillna(x.mode()[0]))

# transformarea datelor categorice (Encoding)

# Ordinal Encoding pentru frecventele muzicale (pastram ierarhia)
label_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Very frequently': 3}
freq_cols = [col for col in df.columns if 'Frequency' in col]
for col in freq_cols:
    df[col] = df[col].map(label_mapping)

# transformarea raspunsurilor binare (Yes/No) in 1 și 0
binary_cols = ['While working', 'Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# transformarea coloanei 'Music effects' (Scara de impact)
df['Music effects'] = df['Music effects'].map({'Improve': 1, 'No effect': 0, 'Worsen': -1})

# One-Hot Encoding pentru variabilele fara ordine (gen favorit, serviciu streaming)
df = pd.get_dummies(df, columns=['Fav genre', 'Primary streaming service'], prefix=['Genre', 'Stream'])

# normalizarea (Scalarea între 0 și 1)
# important pentru ca BPM sau varsta sa nu "domine" restul datelor
scaler = MinMaxScaler()
cols_to_scale = ['Age', 'Hours per day', 'BPM']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# salvarea datasetului final
df.to_csv('mxmh_cleaned.csv', index=False)

print("-" * 30)
print("Procesare finalizată!")
print(f"Număr de rânduri rămase: {df.shape[0]}")
print(f"Număr de coloane finale: {df.shape[1]}")
print("Fișierul 'mxmh_cleaned.csv' este gata pentru Machine Learning.")