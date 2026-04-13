import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('mxmh_cleaned.csv')

# transformam scorul 0-10 în categorii
def categorii_scor(scor):
    if scor <= 3:
        return 0  # risc scazut (0-3)
    elif scor <= 6:
        return 1  # risc moderat (4-6)
    else:
        return 2  # risc ridicat (7-10)

# aplicampentru coloana "Depression"
df['Depression_Cat'] = df['Depression'].apply(categorii_scor)

# separam variabilele de intrare (X) de eticheta pe care vrem sa o prezicem (y)
# eliminam celelalte scoruri ca modelul să prezica depresia doar pe baza muzicii, nu pe baza altor factori
X = df.drop(['Anxiety', 'Depression', 'Insomnia', 'OCD', 'Depression_Cat'], axis=1)
y = df['Depression_Cat']

# impartim datele: 80% pentru antrenare și 20% pentru testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initializam modelul KNN
knn = KNeighborsClassifier(n_neighbors=3)

# antrenăm modelul
knn.fit(X_train, y_train)

# facem predictii pe datele de test (cele 20% pe care modelul nu le-a vazut inca)
y_pred = knn.predict(X_test)

# cream graficul pentru matricea de confuzie
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Prezis Scazut', 'Prezis Moderat', 'Prezis Ridicat'],
            yticklabels=['Real Scazut', 'Real Moderat', 'Real Ridicat'])

plt.xlabel('Predicția Modelului')
plt.ylabel('Realitatea (Adevărul)')
plt.title('Harta de Confuzie a Modelului')
plt.show()

print("REZULTATE MODEL KNN")

print("\nMatricea de Confuzie (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))

print("\nRaport de Clasificare:")
print(classification_report(y_test, y_pred, target_names=['Risc Scazut', 'Risc Moderat', 'Risc Ridicat']))

print(f"Date antrenare: {X_train.shape[0]} rânduri")
print(f"Date testare: {X_test.shape[0]} rânduri")