import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ----- 1. Деректерді генерациялау -----
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'жас': np.random.randint(18, 65, n),
    'айлық_табыс': np.random.randint(80000, 1000000, n),
    'жұмыс_өтілі': np.random.randint(0, 30, n),
    'несие_сомасы': np.random.randint(50000, 2000000, n),
    'несие_саны': np.random.randint(0, 5, n),
    'төлем_тарихы': np.random.choice([0, 1], n, p=[0.3, 0.7]),
    'үй_иесі': np.random.choice([0, 1], n, p=[0.6, 0.4])
})

# ----- 2. Шынайы шулы нысаналы мән -----
data['төлем_қабілетті'] = (
    (data['айлық_табыс'] / (data['несие_сомасы'] + 1) > 0.25) &
    (data['төлем_тарихы'] == 1) &
    (data['жұмыс_өтілі'] > 2)
).astype(int)

# ----- Деректер шынайыға ұқсау үшін деректерге 10% шу қосамыз -----
flip = np.random.choice([0, 1], n, p=[0.9, 0.1])
data['төлем_қабілетті'] = np.abs(data['төлем_қабілетті'] - flip)

# ----- 3. Бөлу -----
X = data.drop('төлем_қабілетті', axis=1)
y = data['төлем_қабілетті']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----- 4. Масштабтау-----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- 5. Модельдер -----
models = {
    'Логистикалық регрессия': LogisticRegression(),
    'Кездейсоқ орман': RandomForestClassifier(n_estimators=100, random_state=42),
    'Градиенттік бустинг': GradientBoostingClassifier(random_state=42)
}

# ----- 6. Бағалау -----
results = []

plt.figure(figsize=(8, 6))
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append([name, acc, prec, rec, f1, auc])

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

# ----- 7. Кесте -----
results_df = pd.DataFrame(results, columns=['Модель', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
print(results_df)

# ----- 8. ROC-график -----
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Модельдердің ROC-қисықтары')
plt.legend()
plt.grid()
plt.show()