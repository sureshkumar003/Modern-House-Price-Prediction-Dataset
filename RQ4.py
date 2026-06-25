import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

df = pd.read_csv("house prediction.xlsx - house prediction.csv", skiprows=5)
X = df[['garage', 'parking', 'garden', 'security']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Ridge(alpha=1.0).fit(X_train, y_train)
print(f"Amenities Ridge Model R2: {r2_score(y_test, model.predict(X_test)):.4f}")