import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("house prediction.xlsx - house prediction.csv", skiprows=5)
X = df[['bedrooms', 'bathrooms', 'floors']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
print(f"Structural Model R2: {r2_score(y_test, model.predict(X_test)):.4f}")