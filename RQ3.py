import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv("house prediction.xlsx - house prediction.csv", skiprows=5)
X, y = df[['distance']], df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
model = LinearRegression().fit(poly.fit_transform(X_train), y_train)
print(f"Polynomial R2: {r2_score(y_test, model.predict(poly.transform(X_test))):.4f}")