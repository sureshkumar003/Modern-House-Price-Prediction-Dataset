import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("house prediction.xlsx - house prediction.csv", skiprows=5)
df['income_encoded'] = df['income_level'].map({'low':0, 'mid':1, 'high':2})
X = df[['crime_rate', 'population_density', 'income_encoded']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor(max_depth=5).fit(X_train, y_train)
print(f"Socio-Economic Tree R2: {r2_score(y_test, model.predict(X_test)):.4f}")