import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ML Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# ---------------- Load Dataset ----------------
csv_path = 'energy_dataset.csv'
df = pd.read_csv(csv_path)

# Separate target and drop appliance usage columns
y = df['energy_usage']
drop_cols = [col for col in df.columns if col.endswith('_usage')]
X = df.drop(columns=drop_cols + ['energy_usage'])

# ---------------- Feature Scaling ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- Split Dataset ----------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------- Define Models ----------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR()
}

best_model = None
best_r2 = -float('inf')
best_name = ""

# ---------------- Train & Evaluate ----------------
print("\n🔍 Model Evaluation:\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"{name}:\n  ✅ MSE: {mse:.2f}\n  ✅ R²: {r2:.4f}\n")

    if r2 > best_r2:
        best_model = model
        best_r2 = r2
        best_name = name

# ---------------- Save Best Model & Scaler ----------------
with open('energy_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"🏆 Best Model: {best_name} (R² = {best_r2:.4f}) saved as energy_model.pkl")
print("📦 Scaler saved as scaler.pkl")
