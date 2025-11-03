from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import the models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# --- 1. Load the Data ---
# This dataset classifies tumors as malignant (0) or benign (1)
data = load_breast_cancer()
X = data.data
y = data.target

# --- 2. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 3. Model 1: Single Decision Tree (Baseline) ---
# This is the same type of model from experiment 8
print("Training Single Decision Tree...")
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
acc_tree = accuracy_score(y_test, y_pred_tree)

# --- 4. Model 2: Random Forest (Bagging Ensemble) ---
# n_estimators=100 means we are building an ensemble of 100 trees
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# --- 5. Model 3: Gradient Boosting (Boosting Ensemble) ---
# This model also builds 100 "weak" learners sequentially
print("Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)

# --- 6. Compare the Results ---
print("\n--- Model Accuracy Comparison ---")
print(f"  Single Decision Tree: {acc_tree * 100:.2f}%")
print(f"  Random Forest (Ensemble): {acc_rf * 100:.2f}%")
print(f"  Gradient Boosting (Ensemble): {acc_gb * 100:.2f}%")