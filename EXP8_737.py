import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# --- 1. Load the Data ---
# The Iris dataset has 4 features (petal/sepal length/width)
# and 3 target classes (species of iris)
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# --- 2. Split Data into Training and Testing Sets ---
# This lets us train the model on one set and test its performance on unseen data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 3. Create and Train the CART (Decision Tree) Model ---
# We create a DecisionTreeClassifier object
# 'random_state=42' ensures we get the same tree every time we run the code
cart_model = DecisionTreeClassifier(random_state=42)

# 'fit' is the training step. The model learns the rules from the training data.
cart_model.fit(X_train, y_train)

# --- 4. Make Predictions on the Test Data ---
y_pred = cart_model.predict(X_test)

# --- 5. Evaluate the Model ---
accuracy = accuracy_score(y_test, y_pred)
print(f"--- Model Evaluation ---")
print(f"Accuracy on test data: {accuracy * 100:.2f}%")
print("This shows how well the model 'categorized' the unseen test data.")

# --- 6. Visualize the Decision Tree ---
print("\n--- Visualizing the Learned Tree ---")
print("Displaying the decision tree rules...")

plt.figure(figsize=(20, 10))
# plot_tree creates a visual representation of the model's rules
plot_tree(cart_model,
          feature_names=feature_names,
          class_names=target_names,
          filled=True,   # Fills nodes with colors based on majority class
          rounded=True,  # Uses rounded boxes
          fontsize=10)

plt.title("Decision Tree Trained on Iris Data", fontsize=20)
plt.show()