import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns # Used for easier plotting

# --- 1. Load the Data ---
# The Iris dataset has 4 features (dimensions) and 3 target classes
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# --- 2. Apply PCA ---
# We are reducing the 4 dimensions down to 2
# n_components=2 means we want to find the top 2 principal components
pca = PCA(n_components=2)
X_r = pca.fit_transform(X) # X_r is the new, reduced dataset

# --- 3. Analyze the Results (Optional) ---
print(f"Original shape (features): {X.shape[1]}")
print(f"Reduced shape (components): {X_r.shape[1]}")
print(f"Explained variance ratio (per component): {pca.explained_variance_ratio_}")
print(f"Total variance explained by 2 components: {sum(pca.explained_variance_ratio_) * 100:.2f}%")

# --- 4. Plot the 2D Results ---
# Put the results into a DataFrame for easy plotting with seaborn
df = pd.DataFrame(data=X_r, columns=['Principal Component 1', 'Principal Component 2'])
df['target'] = y
df['species'] = df['target'].map({0: target_names[0], 1: target_names[1], 2: target_names[2]})

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Principal Component 1', y='Principal Component 2', hue='species', style='species', s=100)
plt.title('PCA of Iris Dataset (4D -> 2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Species')
plt.grid()
plt.show()