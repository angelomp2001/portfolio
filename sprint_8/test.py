import pandas as pd
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# Create the toy dataset
data = {
    'Total Area (m²)': [30, 50, 45],
    'Bedrooms': [1, 2, 2],
    'Bought House': [0, 1, 1]
}

df = pd.DataFrame(data)

# Define features and target
features = df[['Total Area (m²)', 'Bedrooms']]  # Use all features for training
target = df['Bought House']  # Use target variable

# Initialize the Decision Tree Classifier
model = DecisionTreeClassifier(random_state=0)  # Optional: specify random state for reproducibility

# Fit the model
model.fit(features, target)

# Plot the decision tree
#plt.figure(figsize=(10, 6))  # Optional: set the size of the plot
plot_tree(model, feature_names=features.columns, class_names=['Not Bought', 'Bought'], filled=True)
#plt.title('Decision Tree Visualization')
plt.show()