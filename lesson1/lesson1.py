import sys

# data preprocessing
import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split

# logistic regression
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from pycm import ConfusionMatrix
import cv2

# random forest
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# xgboost
from xgboost import XGBClassifier
# import shap

print(sys.version)

data = pd.read_csv('pred_maintenance.csv', index_col="UDI")
data

data.dtypes

# categorical columns
data.pop('Product ID')
product_type = data.pop('Type')

# failure type columns
data.pop('TWF')
data.pop('HDF')
data.pop('PWF')
data.pop('OSF')
data.pop('RNF')

print("Done")

oh_type = pd.get_dummies(product_type)

print("Done")

processed_data = pd.concat([data, oh_type], axis=1)
processed_data

processed_data.dropna(inplace=True)
processed_data = processed_data.loc[:,~processed_data.columns.duplicated()]
processed_data

test_columns = ["Process temperature [K]", "Torque [Nm]", "Tool wear [min]", "Rotational speed [rpm]"]

for feature_name in test_columns:
  plt.figure()
  processed_data.groupby("Machine failure")[feature_name].plot.hist(alpha=0.5, density=True, legend=True)
  plt.xlabel(feature_name)

  # feel free to play with different features to use for X, ie tool wear, air temp, etc
x_single = processed_data['Process temperature [K]']
y_single = processed_data['Machine failure']

# some of this code comes from https://pythonspot.com/linear-regression/

x_single = x_single.values.reshape(len(x_single),1)
y_single = y_single.values.reshape(len(y_single),1)

# Split the data into training/testing sets
x_single_train = x_single[:-500]
x_single_test = x_single[-500:]

# Split the targets into training/testing sets
y_single_train = y_single[:-500]
y_single_test = y_single[-500:]

# Create logistic regression object
temp_model = LogisticRegression()

# Train the model using the training sets
temp_model.fit(x_single_train, y_single_train)
accuracy = temp_model.score(x_single_test, y_single_test)
print(f"Accuracy on test set: {accuracy:.3f}")

# feel free to play with different features to use for X, ie tool wear, air temp, etc
x_single = processed_data['Torque [Nm]']

# some of this code comes from https://pythonspot.com/linear-regression/
x_single = x_single.values.reshape(len(x_single),1)

# Split the data into training/testing sets
x_single_train = x_single[:-500]
x_single_test = x_single[-500:]

# Create logistic regression object
torque_model = LogisticRegression()

# Train the model using the training sets
torque_model.fit(x_single_train, y_single_train)
accuracy = torque_model.score(x_single_test, y_single_test)
print(f"Accuracy on test set: {accuracy:.3f}")

processed_data["Machine failure"].value_counts()

predictions = torque_model.predict(x_single_test)
clean_y_single_test = []

for i in range(len(y_single_test)):
  clean_y_single_test.append(str(y_single_test[i]).replace("[", "").replace("]", ""))

cm1 = ConfusionMatrix(clean_y_single_test, predictions)

cm1.plot(cmap=plt.cm.Blues, number_label=True)

# feel free to play with different features to use for X, ie perimeter_mean, compactness_mean, etc
x_single = processed_data['Torque [Nm]']

# some of this code comes from https://pythonspot.com/linear-regression/

x_single = x_single.values.reshape(len(x_single),1)

# Split the data into training/testing sets
x_single_train = x_single[:-500]
x_single_test = x_single[-500:]

# Create logistic regression object
torque_model_balanced = LogisticRegression(class_weight='balanced')

# Train the model using the training sets
torque_model_balanced.fit(x_single_train, y_single_train)
accuracy = torque_model_balanced.score(x_single_test, y_single_test)
print(f"Accuracy on test set: {accuracy:.3f}")

# format data for CM
clean_y_single_test2 = []

for i in range(len(y_single_test)):
  clean_y_single_test2.append(str(y_single_test[i]).replace("[", "").replace("]", ""))

# Confusion matrix
predictions = torque_model_balanced.predict(x_single_test)

cm2 = ConfusionMatrix(clean_y_single_test2, predictions)

cm2.plot(cmap=plt.cm.Blues, number_label=True)

x_tree = processed_data.drop('Machine failure', axis=1)
y_tree = processed_data['Machine failure']

x_train_tree, x_test_tree, y_train_tree, y_test_tree = train_test_split(x_tree, y_tree, test_size=0.2)

# create a classifier object
tree_model = DecisionTreeClassifier(class_weight='balanced')

# fit the classifier with X and Y data
tree_model.fit(x_train_tree, y_train_tree)

accuracy = tree_model.score(x_test_tree, y_test_tree)

print(f"Accuracy on test set: {accuracy:.3f}")

# Confusion matrix
predictions = tree_model.predict(x_test_tree)

cm3 = ConfusionMatrix(y_test_tree.tolist(), predictions)

cm3.plot(cmap=plt.cm.Blues, number_label=True)

x_forest = processed_data.drop('Machine failure', axis=1)
y_forest = processed_data['Machine failure']

x_train_forest, x_test_forest, y_train_forest, y_test_forest = train_test_split(x_forest, y_forest, test_size=0.2)

forest_model = RandomForestClassifier(class_weight='balanced')
forest_model.fit(x_train_forest, y_train_forest)

accuracy = forest_model.score(x_test_forest, y_test_forest)

print(f"Accuracy on test set: {accuracy:.3f}")

# Confusion matrix
predictions = forest_model.predict(x_test_tree)

cm4 = ConfusionMatrix(y_test_forest.tolist(), predictions)

cm4.plot(cmap=plt.cm.Blues, number_label=True)

# XGBoost doesn't like [] in the feature names so we need to change the names of our columns
processed_data = processed_data.rename(columns={"Air temperature [K]": "Air temperature",
                                                "Process temperature [K]": "Process temperature",
                                                "Rotational speed [rpm]": "Rotational speed",
                                                "Torque [Nm]": "Torque",
                                                "Tool wear [min]": "Tool wear"})

x_boost = processed_data.drop('Machine failure', axis=1)
y_boost = processed_data['Machine failure']

x_train_boost, x_test_boost, y_train_boost, y_test_boost = train_test_split(x_boost, y_boost, test_size=0.2)

boost = XGBClassifier()
boost.fit(x_train_boost, y_train_boost)

accuracy = boost.score(x_test_boost, y_test_boost)
print(f"Accuracy on test set: {accuracy:.3f}")

# Confusion matrix
predictions = boost.predict(x_test_boost)

cm5 = ConfusionMatrix(y_test_boost.tolist(), predictions)

cm5.plot(cmap=plt.cm.Blues, number_label=True)

