# Heart Disease Prediction Case Study - Decision Tree & Random Forest Models

## Table of Contents
* [Random Forest Overview](#random-forest-overview)
* [Problem Statement](#problem-statement)
* [Technologies Used](#technologies-used)
* [Approach for Modeling](#approach-for-modeling)
* [Classification Outcome](#classification-outcome)
* [Conclusion](#conclusion)
* [Acknowledgements](#acknowledgements)

## Random Forest Overview

### What is a Decision Tree?
A **Decision Tree** is a tree-structured algorithm used for classification and regression tasks. It splits the data into subsets based on feature values, creating decision nodes and leaf nodes. Each split is made to maximize the homogeneity (or purity) of the resulting subsets. The final output is a path from the root to a leaf, which represents the prediction for a given input. Decision Trees are simple to interpret and work well with small datasets but are prone to overfitting.

### What is a Random Forest?  
A **Random Forest** is an ensemble learning method that builds multiple Decision Trees (called estimators) and combines their predictions to achieve higher accuracy and robustness. Each tree is trained on a random subset of the training data using the bootstrap aggregation (bagging) technique. Additionally, random subsets of features are used at each split to reduce correlation among trees, improving generalization. Random Forest models are highly flexible, work well with large datasets, and are less prone to overfitting compared to individual Decision Trees.

---

### Purity Measurement Algorithms

Purity in Decision Trees is determined by splitting the data to maximize the homogeneity of subsets. Common purity measurement algorithms include:

1. **Gini Impurity**:  
   Gini measures the probability of incorrectly classifying a randomly chosen instance from the dataset if it were labeled based on the distribution of labels in the node.  
   $$ Gini = 1 - \sum_{i=1}^n p_i^2 $$  
   where $p_i$ is the proportion of instances belonging to class $i$ in the node.

2. **Entropy and Information Gain**:  
   Entropy quantifies the uncertainty in the data. Information Gain measures the reduction in entropy achieved after a split.  
   $$ Entropy = -\sum_{i=1}^n p_i \log_2(p_i) $$  
   $$ Information\ Gain = Entropy_{parent} - \sum_{children} \frac{N_{child}}{N_{parent}} \cdot Entropy_{child} $$  
   where $p_i$ is the class proportion, and $N$ represents the number of samples.

3. **Mean Squared Error (MSE)**:  
   Used for regression problems to minimize variance in the splits.

---

### Hyperparameter Tuning

Tuning hyperparameters is critical to achieving the best model performance. Key hyperparameters for Decision Tree and Random Forest include:

- **For Decision Tree**:
  - `max_depth`: Maximum depth of the tree to prevent overfitting.
  - `min_samples_split`: Minimum samples required to split a node.
  - `min_samples_leaf`: Minimum samples required in a leaf node.
  - `criterion`: The metric used to measure purity (e.g., "gini" or "entropy").

- **For Random Forest**:
  - `n_estimators`: Number of trees in the forest.
  - `max_depth`, `min_samples_split`, `min_samples_leaf`: Same as for Decision Trees.
  - `max_features`: Number of features to consider for each split.
  - `bootstrap`: Whether to use bootstrapping for training subsets.

Hyperparameter tuning can be automated using techniques like Grid Search or Random Search.

---

By combining **Decision Trees** and **Random Forest** models with careful hyperparameter tuning and robust evaluation, powerful predictive models can be developed, as demonstrated in the Heart Disease prediction exercise. The approach described here represent the practical process utilised in industry to predict categorical target parameters for business.


## Problem Statement

Heart disease is one of the leading causes of mortality worldwide, and early prediction of heart disease can significantly aid in preventive healthcare and timely medical intervention.

The goal of this exercise is to build predictive models using **Decision Trees and Random Forest** algorithms to classify whether a person is likely to have heart disease based on their medical and demographic attributes. The dataset includes features such as `age, sex, blood pressure (BP) and cholesterol levels` which are key health-related parameters that are potential indicators of heart disease. These features contain complex relationships that may influence the likelihood of heart disease. The task involves analyzing these factors to develop a robust and accurate prediction system.

**Random Forest** is an appropriate model for this task because it combines multiple decision trees, reducing the risk of overfitting, which is a common issue in single decision trees. By leveraging its ability to provide feature importance, Random Forest also helps in understanding which medical or demographic factors are most significant in predicting heart disease.

## Technologies Used

Python Jypyter Notebook in local PC environment has been used for the exercise. Apart from ususal Python libraries like  Numpy and Pandas, there are machine Learning specific libraries used to prepare, analyse, build model and visualise data. The following model specific libraries have been used in the case study:

- sklearn
- six
- pydotplus
- graphviz


## Approach for Modeling

The following steps are followed to build the Random Forest model for Heart Disease analysis:

1. Import & Prepare Data for Modeling
2. Build & Evaluate Decision Tree Models
3. Build & Evaluate Random Forest Models
4. Assess Feature Importance
5. Conclusion

Some distinguishing processes in this approach include,

- Tuning of model hyper-parameters using `GridSearchCV` method by feeding the hyper-parameters through params grid.

- Visualisation of Decision Trees generated by the models using `export_graphviz` method and `pydotplus` library

- Determination of most important feature variables through feature importance parameter of Random Forest model


## Classification Outcome

In this exercise, a Random Forest model with 30 estimators and hyperparameters such as max_depth=10, max_features=3, and min_samples_leaf=5 was developed to predict the likelihood of heart disease. The final model achieved a **test accuracy score of 0.83**, indicating good predictive performance with no signs of overfitting. This suggests that the model is both robust and capable of generalizing well to unseen data. 

Furthermore, feature importance analysis revealed that **Age** and **Cholesterol** were the most significant predictors of heart disease, providing valuable insights into the factors that contribute to its occurrence. These findings highlight the practical utility of machine learning models like Random Forest in identifying critical health indicators, aiding healthcare professionals in making data-driven decisions for early diagnosis and preventive care.


## Conclusion

Random Forest is a powerful and versatile machine learning algorithm that excels in both **classification** and **regression** problems. By constructing multiple decision trees and aggregating their outputs, it effectively reduces overfitting and improves model generalization. Its ability to handle high-dimensional datasets, work with both categorical and numerical features, and provide insights into feature importance makes it an ideal choice for complex real-world problems.


## Acknowledgements

This case study has been developed as part of Post Graduate Diploma Program on Machine Learning and AI, offered jointly by Indian Institute of Information Technology, Bangalore (IIIT-B) and upGrad.