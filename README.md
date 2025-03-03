# Depressive Disorder Prediction
The project involves analyzing the Behavioral Risk Factor Surveillance System (BRFSS) 2020 dataset to predict whether individuals have a depressive disorder. The dataset contains over 5000 records with a mix of categorical and numerical features, capturing health behaviors, conditions, and demographics of U.S. adults.

The project begins with a thorough data preprocessing phase, which includes cleaning the dataset by removing irrelevant, redundant, and low-variance features. Missing values in both numerical and categorical attributes were handled using statistical techniques, such as mean or median for numerical values and mode for categorical ones. The dataset, initially imbalanced, was balanced using Synthetic Minority Over-sampling Technique (SMOTE) and Both Sampling techniques to ensure that the model performs well across all classes.

After preprocessing, feature selection methods such as Information Gain, Stepwise Feature Selection, and Recursive Feature Elimination were applied to reduce the feature set to the most relevant attributes for the model. The project then focused on building multiple machine learning models, including Logistic Regression, Random Forest, Gradient Boosting Machine (GBM), Linear Discriminant Analysis (LDA), and k-Nearest Neighbors (k-NN). Each model was tuned and evaluated for its performance using metrics such as precision, recall, F-measure, and ROC curves.

GBM emerged as the best-performing model, especially when using BOTH sampling techniques for data balancing and Stepwise Feature Selection. The project also met the extra credit criteria by achieving high accuracy in classification. The result is a highly optimized and effective model that predicts depressive disorders based on health data, with a focus on maximizing both predictive accuracy and model interpretability.

## Step by Step Walkthrough

### 1. Data Preprocessing

The dataset underwent multiple preprocessing steps to improve efficiency. First, hidden attributes (useless or inaccessible) were removed, followed by columns with 100% missing values. Next, attributes with zero and near-zero variance were eliminated to avoid redundant or uninformative features. A missing value threshold was applied to retain only relevant attributes, and manual inspection helped further refine the dataset. Finally, after all preprocessing, 101 attributes remained, with 5 numeric and the rest categorical.

### 2. Train-Test Split

![image](https://github.com/user-attachments/assets/fa76d7c6-9bc5-40f5-a28f-da5215dceb36)

Our pre-processed dataset now goes through a process called “Split Train-Test.” In this process, we split our pre-processed data into training and testing datasets. We use a split ration of 70/30, 70% of dataset is used for training the models and 30% of it used to test the model.  

### 3. Data Balancing

We applied a lot of Data Balancing techniques, but we will discuss just one, which gave us the most optimal performance, which was Both Sampling

![image](https://github.com/user-attachments/assets/a7ff1eaf-f345-4e55-b308-3058108f9501)

Both sampling encompasses strategies that utilize a combination of oversampling and undersampling techniques to address the challenges presented by imbalanced datasets in classification problems. This approach aims to create a more balanced distribution between classes, which is crucial for improving the performance and generalization ability of machine learning models. 

### 4. Feature Selection

Similar to Data Balancing, we applied a lot of techniques which to select the most ideal/optimal set of feature for model development, like Information Gain, Recursive Feature Elimination and Stepwise Feature Selection. But let's just discuss about the Stepwise Feature Selection.

![image](https://github.com/user-attachments/assets/4d5f74aa-c496-4bb9-9267-2605c8c040cd)

Stepwise feature selection is a method used in machine learning and statistics to select the most relevant features (or predictors) for use in model construction. It is a way of improving model performance and interpretability by including only those features that have a significant impact on the model's prediction accuracy, thereby reducing complexity and preventing overfitting. The process involves adding or removing predictors based on certain criteria, step by step until a certain stopping condition is met.

### 5. Model Development

After all of the steps, we decided on several classifications model, and use a mix match of each of the techniques, which we have developed and in that way we had 36 Combinations of Models, whihc used several feature selectiona nd data balancing techniques. Out of the 36 Combinations, we were able to figure out the best set of methods and model, which worked for us, and it was Gradient Boosting Machine or GBM. 

Gradient Boosting Machine (GBM) is a powerful machine learning technique that builds on the concept of boosting to solve various types of problems, including regression, classification, and ranking. GBM is part of a broader family of ensemble methods that aim to improve prediction accuracy by combining the predictions of multiple simpler models, usually decision trees.

Here are the results, which GBM gave us, where we had used BOTH Sampling as Data Balancing Technique and Step Wise Feature Selection. The results were tested on testing data. 

![image](https://github.com/user-attachments/assets/1a7b1016-020e-4d3b-8283-47900e979a62)
![image](https://github.com/user-attachments/assets/6f6386e4-f852-4bb8-bc75-b3694cd13e78)

Also, GBM gave a pretty much a similar performance, when we use BOTH Sampling and Recursive Feature Elimination

![image](https://github.com/user-attachments/assets/9e08c806-45ce-4255-afe3-e380db4daf35)
![image](https://github.com/user-attachments/assets/404eb7bf-b851-4a69-b543-b14da3b58278)

## Conclusion

Following preprocessing, data balancing and feature selection were applied to improve model performance. During model building and testing, we achieved an optimal True Positive Rate (TPR) for both classes. The Gradient Boosting Machine (GBM) model proved to be the most effective, leveraging both sampling techniques and feature selection methods such as Stepwise Selection and Recursive Feature Elimination (RFE).








