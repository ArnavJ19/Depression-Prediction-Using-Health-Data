# Data Mining Term Project
# Team Memebers: Arnav Jain, Rithika Kothapalli
#-------------------------------------------------------------------------------------------------------------------------------

'The following steps are taken on preprocessed data. 

1. Split Test-Train Dataset
2. Data Balancing
3. Feature Selection
4. Model Building (Training)
5. Model Testing

Let us proceed with first modUle, that is Split Test-Train 
'

#-------------------------------------------------------------------------------------------------------------------------------

'Module 1: Split Test-Train'


data=read.csv('Preprocessed_Data.csv') # Loading dataset
cols.dont.want <- c("Unnamed..0","X") 
data <- data[, ! names(data) %in% cols.dont.want, drop = F]

library(rsample) # Library contains "initial_split" functiona
set.seed(123) # For reproducibility
split <- initial_split(data, prop = 0.7) # Using a 70/30 Split i.e. 70% Train, 30% Test
train_set <- training(split) # Train dataset, on which other modules will be exercised
test_set <- testing(split) # Test Dataset for evaluating model performance

write.csv(train_set, "initial_train.csv")
write.csv(test_set, "initial_test.csv")

#----------------------------------------------------------------------------------------------------------------------------------------------------

'Module 2: Data Balancing'

'As per the project guideleines, we are required to use 2 Data Balancing Techniques, we have selected the following

  1. SMOTE Sampling
  2. Both Oversampling
  
First let us look at the class proprtion of the dataset
'
barplot(prop.table(table(train_set$Class)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution of initial training set")

# Number of each class in dataset
print(table(train_set$Class))

'Now we will implement first data balancing technique, SMOTE Data Balancing'

#--------------------------------------------------------------------------------------------------

'Module 1 part A: SMOTE Data Balancing'

'There is a k value in SMOTE function, which needs to calculated, so we have written a script for
finding the optimal K value

Due to technicality, there were continous function call backs whenwe were integrating code for 
SMOTING, but we have added the code for it in file called "Smote.R". 
'

#-------------------------------------------------------------------------------------------------

'Module 1 part B: Both Sampling Data Balancing'

library(rsample)
library(ROSE)# Library necessary for Both Sampling

#Implementing both sampling data balancing
both <- ovun.sample(Class~., data=train_set, method = "both",
                    p = 0.5,
                    seed = 226,
                    N = 3500)$data

#Plotting the graph for checking the proportions of both classes
barplot(prop.table(table(both$Class)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distributionafter data balancing (BOTH)")

table(both$Class) #Printing the distribution of both classes
write.csv(both, "BOTH.csv")

'Now we will head towards the next module, Feature Selection'
#----------------------------------------------------------------------------------------------------------------------------------------------------

'Module 2: Feature Selection'

'Now we proceed to second module of our project, Feature Selection
As per the project guidelines, we are required to use 3 feature selection techniques On 2 Balanced Datasets, 
so we have implemented the following 3 techniques:

1. Stepwise Feature Selection 
2. Information Gain based Feature Selection
3. Recursive Feature Elimination

We implement first technique, Stepwise Feature Selection on both of the balanced datasets
'

#--------------------------------------------------------------------------------------------------

'Module 2, Part A, Subsection 1: Implementing Stepwise Feature Selection on SMOTE Dataset'

library(MASS) # Library Necessary for implementing the Stepwise Feature Selection

#Loading Dataset
SMOTE=read.csv("data_SMOTE.csv")
# Removing unnecessary column
cols.dont.want <- c("Unnamed..0","X") 
SMOTE <- SMOTE[, ! names(SMOTE) %in% cols.dont.want, drop = F]

# Factorizing the target variable 'Class'
SMOTE$Class <- as.factor(SMOTE$Class)

# Implementing stepwise feature selection
initial.model <- glm(Class ~ ., data=SMOTE, family=binomial())
forward.model <- stepAIC(initial.model, direction="both", scope=~., data=SMOTE, trace = 1, steps = 10000)
summary(forward.model)

shortlistedVars <- names(unlist(forward.model[[1]])) # Extracting the name of all variables
shortlistedVars <- shortlistedVars[!shortlistedVars %in% "(Intercept)"] # remove intercept

print(shortlistedVars)

# Filtered out dataset
SMOTE_STEPWISE=SMOTE[,c(shortlistedVars, "Class")]
write.csv(SMOTE_STEPWISE, "SMOTE_STEPWISE.csv")

#--------------------------------------------------------------------------------------------------

'Module 2, Part A, Subsection 2: Implementing Stepwise Feature Selection Both Sampled Dataset'

# Removing unwanted column
both <- both[, ! names(both) %in% cols.dont.want, drop = F]

# Factorizing the target variable 'Class'
both$Class <- as.factor(both$Class)

# Implementing stepwise feature selection
initial.model <- glm(Class ~ ., data=both, family=binomial())
forward.model <- stepAIC(initial.model, direction="both", scope=~., data=both, trace = 1, steps = 10000)
summary(forward.model)

shortlistedVars <- names(unlist(forward.model[[1]])) # Extracting the name of all variables
shortlistedVars <- shortlistedVars[!shortlistedVars %in% "(Intercept)"] # remove intercept

print(shortlistedVars)

# Filtered out dataset
BOTH_STEPWISE=both[,c(shortlistedVars, "Class")]
write.csv(BOTH_STEPWISE, "BOTH_STEPWISE.csv")

#--------------------------------------------------------------------------------------------------

'Module 2, Part B, Subsection 1: Implementing Information Gain-based Selection on SMOTE Dataset'

# Necessary Libraries
library(readr) 
library(infotheo)

SMOTE$Class <- as.factor(SMOTE$Class)

info_gain_values <- numeric(length = ncol(SMOTE) - 1)

# Calculate information gain for each feature
for (i in 1:(ncol(SMOTE) - 1)) {
  feature <- SMOTE[,i]
  
  # Calculate and store information gain
  info_gain_values[i] <- mutinformation(discretize(feature), discretize(SMOTE$Class))
}
# Create a named vector for easier interpretation
names(info_gain_values) <- colnames(SMOTE)[1:(ncol(SMOTE) - 1)]

# Sort features based on information gain in descending order
sorted_features <- sort(info_gain_values, decreasing = TRUE)

# Print the sorted features with their information gain values
print(sorted_features)

barplot(sorted_features, ylab = "Information Gain", xlab = "Features")
selected_features <- names(info_gain_values[info_gain_values > median(sorted_features)])

# Add the target variable to the list of selected features
selected_features <- c(selected_features, "Class")

# Create a new dataset with the selected features
SMOTE_IG <- SMOTE[, selected_features]
write.csv(SMOTE_IG, "SMOTE_IG.csv")
#--------------------------------------------------------------------------------------------------
'Module 2, Part B, Subsection 2: Implementing Information Gain-based Selection on both Dataset'

#Necessary Libraries
library(readr) 
library(infotheo)

both$Class <- as.factor(both$Class)

info_gain_values <- numeric(length = ncol(both) - 1)

# Calculate information gain for each feature
for (i in 1:(ncol(both) - 1)) {
  feature <- both[,i]
  
  # Calculate and store information gain
  info_gain_values[i] <- mutinformation(discretize(feature), discretize(both$Class))
}
# Create a named vector for easier interpretation
names(info_gain_values) <- colnames(both)[1:(ncol(both) - 1)]

# Sort features based on information gain in descending order
sorted_features <- sort(info_gain_values, decreasing = TRUE)

# Print the sorted features with their information gain values
print(sorted_features)

barplot(sorted_features, ylab = "Information Gain", xlab = "Features")
selected_features <- names(info_gain_values[info_gain_values > median(sorted_features)])

# Add the target variable to the list of selected features
selected_features <- c(selected_features, "Class")

# Create a new dataset with the selected features
BOTH_IG <- both[, selected_features]
write.csv(BOTH_IG, "BOTH_IG.csv")

#--------------------------------------------------------------------------------------------------
'Module 2, Part C, Subsection 1: Implementing Recursive Feature Elimination on SMOTE Dataset'

# Load necessary libraries
library(caret)
library(ranger) 

trainX <- SMOTE[, -ncol(SMOTE)]  # Exclude the target variable for training features
trainY <- SMOTE[, ncol(SMOTE)]   # Target variable

# RFE
control <- rfeControl(functions=rfFuncs, method="cv", number=10, verbose=1)

# Choose the number of features you aim to select
numFeatures <- round(ncol(trainX) / 2, 0)

set.seed(123) # For reproducibility
rfeResults <- rfe(trainX, trainY, sizes=c(numFeatures), rfeControl=control)
selectedFeatures <- predictors(rfeResults)
print(selectedFeatures) #printing the selected features
SMOTE_RFE <- SMOTE[, c(selectedFeatures, "Class")]
write.csv(SMOTE_RFE, "SMOTE_RFE.csv")
#--------------------------------------------------------------------------------------------------
'Module 2, Part C, Subsection 2: Implementing Recursive Feature Elimination on both Dataset'

# Load necessary libraries
library(caret)
library(ranger) 

trainX <- both[, -ncol(both)]  # Exclude the target variable for training features
trainY <- both[, ncol(both)]   # Target variable

# RFE
control <- rfeControl(functions=rfFuncs, method="cv", number=10, verbose = 1)

# Choose the number of features you aim to select
numFeatures <- round(ncol(trainX) / 2, 0)

set.seed(123) # For reproducibility
rfeResults <- rfe(trainX, trainY, sizes=c(numFeatures), rfeControl=control)
selectedFeatures <- predictors(rfeResults)
print(selectedFeatures) # printing the selected features
BOTH_RFE <- both[, c(selectedFeatures, "Class")]
write.csv(BOTH_RFE, "BOTH_RFE.csv")
#----------------------------------------------------------------------------------------------------------------------------------------------------
'Module 3: Model Building'
'As per the guidelines, we are required to build 6 models, in a total of 36 models (2x3x6). We have decided upon the following 6 models
  1. Logistic Regression
  2. LDA (Linear Discriminant Analysis)
  3. Decision Tree
  4. Naive Bayes Classifier
  5. GBM (Gradient Boost Machine)
  6. RPart
  
Let us start building these models
'
#--------------------------------------------------------------------------------------------------
'Module 3, Part A, Subsection 1: Logistic Regression using SMOTE_STEPWISE'

#Libraries Necessary for Model
library(caret)
library(ROCR)
library(pROC)
# Processing the data
target_column <- ncol(SMOTE_STEPWISE)
test_set=test_set[,c(colnames(SMOTE_STEPWISE))]
test_set$Class <- as.factor(test_set$Class)
SMOTE_STEPWISE$Class <- as.factor(SMOTE_STEPWISE$Class)

#Creating Training and Testing Samples 
trainX <- SMOTE_STEPWISE[, -ncol(SMOTE_STEPWISE)]
trainY <- SMOTE_STEPWISE[, ncol(SMOTE_STEPWISE)]
testX <- test_set[, -ncol(test_set)]
testY <- test_set[, ncol(test_set)]

# Logistic Regression tuning 
trainControl <- trainControl(method = "cv", # cross-validation
                             number = 10,   # number of folds
                             summaryFunction = twoClassSummary, # for binary classification
                             classProbs = TRUE, # to compute class probabilities
                             search = "grid", verbose=1) # type of search

alphaValues <- seq(0.001, 1, by=0.01) # Mixing parameters from ridge (0) to lasso (1)
lambdaValues <- 10^seq(-1, -9, length.out = 100) # Regularization strengths

# Setting up search grid
glmnetGrid <- expand.grid(alpha = alphaValues, 
                          lambda = lambdaValues)

set.seed(123) # For resproducibility

# Training the model
glmnetModel <- train(x = as.matrix(trainX), y = trainY,
                     method = "glmnet",
                     tuneGrid = glmnetGrid,
                     trControl = trainControl,
                     metric = "ROC",
                     preProcess = c("center", "scale"), maxit = 10^3) 
#Printing the best parameters
print(glmnetModel$bestTune)

# Testing

# Get class probabilities
predictions_prob<- predict(glmnetModel, newdata = as.matrix(testX), type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for Logistic Regression(SMOTE Stepwise)")
abline(a=0, b=1, lty=2, col="gray")

# Find the optimal threshold
optimal_coords <- coords(roc_result, "best", best.method="youden")
optimal_threshold <- optimal_coords$threshold

# Print the optimal threshold
cat("Optimal Threshold:", optimal_threshold, "\n")

# Apply this threshold to convert probabilities to class labels
predicted_classes <- ifelse(predictions_prob[,2] > optimal_threshold, 1, 0)
predicted_classes=as.factor(predicted_classes)

testY <- test_set[, ncol(test_set)]
testY=ifelse(testY=="Y",1 , 0)
testY=as.factor(testY)

cm=confusionMatrix(predicted_classes, testY)
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]

calculate_measures<- function(tp, fp, tn, fn){
  tpr = tp / (tp + fn)
  fpr = fp / (fp + tn)
  tnr = tn / (fp + tn)
  fnr = fn / (fn + tp)
  precision = tp / (tp + fp)
  recall = tpr
  f_measure <- (2 * precision * recall) / (precision + recall)
  mcc <- (tp*tn - fp*fn)/(sqrt(tp+fp)*sqrt(tp+fn)*sqrt(tn+fp)*sqrt(tn+fn))
  total = (tp + fn + fp + tn)
  p_o = (tp + tn) / total
  p_e1 = ((tp + fn) / total) * ((tp + fp) / total)
  p_e2 = ((fp + tn) / total) * ((fn + tn) / total)
  p_e = p_e1 + p_e2
  k = (p_o - p_e) / (1 - p_e)
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  measures <- c('TPR', 'FPR', 'TNR', 'FNR', 'Precision', 'Recall', 'F-measure', 'MCC', 'Kappa','Accuracy', 'AUC')
  values <- c(tpr, fpr, tnr, fnr, precision, recall, f_measure, mcc, k,accuracy, auc(roc_result))
  measure.df <- data.frame(measures, values)
  return (measure.df)
}

performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures



#--------------------------------------------------------------------------------------------------
'Module 3, Part A, Subsection 2: Logistic Regression using SMOTE_IG'

#Libraries Necessary for Model
library(caret)
library(ROCR)

test_set=read.csv("initial_test.csv")
# Processing the data
target_column <- ncol(SMOTE_IG)
test_set=test_set[,c(colnames(SMOTE_IG))]
test_set$Class <- as.factor(test_set$Class)
SMOTE_IG$Class <- as.factor(SMOTE_IG$Class)

#Creating Training and Testing Samples 
trainX <- SMOTE_IG[, -ncol(SMOTE_IG)]
trainY <- SMOTE_IG[, ncol(SMOTE_IG)]
testX <- test_set[, -ncol(test_set)]
testY <- test_set[, ncol(test_set)]

# Logistic Regression tuning 
trainControl <- trainControl(method = "cv", # cross-validation
                             number = 10,   # number of folds
                             summaryFunction = twoClassSummary, # for binary classification
                             classProbs = TRUE, # to compute class probabilities
                             search = "grid", verbose=1) # type of search

alphaValues <- seq(0.001, 1, by=0.01) # Mixing parameters from ridge (0) to lasso (1)
lambdaValues <- 10^seq(-1, -9, length.out = 100) # Regularization strengths

# Setting up search grid
glmnetGrid <- expand.grid(alpha = alphaValues, 
                          lambda = lambdaValues)

set.seed(123) # For resproducibility

# Training the model
glmnetModel <- train(x = as.matrix(trainX), y = trainY,
                     method = "glmnet",
                     tuneGrid = glmnetGrid,
                     trControl = trainControl,
                     metric = "ROC",
                     preProcess = c("center", "scale"), maxit = 10^3) 
#Printing the best parameters
print(glmnetModel$bestTune)

# Testing
predictions_prob<- predict(glmnetModel, newdata = as.matrix(testX), type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for Logistic Regression(SMOTE Information Gain)")
abline(a=0, b=1, lty=2, col="gray")

# Find the optimal threshold
optimal_coords <- coords(roc_result, "best", best.method="youden")
optimal_threshold <- optimal_coords$threshold

# Print the optimal threshold
cat("Optimal Threshold:", optimal_threshold, "\n")

# Apply this threshold to convert probabilities to class labels
predicted_classes <- ifelse(predictions_prob[,2] > optimal_threshold, 1, 0)
predicted_classes=as.factor(predicted_classes)

testY <- test_set[, ncol(test_set)]
testY=ifelse(testY=="Y",1 , 0)
testY=as.factor(testY)

cm=confusionMatrix(predicted_classes, testY)
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures

#--------------------------------------------------------------------------------------------------
'Module 3, Part A, Subsection 3: Logistic Regression using SMOTE_RFE'
test_set=read.csv("initial_test.csv")
# Processing the data
target_column <- ncol(SMOTE_RFE)
test_set=test_set[,c(colnames(SMOTE_RFE))]
test_set$Class <- as.factor(test_set$Class)
SMOTE_RFE$Class <- as.factor(SMOTE_RFE$Class)

#Creating Training and Testing Samples 
trainX <- SMOTE_RFE[, -ncol(SMOTE_RFE)]
trainY <- SMOTE_RFE[, ncol(SMOTE_RFE)]
testX <- test_set[, -ncol(test_set)]
testY <- test_set[, ncol(test_set)]

# Logistic Regression tuning 
trainControl <- trainControl(method = "cv", # cross-validation
                             number = 10,   # number of folds
                             summaryFunction = twoClassSummary, # for binary classification
                             classProbs = TRUE, # to compute class probabilities
                             search = "grid", verbose=1) # type of search

alphaValues <- seq(0.001, 1, by=0.01) # Mixing parameters from ridge (0) to lasso (1)
lambdaValues <- 10^seq(-1, -9, length.out = 100) # Regularization strengths

# Setting up search grid
glmnetGrid <- expand.grid(alpha = alphaValues, 
                          lambda = lambdaValues)

set.seed(123) # For resproducibility

# Training the model
glmnetModel <- train(x = as.matrix(trainX), y = trainY,
                     method = "glmnet",
                     tuneGrid = glmnetGrid,
                     trControl = trainControl,
                     metric = "ROC",
                     preProcess = c("center", "scale"), maxit = 10^3) 
#Printing the best parameters
print(glmnetModel$bestTune)

# Testing
predictions_prob<- predict(glmnetModel, newdata = as.matrix(testX), type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for Logistic Regression(SMOTE RFE)")
abline(a=0, b=1, lty=2, col="gray")

# Find the optimal threshold
optimal_coords <- coords(roc_result, "best", best.method="youden")
optimal_threshold <- optimal_coords$threshold

# Print the optimal threshold
cat("Optimal Threshold:", optimal_threshold, "\n")

# Apply this threshold to convert probabilities to class labels
predicted_classes <- ifelse(predictions_prob[,2] > optimal_threshold, 1, 0)
predicted_classes=as.factor(predicted_classes)

testY <- test_set[, ncol(test_set)]
testY=ifelse(testY=="Y",1 , 0)
testY=as.factor(testY)

cm=confusionMatrix(predicted_classes, testY)
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#--------------------------------------------------------------------------------------------------
'Module 3, Part A, Subsection 4: Logistic Regression using BOTH_IG'
test_set=read.csv("initial_test.csv")
# Processing the data
target_column <- ncol(BOTH_IG)
test_set=test_set[,c(colnames(BOTH_IG))]
test_set$Class <- as.factor(test_set$Class)
BOTH_IG$Class <- as.factor(BOTH_IG$Class)

#Creating Training and Testing Samples 
trainX <- BOTH_IG[, -ncol(BOTH_IG)]
trainY <- BOTH_IG[, ncol(BOTH_IG)]
testX <- test_set[, -ncol(test_set)]
testY <- test_set[, ncol(test_set)]

# Logistic Regression tuning 
trainControl <- trainControl(method = "cv", # cross-validation
                             number = 10,   # number of folds
                             summaryFunction = twoClassSummary, # for binary classification
                             classProbs = TRUE, # to compute class probabilities
                             search = "grid", verbose=1) # type of search

alphaValues <- seq(0.001, 1, by=0.01) # Mixing parameters from ridge (0) to lasso (1)
lambdaValues <- 10^seq(-1, -9, length.out = 100) # Regularization strengths

# Setting up search grid
glmnetGrid <- expand.grid(alpha = alphaValues, 
                          lambda = lambdaValues)

set.seed(123) # For resproducibility

# Training the model
glmnetModel <- train(x = as.matrix(trainX), y = trainY,
                     method = "glmnet",
                     tuneGrid = glmnetGrid,
                     trControl = trainControl,
                     metric = "ROC",
                     preProcess = c("center", "scale"), maxit = 10^3) 
#Printing the best parameters
print(glmnetModel$bestTune)

# Testing
predictions_prob<- predict(glmnetModel, newdata = as.matrix(testX), type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for Logistic Regression(BOTH Information Gain)")
abline(a=0, b=1, lty=2, col="gray")

# Find the optimal threshold
optimal_coords <- coords(roc_result, "best", best.method="youden")
optimal_threshold <- optimal_coords$threshold

# Print the optimal threshold
cat("Optimal Threshold:", optimal_threshold, "\n")

# Apply this threshold to convert probabilities to class labels
predicted_classes <- ifelse(predictions_prob[,2] > optimal_threshold, 1, 0)
predicted_classes=as.factor(predicted_classes)

testY <- test_set[, ncol(test_set)]
testY=ifelse(testY=="Y",1 , 0)
testY=as.factor(testY)

cm=confusionMatrix(predicted_classes, testY)
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#--------------------------------------------------------------------------------------------------
'Module 3, Part A, Subsection 5: Logistic Regression using BOTH_STEPWISE'
test_set=read.csv("initial_test.csv")
# Processing the data
target_column <- ncol(BOTH_STEPWISE)
test_set=test_set[,c(colnames(BOTH_STEPWISE))]
test_set$Class <- as.factor(test_set$Class)
BOTH_STEPWISE$Class <- as.factor(BOTH_STEPWISE$Class)

#Creating Training and Testing Samples 
trainX <- BOTH_STEPWISE[, -ncol(BOTH_STEPWISE)]
trainY <- BOTH_STEPWISE[, ncol(BOTH_STEPWISE)]
testX <- test_set[, -ncol(test_set)]
testY <- test_set[, ncol(test_set)]

# Logistic Regression tuning 
trainControl <- trainControl(method = "cv", # cross-validation
                             number = 10,   # number of folds
                             summaryFunction = twoClassSummary, # for binary classification
                             classProbs = TRUE, # to compute class probabilities
                             search = "grid", verbose=1) # type of search

alphaValues <- seq(0.001, 1, by=0.01) # Mixing parameters from ridge (0) to lasso (1)
lambdaValues <- 10^seq(-1, -9, length.out = 100) # Regularization strengths

# Setting up search grid
glmnetGrid <- expand.grid(alpha = alphaValues, 
                          lambda = lambdaValues)

set.seed(123) # For resproducibility

# Training the model
glmnetModel <- train(x = as.matrix(trainX), y = trainY,
                     method = "glmnet",
                     tuneGrid = glmnetGrid,
                     trControl = trainControl,
                     metric = "ROC",
                     preProcess = c("center", "scale"), maxit = 10^3) 
#Printing the best parameters
print(glmnetModel$bestTune)

# Testing
predictions_prob<- predict(glmnetModel, newdata = as.matrix(testX), type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for Logistic Regression(BOTH Stepwise)")
abline(a=0, b=1, lty=2, col="gray")

# Find the optimal threshold
optimal_coords <- coords(roc_result, "best", best.method="youden")
optimal_threshold <- optimal_coords$threshold

# Print the optimal threshold
cat("Optimal Threshold:", optimal_threshold, "\n")

# Apply this threshold to convert probabilities to class labels
predicted_classes <- ifelse(predictions_prob[,2] > optimal_threshold, 1, 0)
predicted_classes=as.factor(predicted_classes)

testY <- test_set[, ncol(test_set)]
testY=ifelse(testY=="Y",1 , 0)
testY=as.factor(testY)

cm=confusionMatrix(predicted_classes, testY)
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#--------------------------------------------------------------------------------------------------
'Module 3, Part A, Subsection 6: Logistic Regression using BOTH_RFE'
test_set=read.csv("initial_test.csv")
# Processing the data
target_column <- ncol(BOTH_RFE)
test_set=test_set[,c(colnames(BOTH_RFE))]
test_set$Class <- as.factor(test_set$Class)
BOTH_RFE$Class <- as.factor(BOTH_RFE$Class)

#Creating Training and Testing Samples 
trainX <- BOTH_RFE[, -ncol(BOTH_RFE)]
trainY <- BOTH_RFE[, ncol(BOTH_RFE)]
testX <- test_set[, -ncol(test_set)]
testY <- test_set[, ncol(test_set)]

# Logistic Regression tuning 
trainControl <- trainControl(method = "cv", # cross-validation
                             number = 10,   # number of folds
                             summaryFunction = twoClassSummary, # for binary classification
                             classProbs = TRUE, # to compute class probabilities
                             search = "grid", verbose=1) # type of search

alphaValues <- seq(0.001, 1, by=0.01) # Mixing parameters from ridge (0) to lasso (1)
lambdaValues <- 10^seq(-1, -9, length.out = 100) # Regularization strengths

# Setting up search grid
glmnetGrid <- expand.grid(alpha = alphaValues, 
                          lambda = lambdaValues)

set.seed(123) # For resproducibility

# Training the model
glmnetModel <- train(x = as.matrix(trainX), y = trainY,
                     method = "glmnet",
                     tuneGrid = glmnetGrid,
                     trControl = trainControl,
                     metric = "ROC",
                     preProcess = c("center", "scale"), maxit = 10^3) 
#Printing the best parameters
print(glmnetModel$bestTune)


# Testing
predictions_prob<- predict(glmnetModel, newdata = as.matrix(testX), type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for Logistic Regression(BOTH RFE)")
abline(a=0, b=1, lty=2, col="gray")

# Find the optimal threshold
optimal_coords <- coords(roc_result, "best", best.method="youden")
optimal_threshold <- optimal_coords$threshold

# Print the optimal threshold
cat("Optimal Threshold:", optimal_threshold, "\n")

# Apply this threshold to convert probabilities to class labels
predicted_classes <- ifelse(predictions_prob[,2] > optimal_threshold, 1, 0)
predicted_classes=as.factor(predicted_classes)

testY <- test_set[, ncol(test_set)]
testY=ifelse(testY=="Y",1 , 0)
testY=as.factor(testY)

cm=confusionMatrix(predicted_classes, testY)
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part B, Subsection 1: GBM with SMOTE_STEPWISE'
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(SMOTE_STEPWISE))]

test_set$Class <- as.factor(test_set$Class)
SMOTE_STEPWISE$Class <- as.factor(SMOTE_STEPWISE$Class)


trainX <- SMOTE_STEPWISE[, -ncol(SMOTE_STEPWISE)]  
trainY <- SMOTE_STEPWISE[, ncol(SMOTE_STEPWISE)]   # Target variable
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)] 

library(caret)
library(gbm)

# Define the tuning grid
tuningGrid <- expand.grid(
  interaction.depth = seq(1, 3, by=1),
  n.trees = seq(10, 100, by=10),
  shrinkage = c(0.10),
  n.minobsinnode = seq(1,10,by=1)
)

# Set up training control
trainControl <- trainControl(method = "cv",
                             number = 5,
                             classProbs = TRUE, # Necessary for ROC/AUC
                             summaryFunction = twoClassSummary,
                             search = "grid", verbose=1, allowParallel=TRUE, p=0.67)

# Train the model
set.seed(123)
gbmTunedModel <- train(x = trainX, y = trainY,
                       method = "gbm",
                       trControl = trainControl,
                       tuneGrid = tuningGrid,
                       metric = "ROC",
                       verbose = FALSE, distribution="bernoulli")


# Print the best model's details
print(gbmTunedModel)

predictions_prob<- predict(gbmTunedModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for GBM SMOTE STEPWISE")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(gbmTunedModel, newdata = testX, type = "raw")

cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#--------------------------------------------------------------------------------------------------
'Module 3, Part B, Subsection 2: GBM with SMOTE_IG'
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(SMOTE_IG))]
test_set$Class <- as.factor(test_set$Class)
SMOTE_IG$Class <- as.factor(SMOTE_IG$Class)


trainX <- SMOTE_IG[, -ncol(SMOTE_IG)]  
trainY <- SMOTE_IG[, ncol(SMOTE_IG)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)] 

library(caret)
library(gbm)

# Define the tuning grid
tuningGrid <- expand.grid(
  interaction.depth = seq(1, 3, by=1),
  n.trees = seq(10, 100, by=5),
  shrinkage = c(0.10),
  n.minobsinnode = seq(1,10,by=1)
)

# Set up training control
trainControl <- trainControl(method = "cv",
                             number = 5,
                             classProbs = TRUE, # Necessary for ROC/AUC
                             summaryFunction = twoClassSummary,
                             search = "grid", verbose=1, allowParallel=TRUE, p=0.67)

# Train the model
set.seed(123)
gbmTunedModel <- train(x = trainX, y = trainY,
                       method = "gbm",
                       trControl = trainControl,
                       tuneGrid = tuningGrid,
                       metric = "ROC",
                       verbose = FALSE, distribution="bernoulli")


# Print the best model's details
print(gbmTunedModel)

predictions_prob<- predict(gbmTunedModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for GBM SMOTE IG")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(gbmTunedModel, newdata = testX, type = "raw")

cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#--------------------------------------------------------------------------------------------------
'Module 3, Part B, Subsection 3: GBM with SMOTE_RFE'
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(SMOTE_RFE))]
test_set$Class <- as.factor(test_set$Class)
SMOTE_RFE$Class <- as.factor(SMOTE_RFE$Class)


trainX <- SMOTE_RFE[, -ncol(SMOTE_RFE)]  
trainY <- SMOTE_RFE[, ncol(SMOTE_RFE)]   # Target variable
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)] 

library(caret)
library(gbm)

# Define the tuning grid
tuningGrid <- expand.grid(
  interaction.depth = seq(1, 3, by=1),
  n.trees = seq(10, 100, by=10),
  shrinkage = c(0.10),
  n.minobsinnode = seq(1,10,by=1)
)

# Set up training control
trainControl <- trainControl(method = "cv",
                             number = 5,
                             classProbs = TRUE, # Necessary for ROC/AUC
                             summaryFunction = twoClassSummary,
                             search = "grid", verbose=1, allowParallel=TRUE, p=0.67)

# Train the model
set.seed(123)
gbmTunedModel <- train(x = trainX, y = trainY,
                       method = "gbm",
                       trControl = trainControl,
                       tuneGrid = tuningGrid,
                       metric = "ROC",
                       verbose = FALSE, distribution="bernoulli")


# Print the best model's details
print(gbmTunedModel)

predictions_prob <- predict(gbmTunedModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for GBM SMOTE RFE")
abline(a=0, b=1, lty=2, col="gray")

predictions <- predict(gbmTunedModel, newdata = testX, type = "raw")

cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures

#--------------------------------------------------------------------------------------------------
'Module 3, Part B, Subsection 4: GBM with BOTH_STEPWISE'
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(BOTH_STEPWISE))]

test_set$Class <- as.factor(test_set$Class)
BOTH_STEPWISE$Class <- as.factor(BOTH_STEPWISE$Class)


trainX <- BOTH_STEPWISE[, -ncol(BOTH_STEPWISE)]  
trainY <- BOTH_STEPWISE[, ncol(BOTH_STEPWISE)]   # Target variable
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)] 

library(caret)
library(gbm)

# Define the tuning grid
tuningGrid <- expand.grid(
  interaction.depth = seq(1, 3, by=1),
  n.trees = seq(10, 100, by=10),
  shrinkage = c(0.10),
  n.minobsinnode = seq(1,10,by=1)
)

# Set up training control
trainControl <- trainControl(method = "cv",
                             number = 5,
                             classProbs = TRUE, # Necessary for ROC/AUC
                             summaryFunction = twoClassSummary,
                             search = "grid", verbose=1, allowParallel=TRUE, p=0.67)

# Train the model
set.seed(123)
gbmTunedModel <- train(x = trainX, y = trainY,
                       method = "gbm",
                       trControl = trainControl,
                       tuneGrid = tuningGrid,
                       metric = "ROC",
                       verbose = FALSE, distribution="bernoulli")


# Print the best model's details
print(gbmTunedModel)

predictions_prob<- predict(gbmTunedModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for GBM BOTH STEPWISE")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(gbmTunedModel, newdata = testX, type = "raw")

cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#--------------------------------------------------------------------------------------------------
'Module 3, Part B, Subsection 5: GBM with BOTH_IG'
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(BOTH_IG))]

test_set$Class <- as.factor(test_set$Class)
BOTH_IG$Class <- as.factor(BOTH_IG$Class)


trainX <- BOTH_IG[, -ncol(BOTH_IG)]  
trainY <- BOTH_IG[, ncol(BOTH_IG)]   # Target variable
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)] 

library(caret)
library(gbm)

# Define the tuning grid
tuningGrid <- expand.grid(
  interaction.depth = seq(1, 3, by=1),
  n.trees = seq(10, 100, by=10),
  shrinkage = c(0.10),
  n.minobsinnode = seq(1,10,by=1)
)

# Set up training control
trainControl <- trainControl(method = "cv",
                             number = 10,
                             classProbs = TRUE, # Necessary for ROC/AUC
                             summaryFunction = twoClassSummary,
                             search = "grid", verbose=1, allowParallel=TRUE, p=0.67)

# Train the model
set.seed(123)
gbmTunedModel <- train(x = trainX, y = trainY,
                       method = "gbm",
                       trControl = trainControl,
                       tuneGrid = tuningGrid,
                       metric = "ROC",
                       verbose = FALSE, distribution="bernoulli")


# Print the best model's details
print(gbmTunedModel)

predictions_prob<- predict(gbmTunedModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for GBM BOTH IG")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(gbmTunedModel, newdata = testX, type = "raw")

cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#--------------------------------------------------------------------------------------------------
'Module 3, Part B, Subsection 6: GBM with BOTH_RFE'
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(BOTH_RFE))]

test_set$Class <- as.factor(test_set$Class)
BOTH_RFE$Class <- as.factor(BOTH_RFE$Class)


trainX <- BOTH_RFE[, -ncol(BOTH_RFE)]  
trainY <- BOTH_RFE[, ncol(BOTH_RFE)]   # Target variable
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)] 

library(caret)
library(gbm)

# Define the tuning grid
tuningGrid <- expand.grid(
  interaction.depth = seq(1, 3, by=1),
  n.trees = seq(10, 100, by=10),
  shrinkage = c(0.10),
  n.minobsinnode = seq(1,10,by=1)
)

# Set up training control
trainControl <- trainControl(method = "cv",
                             number = 10,
                             classProbs = TRUE, # Necessary for ROC/AUC
                             summaryFunction = twoClassSummary,
                             search = "grid", verbose=1, allowParallel=TRUE, p=0.67)

# Train the model
set.seed(123)
gbmTunedModel <- train(x = trainX, y = trainY,
                       method = "gbm",
                       trControl = trainControl,
                       tuneGrid = tuningGrid,
                       metric = "ROC",
                       verbose = FALSE, distribution="bernoulli")


# Print the best model's details
print(gbmTunedModel)

predictions_prob<- predict(gbmTunedModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for GBM BOTH RFE")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(gbmTunedModel, newdata = testX, type = "raw")

cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part C, Subsection 1: Naive Bayes with SMOTE_STEPWISE'

#Necessary Library
library(caret)
library(e1071)  # Naive Bayes is implemented in the e1071 package
SMOTE_STEPWISE=read.csv('SMOTE_STEPWISE.csv')
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(SMOTE_STEPWISE))]

test_set$Class <- as.factor(test_set$Class)
SMOTE_STEPWISE$Class <- as.factor(SMOTE_STEPWISE$Class)


trainX <- SMOTE_STEPWISE[, -ncol(SMOTE_STEPWISE)]  
trainY <- SMOTE_STEPWISE[, ncol(SMOTE_STEPWISE)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

tuningGrid <- expand.grid(
  laplace = seq(0.1, 1, by = 0.01),  # Laplace smoothing parameter
  usekernel = c(TRUE),  # Whether to use kernel density estimation rather than assuming Gaussian distributions
  adjust = c(0.5, 1, 2)  # Adjustment parameter for kernel density estimation
)

trainControl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, verbose=1)
set.seed(225)
nbModel <- train(x = trainX, y = trainY, method = "naive_bayes", trControl = trainControl, tuneGrid = tuningGrid, metric = "ROC")

#Testing
predictions_prob<- predict(nbModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for NAIVE BAYES SMOTE STEPWISE")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(nbModel, newdata = testX, type = "raw")
cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part C, Subsection 2: Naive Bayes with SMOTE_IG'

#Necessary Library
library(caret)
library(e1071)  # Naive Bayes is implemented in the e1071 packag
SMOTE_IG=read.csv('SMOTE_IG.csv')
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(SMOTE_IG))]

test_set$Class <- as.factor(test_set$Class)
SMOTE_IG$Class <- as.factor(SMOTE_IG$Class)


trainX <- SMOTE_IG[, -ncol(SMOTE_IG)]  
trainY <- SMOTE_IG[, ncol(SMOTE_IG)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

tuningGrid <- expand.grid(
  laplace = seq(0.1, 1, by = 0.01),  # Laplace smoothing parameter
  usekernel = c(TRUE),  # Whether to use kernel density estimation rather than assuming Gaussian distributions
  adjust = c(0.5, 1, 2)  # Adjustment parameter for kernel density estimation
)

trainControl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, verbose=1)
set.seed(225)
nbModel <- train(x = trainX, y = trainY, method = "naive_bayes", trControl = trainControl, tuneGrid = tuningGrid, metric = "ROC")

#Testing
library(pROC)
predictions_prob<- predict(nbModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for NAIVE BAYES SMOTE IG")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(nbModel, newdata = testX, type = "raw")
cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part C, Subsection 3: Naive Bayes with SMOTE_RFE'

#Necessary Library
library(caret)
library(e1071)  # Naive Bayes is implemented in the e1071 packag
SMOTE_RFE=read.csv('SMOTE_RFE.csv')
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(SMOTE_RFE))]

test_set$Class <- as.factor(test_set$Class)
SMOTE_RFE$Class <- as.factor(SMOTE_RFE$Class)


trainX <- SMOTE_RFE[, -ncol(SMOTE_RFE)]  
trainY <- SMOTE_RFE[, ncol(SMOTE_RFE)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

tuningGrid <- expand.grid(
  laplace = seq(0.1, 1, by = 0.01),  # Laplace smoothing parameter
  usekernel = c(TRUE),  # Whether to use kernel density estimation rather than assuming Gaussian distributions
  adjust = c(0.5, 1, 2)  # Adjustment parameter for kernel density estimation
)

trainControl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, verbose=1)
set.seed(225)
nbModel <- train(x = trainX, y = trainY, method = "naive_bayes", trControl = trainControl, tuneGrid = tuningGrid, metric = "ROC")

#Testing
predictions_prob<- predict(nbModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for NAIVE BAYES SMOTE RFE")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(nbModel, newdata = testX, type = "raw")
cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part C, Subsection 4: Naive Bayes with BOTH_STEPWISE'

#Necessary Library
library(caret)
library(e1071)  # Naive Bayes is implemented in the e1071 packag
BOTH_STEPWISE=read.csv('BOTH_STEPWISE.csv')
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(BOTH_STEPWISE))]

test_set$Class <- as.factor(test_set$Class)
BOTH_STEPWISE$Class <- as.factor(BOTH_STEPWISE$Class)


trainX <- BOTH_STEPWISE[, -ncol(BOTH_STEPWISE)]  
trainY <- BOTH_STEPWISE[, ncol(BOTH_STEPWISE)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

tuningGrid <- expand.grid(
  laplace = seq(0.1, 1, by = 0.01),  # Laplace smoothing parameter
  usekernel = c(TRUE),  # Whether to use kernel density estimation rather than assuming Gaussian distributions
  adjust = c(0.5, 1, 2)  # Adjustment parameter for kernel density estimation
)

trainControl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, verbose=1)
set.seed(225)
nbModel <- train(x = trainX, y = trainY, method = "naive_bayes", trControl = trainControl, tuneGrid = tuningGrid, metric = "ROC")

#Testing
predictions_prob<- predict(nbModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for NAIVE BAYES BOTH STEPWISE")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(nbModel, newdata = testX, type = "raw")
cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part C, Subsection 5: Naive Bayes with BOTH_IG'

#Necessary Library
library(caret)
library(e1071)  # Naive Bayes is implemented in the e1071 packag
BOTH_IG=read.csv('BOTH_IG.csv')
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(BOTH_IG))]

test_set$Class <- as.factor(test_set$Class)
BOTH_IG$Class <- as.factor(BOTH_IG$Class)


trainX <- BOTH_IG[, -ncol(BOTH_IG)]  
trainY <- BOTH_IG[, ncol(BOTH_IG)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

tuningGrid <- expand.grid(
  laplace = seq(0.1, 1, by = 0.01),  # Laplace smoothing parameter
  usekernel = c(TRUE),  # Whether to use kernel density estimation rather than assuming Gaussian distributions
  adjust = c(0.5, 1, 2)  # Adjustment parameter for kernel density estimation
)

trainControl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, verbose=1)
set.seed(225)
nbModel <- train(x = trainX, y = trainY, method = "naive_bayes", trControl = trainControl, tuneGrid = tuningGrid, metric = "ROC")

#Testing
predictions_prob<- predict(nbModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for NAIVE BAYES BOTH IG")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(nbModel, newdata = testX, type = "raw")
cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part C, Subsection 5: Naive Bayes with BOTH_RFE'

#Necessary Library
library(caret)
library(e1071)  # Naive Bayes is implemented in the e1071 packag
BOTH_RFE=read.csv('BOTH_RFE.csv')
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(BOTH_RFE))]

test_set$Class <- as.factor(test_set$Class)
BOTH_RFE$Class <- as.factor(BOTH_RFE$Class)


trainX <- BOTH_RFE[, -ncol(BOTH_RFE)]  
trainY <- BOTH_RFE[, ncol(BOTH_RFE)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

tuningGrid <- expand.grid(
  laplace = seq(0.1, 1, by = 0.01),  # Laplace smoothing parameter
  usekernel = c(TRUE),  # Whether to use kernel density estimation rather than assuming Gaussian distributions
  adjust = c(0.5, 1, 2)  # Adjustment parameter for kernel density estimation
)

trainControl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, verbose=1)
set.seed(225)
nbModel <- train(x = trainX, y = trainY, method = "naive_bayes", trControl = trainControl, tuneGrid = tuningGrid, metric = "ROC")

#Testing
predictions_prob<- predict(nbModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for NAIVE BAYES BOTH RFE")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(nbModel, newdata = testX, type = "raw")
cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part D, Subsection 1: LDA WITH SMOTE_STEPWISE'

library(MASS)
library(caret)
SMOTE_STEPWISE=read.csv("SMOTE_STEPWISE.csv")
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(SMOTE_STEPWISE))]

test_set$Class <- as.factor(test_set$Class)
SMOTE_STEPWISE$Class <- as.factor(SMOTE_STEPWISE$Class)


trainX <- SMOTE_STEPWISE[, -ncol(SMOTE_STEPWISE)]  
trainY <- SMOTE_STEPWISE[, ncol(SMOTE_STEPWISE)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

# Train LDA model
set.seed(225)
ldaModel <- lda(trainY ~ ., data = trainX)

#Testing
predictions_prob<- predict(ldaModel, newdata = testX)
roc_result <- roc(as.factor(testY), predictions_prob$posterior[,2])
plot(roc_result, main="ROC Curve for LDA SMOTE STEPWISE")
abline(a=0, b=1, lty=2, col="gray")

predictions <- predict(ldaModel, newdata = testX)
predClasses <- predictions$class
cm=confusionMatrix(predClasses, as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part D, Subsection 2: LDA WITH SMOTE_IG'
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(SMOTE_IG))]

test_set$Class <- as.factor(test_set$Class)
SMOTE_IG$Class <- as.factor(SMOTE_IG$Class)


trainX <- SMOTE_IG[, -ncol(SMOTE_IG)]  
trainY <- SMOTE_IG[, ncol(SMOTE_IG)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

# Train LDA model
set.seed(225)
ldaModel <- lda(trainY ~ ., data = trainX)

#Testing
predictions_prob<- predict(ldaModel, newdata = testX)
roc_result <- roc(as.factor(testY), predictions_prob$posterior[,2])
plot(roc_result, main="ROC Curve for LDA SMOTE IG")
abline(a=0, b=1, lty=2, col="gray")

predictions <- predict(ldaModel, newdata = testX)
predClasses <- predictions$class
cm=confusionMatrix(predClasses, as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part D, Subsection 3: LDA WITH SMOTE_RFE'
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(SMOTE_RFE))]

test_set$Class <- as.factor(test_set$Class)
SMOTE_RFE$Class <- as.factor(SMOTE_RFE$Class)


trainX <- SMOTE_RFE[, -ncol(SMOTE_RFE)]  
trainY <- SMOTE_RFE[, ncol(SMOTE_RFE)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

# Train LDA model
set.seed(225)
ldaModel <- lda(trainY ~ ., data = trainX)

#Testing
predictions_prob<- predict(ldaModel, newdata = testX)
roc_result <- roc(as.factor(testY), predictions_prob$posterior[,2])
plot(roc_result, main="ROC Curve for LDA SMOTE RFE")
abline(a=0, b=1, lty=2, col="gray")

predictions <- predict(ldaModel, newdata = testX)
predClasses <- predictions$class
cm=confusionMatrix(predClasses, as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part D, Subsection 1: LDA WITH BOTH_STEPWISE'
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(BOTH_STEPWISE))]

test_set$Class <- as.factor(test_set$Class)
BOTH_STEPWISE$Class <- as.factor(BOTH_STEPWISE$Class)


trainX <- BOTH_STEPWISE[, -ncol(BOTH_STEPWISE)]  
trainY <- BOTH_STEPWISE[, ncol(BOTH_STEPWISE)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

# Train LDA model
set.seed(225)
ldaModel <- lda(trainY ~ ., data = trainX)

#Testing
predictions_prob<- predict(ldaModel, newdata = testX)
roc_result <- roc(as.factor(testY), predictions_prob$posterior[,2])
plot(roc_result, main="ROC Curve for LDA BOTH STEPWISE")
abline(a=0, b=1, lty=2, col="gray")

predictions <- predict(ldaModel, newdata = testX)
predClasses <- predictions$class
cm=confusionMatrix(predClasses, as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part D, Subsection 5: LDA WITH BOTH IG'
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(BOTH_IG))]

test_set$Class <- as.factor(test_set$Class)
BOTH_IG$Class <- as.factor(BOTH_IG$Class)


trainX <- BOTH_IG[, -ncol(BOTH_IG)]  
trainY <- BOTH_IG[, ncol(BOTH_IG)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

# Train LDA model
set.seed(225)
ldaModel <- lda(trainY ~ ., data = trainX)

#Testing
predictions_prob<- predict(ldaModel, newdata = testX)
roc_result <- roc(as.factor(testY), predictions_prob$posterior[,2])
plot(roc_result, main="ROC Curve for LDA BOTH IG")
abline(a=0, b=1, lty=2, col="gray")

predictions <- predict(ldaModel, newdata = testX)
predClasses <- predictions$class
cm=confusionMatrix(predClasses, as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part D, Subsection 6: LDA WITH BOTH RFE'
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(BOTH_RFE))]

test_set$Class <- as.factor(test_set$Class)
BOTH_RFE$Class <- as.factor(BOTH_RFE$Class)


trainX <- BOTH_RFE[, -ncol(BOTH_RFE)]  
trainY <- BOTH_RFE[, ncol(BOTH_RFE)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

# Train LDA model
set.seed(225)
ldaModel <- lda(trainY ~ ., data = trainX)

#Testing
predictions_prob<- predict(ldaModel, newdata = testX)
roc_result <- roc(as.factor(testY), predictions_prob$posterior[,2])
plot(roc_result, main="ROC Curve for LDA BOTH RFE")
abline(a=0, b=1, lty=2, col="gray")

predictions <- predict(ldaModel, newdata = testX)
predClasses <- predictions$class
cm=confusionMatrix(predClasses, as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part E, Subsection 1: RANDOM FOREST WITH SMOTE STEPWISE'

test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(SMOTE_STEPWISE))]
test_set$Class <- as.factor(test_set$Class)
SMOTE_STEPWISE$Class <- as.factor(SMOTE_STEPWISE$Class)

trainX <- SMOTE_STEPWISE[, -ncol(SMOTE_STEPWISE)]  
trainY <- SMOTE_STEPWISE[, ncol(SMOTE_STEPWISE)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

trainControl <- trainControl(method="cv", number=5, savePredictions="final", classProbs=TRUE, summaryFunction=twoClassSummary, verbose=1)

tuningGrid <- expand.grid(
  mtry = seq(10,50, by=5), 
  splitrule = c("gini", "extratrees"),
  min.node.size = seq(2000,3100,by=100)
)
set.seed(123)
rfModel <- train(
  x = trainX, y = trainY,
  method = "ranger", 
  trControl = trainControl, 
  tuneGrid = tuningGrid, 
  metric = "ROC",
  importance = 'impurity', replace=TRUE # Optionally, capture variable importance
)
predictions_prob<- predict(rfModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for RANDOM FOREST SMOTE STEPWISE")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(rfModel, newdata = testX, type = "raw")
cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part E, Subsection 2: RANDOM FOREST WITH SMOTE IG'

test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(SMOTE_IG))]
test_set$Class <- as.factor(test_set$Class)
SMOTE_IG$Class <- as.factor(SMOTE_IG$Class)

trainX <- SMOTE_IG[, -ncol(SMOTE_IG)]  
trainY <- SMOTE_IG[, ncol(SMOTE_IG)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

trainControl <- trainControl(method="cv", number=5, savePredictions="final", classProbs=TRUE, summaryFunction=twoClassSummary, verbose=1)

tuningGrid <- expand.grid(
  mtry = seq(10,50, by=5), 
  splitrule = c("gini", "extratrees"),
  min.node.size = seq(2000,3100,by=100)
)
set.seed(123)
rfModel <- train(
  x = trainX, y = trainY,
  method = "ranger", 
  trControl = trainControl, 
  tuneGrid = tuningGrid, 
  metric = "ROC",
  importance = 'impurity', replace=TRUE # Optionally, capture variable importance
)
predictions_prob<- predict(rfModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for RANDOM FOREST SMOTE IG")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(rfModel, newdata = testX, type = "raw")
cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part E, Subsection 3: RANDOM FOREST WITH SMOTE RFE'

test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(SMOTE_RFE))]
test_set$Class <- as.factor(test_set$Class)
SMOTE_RFE$Class <- as.factor(SMOTE_RFE$Class)

trainX <- SMOTE_RFE[, -ncol(SMOTE_RFE)]  
trainY <- SMOTE_RFE[, ncol(SMOTE_RFE)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

trainControl <- trainControl(method="cv", number=5, savePredictions="final", classProbs=TRUE, summaryFunction=twoClassSummary, verbose=1)

tuningGrid <- expand.grid(
  mtry = seq(10,50, by=5), 
  splitrule = c("gini", "extratrees"),
  min.node.size = seq(2000,3100,by=100)
)
set.seed(123)
rfModel <- train(
  x = trainX, y = trainY,
  method = "ranger", 
  trControl = trainControl, 
  tuneGrid = tuningGrid, 
  metric = "ROC",
  importance = 'impurity', replace=TRUE # Optionally, capture variable importance
)
predictions_prob<- predict(rfModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for RANDOM FOREST SMOTE RFE")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(rfModel, newdata = testX, type = "raw")
cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part E, Subsection 1: RANDOM FOREST WITH BOTH STEPWISE'

test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(BOTH_STEPWISE))]
test_set$Class <- as.factor(test_set$Class)
BOTH_STEPWISE$Class <- as.factor(BOTH_STEPWISE$Class)

trainX <- BOTH_STEPWISE[, -ncol(BOTH_STEPWISE)]  
trainY <- BOTH_STEPWISE[, ncol(BOTH_STEPWISE)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

trainControl <- trainControl(method="cv", number=5, savePredictions="final", classProbs=TRUE, summaryFunction=twoClassSummary, verbose=1)

tuningGrid <- expand.grid(
  mtry = seq(10,50, by=5), 
  splitrule = c("gini", "extratrees"),
  min.node.size = seq(2000,3100,by=100)
)
set.seed(123)
rfModel <- train(
  x = trainX, y = trainY,
  method = "ranger", 
  trControl = trainControl, 
  tuneGrid = tuningGrid, 
  metric = "ROC",
  importance = 'impurity', replace=TRUE # Optionally, capture variable importance
)
predictions_prob<- predict(rfModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for RANDOM FOREST BOTH STEPWISE")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(rfModel, newdata = testX, type = "raw")
cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part E, Subsection 5: RANDOM FOREST WITH BOTH IG'

test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(BOTH_IG))]
test_set$Class <- as.factor(test_set$Class)
BOTH_IG$Class <- as.factor(BOTH_IG$Class)

trainX <- BOTH_IG[, -ncol(BOTH_IG)]  
trainY <- BOTH_IG[, ncol(BOTH_IG)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

trainControl <- trainControl(method="cv", number=5, savePredictions="final", classProbs=TRUE, summaryFunction=twoClassSummary, verbose=1)

tuningGrid <- expand.grid(
  mtry = seq(10,50, by=5), 
  splitrule = c("gini", "extratrees"),
  min.node.size = seq(2000,3100,by=100)
)
set.seed(123)
rfModel <- train(
  x = trainX, y = trainY,
  method = "ranger", 
  trControl = trainControl, 
  tuneGrid = tuningGrid, 
  metric = "ROC",
  importance = 'impurity', replace=TRUE # Optionally, capture variable importance
)
predictions_prob<- predict(rfModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for RANDOM FOREST BOTH IG")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(rfModel, newdata = testX, type = "raw")
cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part E, Subsection 6: RANDOM FOREST WITH BOTH RFE'

test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(BOTH_RFE))]
test_set$Class <- as.factor(test_set$Class)
BOTH_RFE$Class <- as.factor(BOTH_RFE$Class)

trainX <- BOTH_RFE[, -ncol(BOTH_RFE)]  
trainY <- BOTH_RFE[, ncol(BOTH_RFE)]   
testX <- test_set[, -ncol(test_set)]     
testY <- test_set[, ncol(test_set)]

trainControl <- trainControl(method="cv", number=5, savePredictions="final", classProbs=TRUE, summaryFunction=twoClassSummary, verbose=1)

tuningGrid <- expand.grid(
  mtry = seq(10,50, by=5), 
  splitrule = c("gini", "extratrees"),
  min.node.size = seq(2000,3100,by=100)
)
set.seed(123)
rfModel <- train(
  x = trainX, y = trainY,
  method = "ranger", 
  trControl = trainControl, 
  tuneGrid = tuningGrid, 
  metric = "ROC",
  importance = 'impurity', replace=TRUE # Optionally, capture variable importance
)
predictions_prob<- predict(rfModel, newdata = testX, type = "prob")
roc_result <- roc(as.factor(testY), predictions_prob[,2])
plot(roc_result, main="ROC Curve for RANDOM FOREST BOTH RFE")
abline(a=0, b=1, lty=2, col="gray")


predictions <- predict(rfModel, newdata = testX, type = "raw")
cm=confusionMatrix(as.factor(predictions), as.factor(testY))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures
#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part F, Subsection 1: RPART WITH SMOTE STEPWISE'

train_data <- read.csv("SMOTE_STEPWISE.csv")
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(BOTH_RFE))]

train_control <- trainControl(method = "cv", number = 10, search = "grid")
tune_grid <- expand.grid(.k = seq(3, 21, 2))  # k values from 3 to 21, odd numbers to avoid ties

set.seed(123)
knn_model <- train(Class ~ ., data = train_data, method = "knn",
                   trControl = train_control, tuneGrid = tune_grid,
                   preProcess = c("center", "scale"))  # Preprocessing: standardize variables

plot(knn_model)
prob_predictions <- predict(knn_model, newdata = test_set, type = "prob")
roc_result <- roc(response = test_set$Class, predictor = prob_predictions[,2])
plot(roc_result, main="ROC Curve for KNN Model SMOTE STEPWISE")
abline(a=0, b=1, lty=2, col="gray")  # Diagonal reference line
auc_value <- auc(roc_result)

cat("AUC Value:", auc_value, "\n")

print(knn_model$bestTune)

predictions <- predict(knn_model, newdata = test_set, type = "raw")


cm=confusionMatrix(as.factor(predictions), as.factor(test_set$Class))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures

#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part F, Subsection 2: RPART WITH SMOTE IG'

train_data <- SMOTE_IG
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(train_data))]

train_control <- trainControl(method = "cv", number = 10, search = "grid")
tune_grid <- expand.grid(.k = seq(3, 21, 2))  # k values from 3 to 21, odd numbers to avoid ties

set.seed(123)
knn_model <- train(Class ~ ., data = train_data, method = "knn",
                   trControl = train_control, tuneGrid = tune_grid,
                   preProcess = c("center", "scale"))  # Preprocessing: standardize variables

plot(knn_model)
prob_predictions <- predict(knn_model, newdata = test_set, type = "prob")
roc_result <- roc(response = test_set$Class, predictor = prob_predictions[,2])
plot(roc_result, main="ROC Curve for KNN Model SMOTE IG")
abline(a=0, b=1, lty=2, col="gray")  # Diagonal reference line
auc_value <- auc(roc_result)

cat("AUC Value:", auc_value, "\n")

print(knn_model$bestTune)

predictions <- predict(knn_model, newdata = test_set, type = "raw")


cm=confusionMatrix(as.factor(predictions), as.factor(test_set$Class))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures


#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part F, Subsection 3: RPART WITH SMOTE RFE'

train_data <- SMOTE_RFE
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(train_data))]
train_control <- trainControl(method = "cv", number = 10, search = "grid")
tune_grid <- expand.grid(.k = seq(3, 21, 2))  # k values from 3 to 21, odd numbers to avoid ties

set.seed(123)
knn_model <- train(Class ~ ., data = train_data, method = "knn",
                   trControl = train_control, tuneGrid = tune_grid,
                   preProcess = c("center", "scale"))  # Preprocessing: standardize variables

plot(knn_model)# Assuming class probabilities are needed for ROC curve
prob_predictions <- predict(knn_model, newdata = test_set, type = "prob")
roc_result <- roc(response = test_set$Class, predictor = prob_predictions[,2])
plot(roc_result, main="ROC Curve for KNN Model SMOTE RFE")
abline(a=0, b=1, lty=2, col="gray")  # Diagonal reference line
auc_value <- auc(roc_result)

cat("AUC Value:", auc_value, "\n")

print(knn_model$bestTune)

predictions <- predict(knn_model, newdata = test_set, type = "raw")

cm=confusionMatrix(as.factor(predictions), as.factor(test_set$Class))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures

#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part F, Subsection 4: RPART WITH BOTH STEPWISE'

train_data <- BOTH_STEPWISE
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(train_data))]

train_control <- trainControl(method = "cv", number = 10, search = "grid")
tune_grid <- expand.grid(.k = seq(3, 21, 2))  # k values from 3 to 21, odd numbers to avoid ties

set.seed(123)
knn_model <- train(Class ~ ., data = train_data, method = "knn",
                   trControl = train_control, tuneGrid = tune_grid,
                   preProcess = c("center", "scale"))  # Preprocessing: standardize variables

plot(knn_model)# Assuming class probabilities are needed for ROC curve
prob_predictions <- predict(knn_model, newdata = test_set, type = "prob")
roc_result <- roc(response = test_set$Class, predictor = prob_predictions[,2])
plot(roc_result, main="ROC Curve for KNN Model BOTH STEPWISE")
abline(a=0, b=1, lty=2, col="gray")  # Diagonal reference line
auc_value <- auc(roc_result)

cat("AUC Value:", auc_value, "\n")

print(knn_model$bestTune)

predictions <- predict(knn_model, newdata = test_set, type = "raw")


cm=confusionMatrix(as.factor(predictions), as.factor(test_set$Class))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures

#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part F, Subsection 5: RPART WITH BOTH IG'

train_data <- BOTH_IG
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(train_data))]
train_control <- trainControl(method = "cv", number = 10, search = "grid")
tune_grid <- expand.grid(.k = seq(3, 21, 2))  # k values from 3 to 21, odd numbers to avoid ties

set.seed(123)
knn_model <- train(Class ~ ., data = train_data, method = "knn",
                   trControl = train_control, tuneGrid = tune_grid,
                   preProcess = c("center", "scale"))  # Preprocessing: standardize variables

plot(knn_model)# Assuming class probabilities are needed for ROC curve
prob_predictions <- predict(knn_model, newdata = test_set, type = "prob")
roc_result <- roc(response = test_set$Class, predictor = prob_predictions[,2])
plot(roc_result, main="ROC Curve for KNN Model BOTH IG")
abline(a=0, b=1, lty=2, col="gray")  # Diagonal reference line
auc_value <- auc(roc_result)

cat("AUC Value:", auc_value, "\n")

print(knn_model$bestTune)

predictions <- predict(knn_model, newdata = test_set, type = "raw")


cm=confusionMatrix(as.factor(predictions), as.factor(test_set$Class))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures

#---------------------------------------------------------------------------------------------------------------------------
'Module 3, Part F, Subsection 6: RPART WITH BOTH RFE'

train_data <- BOTH_RFE
test_set=read.csv('initial_test.csv')
test_set=test_set[,c(colnames(train_data))]

train_control <- trainControl(method = "cv", number = 10, search = "grid")
tune_grid <- expand.grid(.k = seq(3, 21, 2))  # k values from 3 to 21, odd numbers to avoid ties

set.seed(123)
knn_model <- train(Class ~ ., data = train_data, method = "knn",
                   trControl = train_control, tuneGrid = tune_grid,
                   preProcess = c("center", "scale"))  # Preprocessing: standardize variables

plot(knn_model)# Assuming class probabilities are needed for ROC curve
prob_predictions <- predict(knn_model, newdata = test_set, type = "prob")
roc_result <- roc(response = test_set$Class, predictor = prob_predictions[,2])
plot(roc_result, main="ROC Curve for KNN Model BOTH RFE")
abline(a=0, b=1, lty=2, col="gray")  # Diagonal reference line
auc_value <- auc(roc_result)

cat("AUC Value:", auc_value, "\n")

print(knn_model$bestTune)

predictions <- predict(knn_model, newdata = test_set, type = "raw")


cm=confusionMatrix(as.factor(predictions), as.factor(test_set$Class))
cm
cm <- cm$table
tp = cm[1,1]
fp = cm[1,2]
tn = cm[2,2]
fn = cm[2,1]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 0")
performance_measures
tp = cm[2,2]
fp = cm[2,1]
tn = cm[1,1]
fn = cm[1,2]
performance_measures = calculate_measures(tp, fp, tn, fn)
print("Class 1")
performance_measures

#----------------------------------------------------------------------------------------------------------------------------------------------------







