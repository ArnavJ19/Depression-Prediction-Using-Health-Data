library(DMwR)
library(caret) #For hyperparameter optimization
data=read.csv('initial_train.csv')
data$Class <- as.factor(data$Class)
barplot(prop.table(table(data$Class)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution")
##############################################################################################################################################
# Hyperparameter Optimization
evaluate_smote <- function(k) {
  smote_data <- SMOTE(Class ~ ., data = data, k = k, perc.over = 100)
  train_control <- trainControl(method="cv", number=10)
  model <- train(x=smote_data[,1:100], y=smote_data$Class, method="lvq", trControl=train_control)
  return(mean(model$resample$Accuracy))
}
# Grid search setup
k_values <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) # K values to test
# Perform grid search
results <- expand.grid(k = k_values)
results$accuracy <- apply(results, 1, function(params) evaluate_smote(params['k']))
# View results
print(results)
# Find the best parameters
plot(results)
best_parameters <- results[which.max(results$accuracy),]
print(best_parameters)
#################################################################################################################

# Applying SMOTE WITH BEST PARAMETERS
data_smote <- SMOTE(Class ~ ., data = data, perc.over = 260, perc.under=150, k = best_parameters$k)
barplot(prop.table(table(data_smote$Class)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution after SMOTE Data Balancing")
prop.table(table(data_smote$Class))
################################################################################################################################################

#Saving the dataset
write.csv(data_smote, "data_SMOTE.csv", row.names = FALSE)
