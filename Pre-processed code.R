# Dropping the columns 
library(caret)
dataset=read.csv('project_dataset_5K.csv')

# Dropping the hidden values 
hidden_columns <- c("IDATE", "SEQNO", "X_PSU", "X_STSTR", "X_STRWT", "X_RAWRAKE", "X_WT2RAKE", "X_CLLCPWT", "X_DUALCOR", "X_LLCPWT2", "X_LLCPWT'") # Replace these names with actual column names marked as "Hidden"
dataset_cleaned <- dataset[ , !(names(dataset) %in% hidden_columns)]

# Dropping columns with 0 dataentry 
dataset_cleaned <- dataset_cleaned[, colSums(is.na(dataset_cleaned)) < nrow(dataset_cleaned)]

# Dropping columns with 0 variance
variances <- sapply(dataset_cleaned, function(x) if(all(is.na(x))) NA else var(x, na.rm = TRUE))
valid_columns <- names(variances[!is.na(variances) & variances > 0])
dataset_cleaned <- dataset_cleaned[, valid_columns]

# Dropping Columns with near 0 variance
X=nearZeroVar(dataset_cleaned, freqCut = 80/5, names = TRUE)
dataset_cleaned <- dataset_cleaned[ , !(names(dataset_cleaned) %in% X)]

'''Dropping columns with misisng values threshold'''

total_missing_values <- sum(is.na(dataset_cleaned))
avg_missing_value_per_attribute=total_missing_values/222
threshold=(avg_missing_value_per_attribute/5000)*100
missing_percentage_threshold <- threshold

threshold_count <- nrow(dataset_cleaned) * (missing_percentage_threshold / 100)
columns_to_drop <- sapply(dataset_cleaned, function(x) sum(is.na(x)) > threshold_count)
dataset_cleaned <- dataset_cleaned[, !columns_to_drop]

'''Cleaning further unwanted columns'''

unwanted_columns <- c("FMONTH","IMONTH","IDAY","IYEAR","DISPCODE","CSTATE1","LANDLINE") # Replace these names with actual column names marked as "Hidden"
dataset_cleaned <- dataset_cleaned[ , !(names(dataset_cleaned) %in% unwanted_columns)]
write.csv(dataset_cleaned, "dataset_cleaned_1_threshold.csv")
