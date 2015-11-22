
####################################################################
## Pratical Machine Learning Project, PChheng
## Goal:  predict the manner in which they did the exercise
####################################################################

# Load dataset:

setwd("\\Users\\pxc233\\Documents\\Phalkun\\Classes\\Coursera\\Practical Machine Learning\\Project\\Data")

pml_training <- read.csv("pml-training.csv")
pml_testing <- read.csv("pml-testing.csv")

# Checking train dataset:
str(pml_training) # 19622 obs. of  160 variables
summary(pml_training) 

table(pml_training$classe)

# A    B    C    D    E 
# 5580 3797 3422 3216 3607 

# Checking test dataset:
str(pml_testing) # 20 obs. of  160 variables
summary(pml_testing) 

########################################

library(Hmisc) # lots of uses
describe(pml_training)
View(pml_training)
names(pml_training)
contents(pml_training)

########################################

# Data Preprocesssing
#######################

# Checking variables with complete obs:

train_comp_obs <- sum(complete.cases(pml_training))
train_comp_obs # 406 obs


test_comp_obs <- sum(complete.cases(pml_testing))
test_comp_obs # 0 obs


# Removing var or colunm with NA from training dataset:

pml_training <- pml_training[, colSums(is.na(pml_training)) == 0]
str(pml_training) # 19,622 obs. of  93 variables
names(pml_training)
summary(pml_training) # No NA

# Removing var that are not important from training dataset:

train_remove <- grepl("^X|timestamp|window", names(pml_training))
summary(train_remove) # 6 more var removed

pml_training <- pml_training[, !train_remove]
str(pml_training) # 19622 obs. of  87 variables
names(pml_training)

# Subset only numeric var in the training dataset
pml_training_final <- pml_training[, sapply(pml_training, is.numeric)]
str(pml_training_final) # 19622 obs. of  52 variables:

# Since the required 'classe' var was removed, we add it back to the final training daset
pml_training_final$classe <- pml_training$classe
str(pml_training_final) # 19622 obs. of  53 variables

names(pml_training_final) # 53 variables
summary(pml_training_final)
table(pml_training_final$classe)

#
# A    B    C    D    E 
# 5580 3797 3422 3216 3607 



# Slicing the data
###################

set.seed(32343) # to get the same resampled numbers generated
library(caret)

# create training set with with 75% of data and validation set 
inTrain <- createDataPartition(pml_training_final$classe, p=0.75, list=FALSE)

training_data <- pml_training_final[inTrain,]
testing_data <- pml_training_final[-inTrain,]

# dimension of original training dataset and validation set
rbind("original dataset" = dim(training_data),"training set" = dim(testing_data))

#dim(training_data) # 14718 obs and 53 var
#dim(testing_data) # 4904 obs and 53 var

# Data Modeling
################

library(caret)
library(randomForest)

# cvControl <- trainControl(## 5-fold CV
#                           method="repeatedcv", 
#                           number = 5, 
#                           ## repeated 10 times
#                           repeats = 3)

cvControl <- trainControl(method="cv", number=5) 
rfmodFit <- train(classe ~ .,data=training_data, method="rf", trControl=cvControl, ntree=250)
rfmodFit


# Use the fitted model on the test set (testSA)
rf_predict <- predict(rfmodFit, testing_data)

# Use function confusionMatrix to get summary of the results of the model
CM_rf <- confusionMatrix(testing_data$classe, rf_predict)
CM_rf 

# to calculate Accuracy use postResample function
Accuracy <- postResample(rf_predict, testing_data$classe)
Accuracy # Accuracy=0.9893964

# Calculate estimated out-of-sample error (OOSE)

OOSE <- 1 - Accuracy[1]
OOSE # 0.01060359 

# or another way       
OOSE_2 <- 1- as.numeric(confusionMatrix(testing_data$classe, rf_predict)$overall[1])
OOSE_2 #  0.01060359


rfmodFit2 <- randomForest(formula = classe ~ ., data = training_data) 
rfmodFit2 # OOB = 0.46%


# Predicting on the orginal test dataset
########################################

# Removing var or colunm with NA from test dataset:

pml_testing <- pml_testing[, colSums(is.na(pml_testing)) == 0]
str(pml_testing) # 20 obs. of  60 variables
names(pml_testing)
summary(pml_testing) # No NA

# Removing var are not important from test dataset:

test_remove <- grepl("^X|timestamp|window", names(pml_testing))
summary(test_remove) # 6 more var removed

pml_testing <- pml_testing[, !test_remove]
str(pml_testing) # 20 obs. of  54 variables

pml_testing_final <- pml_testing[, sapply(pml_testing, is.numeric)]
summary(pml_testing_final)
str(pml_testing_final) # 20 obs and 53 variables
names(pml_testing_final) # 53 variables

## So there are the same 53 var or features while only 52 in traning dataset

# remove 'problem_id' var from the test dataset
pml_testing_final <- pml_testing_final[, -53]
names(pml_testing_final) # 52 var

## So there are the same 52 var or features as in traning dataset


# Using the fitted model to Predict for the orginal test dataset
predict_test <- predict(rfmodFit, pml_testing_final)
predict_test

predict_test[1:20]



# Visualization
###############

# tabulate results
table(rf_predict,testing_data$classe)

# rf_predict    A    B    C    D    E
# A 1391    4    0    0    0
# B    4  940   19    0    0
# C    0    5  834   16    0
# D    0    0    2  787    1
# E    0    0    0    1  900


# Check overall correct results of prediction by the rf algorithm 
testing_data$predRight <- rf_predict==testing_data$classe
table(testing_data$predRight)


# Decision tree plot
library(rpart)
library(rpart.plot)
treeplot_model <- rpart(classe ~ ., data=training_data, method="class")
prp(treeplot_model,
    box.col=c("red", "lightblue", "yellow", "pink", "palegreen3")[treeplot_model$frame$yval])



# Creating files for Prediction Assignment Submission
######################################################

answers <- as.vector(predict_test) 

pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}



pml_write_files(answers)



#########END###########



