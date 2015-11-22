
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


# rfmodFit2 <- randomForest(formula = classe ~ ., data = training_data) 
# rfmodFit2


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






###################





# You should create a report describing how you built your model, 



# how you used cross validation, 



# what you think the expected out of sample error is, and 



# why you made the choices you did. 



# You will also use your prediction model to predict 20 different test cases














library(AppliedPredictiveModeling)
library(caret)
data(AlzheimerDisease)
#summary(AlzheimerDisease)

adData = data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis, p = 0.50,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]


# Q2
# Load the cement data using the commands:

library(AppliedPredictiveModeling)
data(concrete)

library(caret)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

# Make a histogram and confirm the SuperPlasticizer variable is skewed
hist(training$Superplasticizer, main="", xlab="Super Plasticizer")

### The log transform produces negative values which can not be used by some classifiers.


# Q3
# Load the Alzheimer's disease data using the commands:

library(caret)
library(AppliedPredictiveModeling)
set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

# Find all the predictor variables in the training set that begin with IL. 
str(training) # thre are 251 obs and 131 var

ILpv <- grep("^IL", colnames(training), value=TRUE)
head(ILpv)

# Perform principal components on these variables with the preProcess() function from the caret package. 
# Calculate the number of principal components needed to capture 80% of the variance

preProc <- preProcess(training[, ILpv], method = "pca", thresh = 0.8)
preProc$rotation # 7 pc

# Q4:
# Create a training data set consisting of only the predictors with variable names beginning with IL and the diagnosis. 
# Build two predictive models, one using the predictors as they are and one using PCA with principal components 
# explaining 80% of the variance in the predictors. Use method="glm" in the train function. 
# What is the accuracy of each method in the test set? Which is more accurate?

# Load the Alzheimer's disease data using the commands:
library(caret)
library(AppliedPredictiveModeling)

set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]


set.seed(3433)
# grep predictors starting with IL
ILpv <- grep("^IL", colnames(training), value=TRUE)

# Subset predictors starting with IL
predictors_IL <- predictors[, ILpv]

# Create a training data set consisting of only IL predictors and diagnosis
ad_Data = data.frame(predictors_IL,diagnosis)
inTrain = createDataPartition(ad_Data$diagnosis, p = 3/4)[[1]]
training = ad_Data[ inTrain,]
testing = ad_Data[-inTrain,]

# train data using the predictors as they are 
modelFit_Non_PCA <- train(diagnosis ~ .,method="glm",data=training)

predictions_Non_PCA <- predict(modelFit_Non_PCA, newdata = testing)

CM_Non_PCA <- confusionMatrix(predictions_Non_PCA,testing$diagnosis)
print(CM_Non_PCA) # Accuracy : 0.6463  

# train data using one using PCA with principal components 
# explaining 80% of the variance
preProc <- preProcess(training, method = "pca", thresh = 0.8)
trainPCA <- predict(preProc, training)
modelFit_PCA <- train(diagnosis ~ .,method="glm",data=trainPCA)

testPCA <- predict(preProc,testing)
CM_PCA <- confusionMatrix(testing$diagnosis,predict(modelFit_PCA,testPCA))
print(CM_PCA) # Accuracy : 0.7195














library(ISLR); library(ggplot2); library(caret);
data(Wage); Wage <- subset(Wage,select=-c(logwage))
summary(Wage)

# Get training/test sets

inTrain <- createDataPartition(y=Wage$wage,
                               p=0.7, list=FALSE)
training <- Wage[inTrain,]; testing <- Wage[-inTrain,]
dim(training); dim(testing)

# Feature plot

featurePlot(x=training[,c("age","education","jobclass")],
            y = training$wage,
            plot="pairs")

# Plot age versus wage
qplot(age,wage,data=training)

# Plot age versus wage colour by jobclass
qplot(age,wage,colour=jobclass,data=training)


# Plot age versus wage colour by education
qplot(age,wage,colour=education,data=training)


# Fit a linear model
modFit<- train(wage ~ age + jobclass + education,
               method = "lm",data=training)
finMod <- modFit$finalModel
print(modFit)

# Diagnostics
plot(finMod,1,pch=19,cex=0.5,col="#00000010")

# Color by variables not used in the model
qplot(finMod$fitted,finMod$residuals,colour=race,data=training)

# Plot by index
plot(finMod$residuals,pch=19)


# Predicted versus truth in test set
pred <- predict(modFit, testing)
qplot(wage,pred,colour=year,data=testing)

# If you want to use all covariates
modFitAll<- train(wage ~ .,data=training,method="lm")
pred <- predict(modFitAll, testing)
qplot(wage,pred,data=testing)








