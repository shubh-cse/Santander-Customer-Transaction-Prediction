#################################Santander Customer Transaction Prediction###########################################

#remove all the objects stored
rm(list=ls())

#set working directory
setwd("C:/Users/Shubh Gupta/Desktop/DataScience/Project/Santander-Customer-Transaction-Prediction")

#check working directory
getwd()

#importing libraries
library(gridExtra)
library(grid)
library(ggplot2)
library(lattice)
library(usdm)
library(pROC)
library(caret)
library(rpart)
library(DataCombine)
library(ROSE)
library(e1071)
library(xgboost)

#loading the train data
train=read.csv('train.csv')

#loading the test data 
test=read.csv('test.csv')

#size of train data
dim(train)

#size of test data
dim(test)

#describing the train data
summary(train)

#describing the test data
summary(test)

#exploring the train data
head(train)

#exploring the test data
head(test)

#storing ID_code of train data into another variable
train_ID_code=train$ID_code

#storing ID_code of test data into another variable
test_ID_code=test$ID_code

#dropping ID_code from train data
train$ID_code=NULL

#dropping ID_code from test data
test$ID_code=NULL

#checking the size of train data again after dropping ID_code
dim(train)

#checking the size of test data again after dropping ID_code
dim(test)

#count of both the classes in the target variable
table(train$target)

###########################################Observation#########################################################
#It is an imbalanced dataset.
#The frequency of the people making a transaction is much less than the people who will not make the transaction.

########################################Missing Value Analysis###################################################

#function to calculate the missing values
findMissingValue =function(df)
{
  missing_val =data.frame(apply(df,2,function(x){sum(is.na(x))}))
  missing_val$Columns = row.names(missing_val)
  names(missing_val)[1] =  "Missing_percentage"
  missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train)) * 100
  missing_val = missing_val[order(-missing_val$Missing_percentage),]
  row.names(missing_val) = NULL
  missing_val = missing_val[,c(2,1)]
  return (missing_val)
}

#checking missing value in train data
findMissingValue(train)

#checking missing value in test data
findMissingValue(test)

###########################################Observation#########################################################
#No missing values in the train and test data

#splitting the dependent and independent variables from the train dataset
independent_var=(colnames(train)!='target')
X=train[,independent_var]
Y=train$target

#checking the shape of independent variables
dim(X)

#checking the shape of dependent variable
dim(Y)

###########################################Multicollinearity Analysis###########################################

#checking the correlation between independent variables
cor=vifcor(X)
print(cor)

###########################################Observation#########################################################
#No variable from the 200 input variables has collinearity problem.

####################################Distribution of Independent Variables#####################################

#function to check the distribution of independent variables
plot_distribution=function(X)
{
  variblename =colnames(X)
  temp=1
  for(i in seq(10,dim(X)[2],10))
  {
    plot_helper(temp,i ,variblename)
    temp=i+1
  }
}
plot_helper=function(start ,stop, variblename)
{ 
  par(mar=c(2,2,2,2))
  par(mfrow=c(4,3))
  for (i in variblename[start:stop])
  {
    plot(density(X[[i]]) ,main=i )
  }
}

#distribution of train data
plot_distribution(X)

#distribution of test data
plot_distribution(test)

###########################################Observation#########################################################
#Both the train and test data are almost normally distributed.

############################################Outlier Analysis###################################################

#function to plot a boxplot for checking outliers in the dataset
plot_boxplot=function(X)
{
  variblename =colnames(X)
  temp=1
  for(i in seq(10,dim(X)[2],10))
  {
    plot_helper(temp,i ,variblename)
    temp=i+1
  }
}
plot_helper=function(start ,stop, variblename)
{ 
  par(mar=c(2,2,2,2))
  par(mfrow=c(4,3))
  for (i in variblename[start:stop])
  {
    boxplot(X[[i]] ,main=i)
  }
}

#outlier analysis on the train data
plot_boxplot(X)

#outlier analysis on the test data
plot_boxplot(test)

###########################################Observation#########################################################
#data points below lower fense and above upper fense will be declared as outliers

############################################Replacing outliers with NA#########################################

#function to replace outliers in the dataset with NA
fill_outlier_with_na=function(df)
{
  cnames=colnames(df)
  for(i in cnames)
  {
    val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
    df[,i][df[,i] %in% val] = NA
  }
  return (df)
}

#replacing outliers from the train data
X=fill_outlier_with_na(X)

#count of ouliers in the train data
sum(is.na(X))

#replacing outliers from the test data
test=fill_outlier_with_na(test)

#count of ouliers in the test data
sum(is.na(test))

###########################################Observation#########################################################
#Total number of ouliers in the train data are 26533
#Total number of outliers in the test data are 27087

###########################################Mean Imputation#######################################################

#function to replace NA with the mean of that particular variable
fill_outlier_with_mean=function(df)
{
  cnames=colnames(df)
  for(i in cnames)
  {
    
    df[is.na(df[,i]), i] <- mean(df[,i], na.rm = TRUE)
  }
  return (df)
}

#mean imputation in the train data
X=fill_outlier_with_mean(X)

#count of NA values in the train data after imputation
sum(is.na(X))

#mean imputation in the test data
test=fill_outlier_with_mean(test)

#count of NA values in the test data after imputation
sum(is.na(test))

###########################################Observation#########################################################
#Total number of NA values in both train and test data after mean imputation is 0

###########################################Standardization#######################################################

#standardization is done to scale all the variables in the same range
#standardization=(x-mean(x)/sd(x)

#function to perform standardization
standardization=function(df)
{
  cnames =colnames(df)
  for( i in   cnames ){
    df[,i]=(df[,i] -mean(df[,i] ,na.rm=T))/sd(df[,i])
  }
  return(df)
}

#standardization of train data
X=standardization(X)

#standardization of test data
test=standardization(test)

#combining dependent and independent variables
std_train=cbind(X,Y)

#splitting the dataset into train and test
#70% data for training and 30% data for testing
set.seed(123)
train.index=createDataPartition(std_train$Y , p=.70 ,list=FALSE)
train=std_train[train.index,]
test=std_train[-train.index,]

#shape of train data after splitting
dim(train)

#shape of test data after splitting
dim(test)

#making the imbalanced dataset balanced by keeping both the target classes in equal ratios of 50:50
over_sample=ovun.sample(Y~.,data =train,method='over' )$data

##########################################Model Training#######################################################
#We will use three models for training the dataset :-
#1. Logistic Regression
#2. Decision Tree
#3. XGBoost

###############################Generic function to calculate various classification metrics#####################

#function to check performance of classification models
metric_fun=function(conf_matrix)
{
  model_parm =list()
  tpr=conf_matrix[1,1]
  fnr=conf_matrix[1,2]
  fpr=conf_matrix[2,1]
  tnr=conf_matrix[2,2]
  p=(tpr)/(tpr+fpr)
  r=(tpr)/(tpr+fnr)
  s=(tnr)/(tnr+fpr)
  f1=2*((p*r)/(p+r))
  print(paste("accuracy",round((tpr+tnr)/(tpr+tnr+fpr+fnr),2)))
  print(paste("precision",round(p ,2)))
  print(paste("recall",round(r,2)))
  print(paste("specificity",round(s,2)))
  print(paste("fpr",round((fpr)/(fpr+tnr),2)))
  print(paste("fnr",round((fnr)/(fnr+tpr),2)))
  print(paste("f1",round(f1,2)))
}

#############################################Logistic Regression################################################

logit=glm(formula=Y~.,data=over_sample,family='binomial')
summary(logit)
y_prob=predict(logit,test[-201],type='response')
y_pred=ifelse(y_prob >0.5, 1, 0)
conf_matrix= table(test[,201],y_pred)
metric_fun(conf_matrix)
roc=roc(test[,201],y_prob)
print(roc)
plot(roc ,main="Logistic Regression roc-auc curve")

###################################################Observation#################################################
# Accuracy --> 78%
# Precision --> 97%
# Recall --> 78%
# Specificity --> 77%
# FPR --> 23%
# FNR --> 22%
# F1 Score --> 87%
# AUC --> 85%

#############################################Decision Tree#####################################################

rm_model=rpart(Y~.,data=over_sample)
summary(rm_model)
y_prob=predict(rm_model,test[-201])
y_pred=ifelse(y_prob>0.5,1,0)
conf_matrix=table(test[,201],y_pred)
metric_fun(conf_matrix)
roc=roc(test[,201],y_prob )
print(roc)
plot(roc,main="Decision Tree roc-auc curve")

###################################################Observation#################################################
# Accuracy --> 62%
# Precision --> 93%
# Recall --> 63%
# Specificity --> 54%
# FPR --> 46%
# FNR --> 37%
# F1 Score --> 75%
# AUC --> 59%

#############################################XGBoost###########################################################

xgb = xgb.train(data = over_sample, max.depth = 5, eta = 1, nrounds = 500)
y_prob =predict(xgb,as.matrix(test[,-201]))
y_pred = ifelse(y_prob >0.5, 1, 0)
conf_matrix= table(test[,201] , y_pred)
metric_fun(conf_matrix)
roc=roc(test[,201], y_prob )
print(roc)
plot(roc ,main="XGBoost roc-auc curve")

###################################################Observation#################################################
# Accuracy --> 78%
# Precision --> 97%
# Recall --> 78%
# Specificity --> 77%
# FPR --> 23%
# FNR --> 22%
# F1 Score --> 87%
# AUC --> 85%

##################################################Model Selection#############################################
#We will select final model based on following parameters :-
#1. High Accuracy
#2. High F1 Score
#3. High AUC
#4. Low FPR
#5. Low FNR

#################################################Freezed Model################################################
#We will freeze XGBoost as our final model based on the above parameters.