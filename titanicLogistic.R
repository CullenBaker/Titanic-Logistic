library(stringr)
library(mice)
library(miceadds)
library(lmtest)
library(ResourceSelection)
library(ROCR)
library(boot)

# Import training data
data.train <- read.csv('train.csv', header=TRUE)
data.test <- read.csv('test.csv', header=TRUE)
attach(data.train)
# Data cleaning

# Consider Parch and Sibsp as categorical? 1 if > 0, 0 otherwise??

# Ensure Categorical variables are being treated as such
is.factor(data.train$Survived) # False
data.train$Survived <- factor(data.train$Survived)
is.factor(data.train$Pclass) # False
data.train$Pclass <- factor(data.train$Pclass)
data.test$Pclass <- factor(data.test$Pclass)
is.factor(data.train$Sex) # True
is.factor(data.train$Embarked) # True

# See distribution of the Embarked variable
table(data.train$Embarked)
# There are two empty cells. For the sake of simplicity we will set them to 'S', which contains 72.4%
# of the observations for this variable
data.train$Embarked <- sub("^$", "S", data.train$Embarked)
data.train$Embarked <- factor(data.train$Embarked)
data.train$Embarked <- relevel(data.train$Embarked, ref = "S")
data.test$Embarked <- relevel(data.test$Embarked, ref = "S")
contrasts(data.train$Embarked)

# See distribution of the Cabin variable
table(data.train$Cabin)
# This looks awful. Let's group by level instead

Cab2Level <- function(x) {
  if ((x == "") | (str_sub(x, 1, 1) == "T") | (str_sub(x, 1, 1) == "G")) {
    return("N") }
  else {
    return(str_sub(x, 1, 1))}
}

data.train$Level <- sapply(data.train$Cabin, Cab2Level)
data.test$Level <- sapply(data.test$Cabin, Cab2Level)
table(data.train$Level) # Much better
is.factor(data.train$Level) # False
data.train$Level <- factor(data.train$Level)
data.test$Level <- factor(data.test$Level)
data.train$Level <- relevel(data.train$Level, ref = "N")
data.test$Level <- relevel(data.test$Level, ref = "N")
contrasts(data.train$Level)

# 0-25, 26-50, 51-80
#Age2Level <- function(x) {
#  if (is.na(x)) {
 #   return(NA)
  #}
  #else if (x < 25) {
   # return("Young") 
  #}
  #else if (x > 25 & x <= 50) {
   # return("Middle")
  #}
  #else {
  #  return("Old")
  #}
#}

#data.train$Old <- sapply(data.train$Age, Age2Level)
#data.test$Old <- sapply(data.test$Age, Age2Level)
#table(data.train$Old) # Much better
#is.factor(data.train$Old) # False
#data.train$Old <- factor(data.train$Old)
#data.test$Old <- factor(data.test$Old)
#data.train$Old <- relevel(data.train$Old, ref = "Middle")
#data.test$Old <- relevel(data.test$Old, ref = "Middle")
#contrasts(data.train$Old)

# Find columns with NA values
colSums(is.na(data.train))
colSums(is.na(data.test))
# Age has 177 NAs (86 in test), but every other column has none. Test also has 1 NA for fare. Because Fare
# will eventually be deleted, we ignore this and set the one NA value to the mean Fare.
data.test$Fare[is.na(data.test$Fare)] <- round(mean(data.test$Fare, na.rm = TRUE))

# Upon further analysis of the data frame, the column 'Cabin' has numerous missing values

# Delete useless columns from dataset
#data.train$PassengerId <- NULL 
data.train$Name <- NULL 
data.train$Ticket <- NULL 
data.train$Cabin <- NULL

#data.test$PassengerId <- NULL 
data.test$Name <- NULL 
data.test$Ticket <- NULL 
data.test$Cabin <- NULL

# Data imputation/dealing with missing values. Options: 1) Remove rows with NA values and create a model
# based upon this, and then remove columns with missing values and create a model based on this dataset, 
# then use a weighted voting scheme for the test set. 2) Use random imputation or another method of 
# imputation 3) Use an ensemble  of methods from both 1) and 2). Make sure to observe their effectiveness 
# for the situation where age is known, and the situation where age is unknown


# Create first two datasets
data.train.1 <- na.omit(data.train)
data.train.1$Level <- relevel(data.train.1$Level, ref = "N")
data.test.1 <- na.omit(data.test)
data.test.1$Level <- relevel(data.test.1$Level, ref = "N")

data.train.2 <- data.train
data.train.2$Level <- relevel(data.train.2$Level, ref = "N")
data.train.2$Age <- NULL 


detach(data.train)

# Function to perform univariate Logit Regression analysis on the varaibles of the dataset
PartialLogit <- function(dataset, y) {
  df <- data.frame("Variable" = character(1), "Coefficient" = 0, "Standard Error" = 0, "P-value" = 0, stringsAsFactors=FALSE)
  yvar <- names(dataset)[y]
  for (i in names(dataset)[-y]) {
    form <- paste(yvar,'~',i)
    result <- glm(form, data = dataset, family = "binomial")
    summ <- summary(result)
    n <- dim(summ$coefficients)[1]
    for (j in 2:n) {
      k <- row.names(summ$coefficients)[j]
      cf <- round(summ$coefficients[j, 1], 4)
      se <- round(summ$coefficients[j, 2], 4)
      pval <- round(summ$coefficients[j, 4], 6)
      df <- rbind(df, setNames(as.list(c(as.character(k), cf, se, pval)), names(df)))
      }
  }
  return(df[-1,])
}

# Model Buliding and variable selection
# First on data.train.1 
# perform univariate analysis
PartialLogit(data.train.1, 2)
# Clearly SibSp is the only variable that has no moderately significant relationship with Survived. Thus,
# we shall drop it. Does this mean it won't be useful with interaction terms?
# data.train.1$SibSp <- NULL #####

# Fit the full model
fullMod.1 <- glm(Survived~Pclass+Sex+Age+Parch+Fare+Embarked+Level, data = data.train.1, family = "binomial")
summary(fullMod.1)  
# Here we see Fare and Parch seem insignificant given the other variables are in the model, and Embarked
# is right on the border of significance. Let's try fitting a reduced model.

reduced.1.1 <- glm(Survived~Pclass+Sex+Age+Embarked+Level, data = data.train.1, family = "binomial")
summary(reduced.1.1) 
# Looks better. Adds significance to Embarked. Let's observe the change in coefficients

delta.coef <- abs((coef(reduced.1.1) - coef(fullMod.1)[-c(6,7)])/coef(fullMod.1)[-c(6,7)])
round(delta.coef, 3)
# Only LevelC changes over 20% (though Pclass2 is close). Because neither of these sub-variables was
# statistically significant in the first place, we conclude that the change in coefficients is within
# reason

# Next use ANOVA to test difference between the models
anova(fullMod.1, reduced.1.1, test = 'Chisq')

# Use Loglikelihood ration test to check for significant difference in the models
lrtest(fullMod.1, reduced.1.1)
# Both results suggest that there is no significant difference between the full model and the reduced

# Check for linearity of the regressor variables
attach(data.train.1)
z <- coef(reduced.1.1)[1] + coef(reduced.1.1)[2]*(as.integer(Pclass==2)) + coef(reduced.1.1)[3]*(as.integer(Pclass==3)) + coef(reduced.1.1)[4]*(as.integer(Sex=='male')) + coef(reduced.1.1)[5]*Age + coef(reduced.1.1)[6]*(as.integer(Embarked=='C')) + coef(reduced.1.1)[7]*(as.integer(Embarked=='Q')) + coef(reduced.1.1)[8]*(as.integer(Level=='A')) + coef(reduced.1.1)[9]*(as.integer(Level=='B')) + coef(reduced.1.1)[10]*(as.integer(Level=='C')) + coef(reduced.1.1)[11]*(as.integer(Level=='D')) + coef(reduced.1.1)[12]*(as.integer(Level=='E')) + coef(reduced.1.1)[13]*(as.integer(Level=='F'))
pr <- 1/(1+exp(-z))
plot(Age, log(pr/(1-pr)))
scatter.smooth(Age, log(pr/(1-pr)), cex = 0.75)
# Transforming this may help
hist(Age)
# Skewed right
detach(data.train.1)
#scatter.smooth(Pclass, log(pr/(1-pr)), cex = 0.5)
#scatter.smooth(Sex, log(pr/(1-pr)), cex = 0.5)
#scatter.smooth(Embarked, log(pr/(1-pr)), cex = 0.5)
#scatter.smooth(Level, log(pr/(1-pr)), cex = 0.5)

# Try a log transformation of Age
data.train.1$Age2 <- log(data.train.1$Age)
attach(data.train.1)
reduced.1.2 <- glm(Survived~Pclass+Sex+Age2+Embarked+Level, data = data.train.1, family = "binomial")
summary(reduced.1.2)
z2 <- coef(reduced.1.2)[1] + coef(reduced.1.2)[2]*(as.integer(Pclass==2)) + coef(reduced.1.2)[3]*(as.integer(Pclass==3)) + coef(reduced.1.2)[4]*(as.integer(Sex=='male')) + coef(reduced.1.2)[5]*Age2 + coef(reduced.1.2)[6]*(as.integer(Embarked=='C')) + coef(reduced.1.2)[7]*(as.integer(Embarked=='Q')) + coef(reduced.1.2)[8]*(as.integer(Level=='A')) + coef(reduced.1.2)[9]*(as.integer(Level=='B')) + coef(reduced.1.2)[10]*(as.integer(Level=='C')) + coef(reduced.1.2)[11]*(as.integer(Level=='D')) + coef(reduced.1.2)[12]*(as.integer(Level=='E')) + coef(reduced.1.2)[13]*(as.integer(Level=='F'))
pr2 <- 1/(1+exp(-z2))
plot(Age2, log(pr2/(1-pr2)))
scatter.smooth(Age2, log(pr2/(1-pr2)), cex = 0.75)
hist(Age2)
# Better deviance, but all the plots look worse
lrtest(reduced.1.1, reduced.1.2)
detach(data.train.1)

# Try with exponential transformation
# See https://pdfs.semanticscholar.org/0155/d248c26aafa9ce66b48ad21aa874aa7e1553.pdf for more details
# on the transformation
data.train.1$Age3 <- data.train.1$Age**(-1/4)
attach(data.train.1)
reduced.1.3 <- glm(Survived~Pclass+Sex+Age3+Embarked+Level, data = data.train.1, family = "binomial")
summary(reduced.1.3)
z3 <- coef(reduced.1.3)[1] + coef(reduced.1.3)[2]*(as.integer(Pclass==2)) + coef(reduced.1.3)[3]*(as.integer(Pclass==3)) + coef(reduced.1.3)[4]*(as.integer(Sex=='male')) + coef(reduced.1.3)[5]*Age3 + coef(reduced.1.3)[6]*(as.integer(Embarked=='C')) + coef(reduced.1.3)[7]*(as.integer(Embarked=='Q')) + coef(reduced.1.3)[8]*(as.integer(Level=='A')) + coef(reduced.1.3)[9]*(as.integer(Level=='B')) + coef(reduced.1.3)[10]*(as.integer(Level=='C')) + coef(reduced.1.3)[11]*(as.integer(Level=='D')) + coef(reduced.1.3)[12]*(as.integer(Level=='E')) + coef(reduced.1.3)[13]*(as.integer(Level=='F'))
pr3 <- 1/(1+exp(-z3))
plot(Age3, log(pr3/(1-pr3)))
scatter.smooth(Age3, log(pr3/(1-pr3)), cex = 0.2)
hist(Age3)
# Better deviance but worse looking plots
lrtest(reduced.1.1, reduced.1.3)
lrtest(reduced.1.3, reduced.1.2)


#data.train.1$Age4 <- Age**(-1)
#data.train.1$Age5 <- Age**(2)
#attach(data.train.1)
#reduced.1.4 <- glm(Survived~Pclass+Sex+Age4+Age5+Embarked+Level, data = data.train.1, family = "binomial")
#summary(reduced.1.4)
#z4 <- coef(reduced.1.4)[1] + coef(reduced.1.4)[2]*(as.integer(Pclass==2)) + coef(reduced.1.4)[3]*(as.integer(Pclass==3)) + coef(reduced.1.4)[4]*(as.integer(Sex=='male')) + coef(reduced.1.4)[5]*Age4 + coef(reduced.1.4)[6]*Age5 + coef(reduced.1.4)[7]*(as.integer(Embarked=='C')) + coef(reduced.1.4)[8]*(as.integer(Embarked=='Q')) + coef(reduced.1.4)[9]*(as.integer(Level=='A')) + coef(reduced.1.4)[10]*(as.integer(Level=='B')) + coef(reduced.1.4)[11]*(as.integer(Level=='C')) + coef(reduced.1.4)[12]*(as.integer(Level=='D')) + coef(reduced.1.4)[13]*(as.integer(Level=='E')) + coef(reduced.1.4)[14]*(as.integer(Level=='F')) + coef(reduced.1.4)[15]*(as.integer(Level=='G'))
#pr4 <- 1/(1+exp(-z4))
#plot(Age4, log(pr4/(1-pr4)))
#scatter.smooth(Age4, log(pr4/(1-pr4)))
#hist(Age4)
# Better deviance but worse looking plots
#lrtest(reduced.1.4, reduced.1.2)



# Check interaction terms
reduced.1.4 <- glm(Survived~Pclass+Sex+Age+Embarked+Level+Age2:Pclass+Age:Sex+Age:Embarked+Age:Level, data = data.train.1, family = "binomial")
summary(reduced.1.4)
# Res Dev: 581.50

reduced.1.5 <- glm(Survived~Pclass+Sex+Age2+Embarked+Level+Age2:Pclass+Age2:Sex+Age2:Embarked+Age2:Level, data = data.train.1, family = "binomial")
summary(reduced.1.5)
# Res Dev: 579.77
# difference in deviance from normal Age subside when used with interaction terms

reduced.1.6 <- glm(Survived~Pclass+Sex+Age3+Embarked+Level+Age3:Pclass+Age3:Sex+Age3:Embarked+Age3:Level, data = data.train.1, family = "binomial")
summary(reduced.1.6)
# Res Dev: 579.58
# Lowest Deviance

# Add more interactions
reduced.1.7 <- glm(Survived~Pclass*Sex*Age+Embarked+Level, data = data.train.1, family = "binomial")
summary(reduced.1.7) 
# Res Dev: 582.92 - Not an improvement

reduced.1.8 <- glm(Survived~Pclass*Sex*Age2+Embarked+Level+Age2:SibSp, data = data.train.1, family = "binomial")
summary(reduced.1.8) 
lrtest(reduced.1.5, reduced.1.8)
# Res Dev: 561.91 - significant improvement over both reduced.1.4 and reduced.1.5

reduced.1.9 <- glm(Survived~Pclass*Sex*Age3+Level+Age3:SibSp, data = data.train.1, family = "binomial")
summary(reduced.1.9) 
lrtest(reduced.1.6, reduced.1.9)
# Res Dev: 551.41 - significant improvement over both reduced.1.4 and reduced.1.5
# Looks to be the optimal model

reduced.1.10 <- glm(Survived~Pclass*Sex*Age3+Embarked+Level+Age3:SibSp, data = data.train.1, family = "binomial")
summary(reduced.1.10)
lrtest(reduced.1.10, reduced.1.9)
# Proves Embarked can be dropped

reduced.1.11 <- glm(Survived~Pclass*Sex*Age3+Level+Age3:SibSp+SibSp, data = data.train.1, family = "binomial")
summary(reduced.1.11) 
lrtest(reduced.1.11, reduced.1.9)
# Res Dev: 547.51 - significant improvement over all other models
# Test Final model for linearity assumptions?


# Assess the fit of the model
hoslem.test(reduced.1.9$y, fitted(reduced.1.9))
hoslem.test(reduced.1.11$y, fitted(reduced.1.11))
# The p-values are 0.73 and 0.12 respectively, indicating that there is no significant difference 
# between the observed and predicted values

PredProb <- predict(reduced.1.11, type = "response")
plot(PredProb, jitter(as.numeric(Survived), 0.5), cex = 0.5)
# rocplot(reduced.1.9)
histogram(~PredProb|Survived)

# Cost function for binary response variable
cost <- function(r, pi = 0) mean(abs(r-pi) > 0.5)

# Use K-folds cross validation to see how the model performs on unseen data
cv.err<-cv.glm(data.train.1, reduced.1.9)
# Leave one out method
cv.err$delta[1] 
# K = 10 method
cv.glm(data.train.1, reduced.1.9, K=10)$delta[1] 
cv.glm(data.train.1, reduced.1.9, K=10, cost = cost)$delta[1] 

# Compare with reduced.1.11
cv.err2<-cv.glm(data.train.1, reduced.1.11)
# Leave one out method
cv.err2$delta[1] 
# K = 10 method
cv.glm(data.train.1, reduced.1.11, K=10)$delta[1]
cv.glm(data.train.1, reduced.1.11, K=10, cost = cost)$delta[1] 

# Create ROC curve
sample <- sample.int(nrow(data.train.1), floor(.80*nrow(data.train.1)), replace = F)
samp.train <- data.train.1[sample, ]
samp.test <- data.train.1[-sample, ]

# reduced.1.9
reduced.roc.1 <- glm(Survived~Pclass*Sex*Age3+Level+Age3:SibSp, data = samp.train, family = "binomial")
summary(reduced.roc.1)

# reduced.1.11
reduced.roc.2 <- glm(Survived~Pclass*Sex*Age3+Level+Age3:SibSp+SibSp, data = samp.train, family = "binomial")
summary(reduced.roc.2)

# reduced.1.8
reduced.roc.3 <- glm(Survived~Pclass*Sex*Age2+Embarked+Level+Age2:SibSp, data = samp.train, family = "binomial")
summary(reduced.roc.3)

ROCcalc <- function(mdl, test, response.index) {
  preds <- predict(mdl, newdata = test, type="response")
  rates <- prediction(preds, test[response.index])
  roc_result <- performance(rates, measure = "tpr", x.measure = "fpr")
  # ROC Plot
  rocplot <- plot(roc_result, main = "ROC Curve for Titanic")
  lines(x = c(0,1), y = c(0,1), col="red")
  # AUC
  auc <- performance(rates, measure = "auc")
  # Confusion Matrix
  ConfMat <- table(as.numeric(unlist(test[response.index]))-1, preds > 0.5)
  
  return(list(AUC = auc@y.values, ROCplot = rocplot, ConfusionMatrix = ConfMat))
}

ROCcalc(reduced.roc.1, samp.test, 2)
ROCcalc(reduced.roc.2, samp.test, 2)
ROCcalc(reduced.roc.3, samp.test, 2)


detach(data.train.1)
######################
## Data without Age ##
######################


# Model Buliding and variable selection
# On data.train.2
# perform univariate analysis
PartialLogit(data.train.2, 2)
# Clearly SibSp is the only variable that has no moderately significant relationship with Survived. Thus,
# we will only explore it when used with interaction terms
attach(data.train.2)

# Fit the full model
fullMod.2 <- glm(Survived~Pclass+Sex+Parch+Fare+Embarked+Level, data = data.train.2, family = "binomial")
summary(fullMod.2)  
# Here we see Fare and Parch seem insignificant given the other variables are in the model
#Let's try fitting a reduced model.

reduced.2.1 <- glm(Survived~Pclass+Sex+Embarked+Level, data = data.train.2, family = "binomial")
summary(reduced.2.1) 
# Looks better. Adds significance to Embarked. Let's observe the change in coefficients

delta.coef2 <- abs((coef(reduced.2.1) - coef(fullMod.2)[-c(5,6)])/coef(fullMod.2)[-c(5,6)])
round(delta.coef2, 3)
# The only significant regressor that changes over 20% is Pclass2, which is 20.9%. This is right on the
# border, but we'll use our discretion and procede with the reduced model

# Next use ANOVA to test difference between the models
anova(fullMod.2, reduced.2.1, test = 'Chisq')
# High p-value indicates that the reduced model is not subtantially different from the full model

# Use Loglikelihood ration test to check for significant difference in the models
lrtest(fullMod.2, reduced.2.1)
# Both results suggest that there is no significant difference between the full model and the reduced

# Because all regressor variables are categorical, we do not check for linearity

# Check interaction terms
reduced.2.2 <- glm(Survived~Pclass+Sex+Embarked+Level+Sex:Pclass+Sex:SibSp, data = data.train.2, family = "binomial")
summary(reduced.2.2)
lrtest(reduced.2.2, reduced.2.1)

reduced.2.3 <- glm(Survived~Pclass+Sex+Embarked+Level+Sex:Pclass+Sex:SibSp+Sex:Parch+Pclass:Parch, data = data.train.2, family = "binomial")
summary(reduced.2.3)
lrtest(reduced.2.3, reduced.2.2)
# These interactions add a significant edge to the model (I came about these two using a guess and check
# method)


# Assess the fit of the model
hoslem.test(reduced.2.3$y, fitted(reduced.2.3))
# The p-value is 0.9922, indicating that there is no significant difference between observed and 
# predicted values

PredProb2 <- predict(reduced.2.3, type = "response")
plot(PredProb2, jitter(as.numeric(Survived), 0.5), cex = 0.5)
histogram(~PredProb2|Survived)
# Plots look good, not amazing, but considerably better than random guessing


# Use K-folds cross validation to see how the model performs on unseen data
cv.err2 <- cv.glm(data.train.2, reduced.2.3)
# Leave one out method
cv.err2$delta[1] 
# K = 10 method
cv.glm(data.train.2, reduced.2.3, K=10)$delta[1] 
cv.glm(data.train.2, reduced.2.3, K=10, cost = cost)$delta[1] 

# Use K-folds cross validation to see how the model performs on unseen data
cv.err2 <- cv.glm(data.train.2, reduced.2.2)
# Leave one out method
cv.err2$delta[1] 
# K = 10 method
cv.glm(data.train.2, reduced.2.2, K=10)$delta[1] 
cv.glm(data.train.2, reduced.2.2, K=10, cost = cost)$delta[1] 

# Create ROC curve
sample2 <- sample.int(nrow(data.train.2), floor(.80*nrow(data.train.2)), replace = F)
samp.train2 <- data.train.2[sample2, ]
samp.test2 <- data.train.2[-sample2, ]

# reduced.2.3
reduced.roc.4 <- glm(Survived~Pclass+Sex+Embarked+Level+Sex:Pclass+Sex:SibSp+Sex:Parch+Pclass:Parch, data = samp.train2, family = "binomial")
summary(reduced.roc.4)
# reduced.2.2
reduced.roc.5 <- glm(Survived~Pclass+Sex+Embarked+Level+Sex:Pclass+Sex:SibSp, data = samp.train2, family = "binomial")
summary(reduced.roc.5)

ROCcalc(reduced.roc.4, samp.test2, 2)
ROCcalc(reduced.roc.5, samp.test2, 2)
# reduced.2.3 and reduced.2.2 have similar results. Potentially using an ensemble approach is best

detach(data.train.2)

###########################
## Data with Imputations ##
###########################

# Imputation with MICE using the predictive mean matching method
data.train$Age2 <- log(data.train$Age)
data.train$Age3 <- data.train$Age^(-1/4)

data.train.3.temp <- mice(data.train, m = 5, meth= c('pmm'))
predM = data.train.3.temp$predictorMatrix
predM[, c("Survived")]=0
data.train.3.temp <- mice(data.train, m = 5, meth= c('pmm'), predictorMatrix=predM)

data.train.3.temp <- mice(data.train, m = 5, meth= c('pmm'))
summary(data.train.3.temp)
# Don't allow Survived to predict Age since we won't be able to use this variable with unforeseen data
predM = data.train.3.temp$predictorMatrix
predM[, c("Survived")]=0
data.train.3.temp <- mice(data.train, m = 5, meth= c('pmm'), predictorMatrix=predM)

data.train.3.temp$imp$Age
data.train.3.temp$imp$Age2
data.train.3.temp$imp$Age3
# x <- as.numeric(as.vector(data.train.3.temp$imp$Age[1,])) to potentially use a different metric

# Graphs to ensure a similar distribution
xyplot(data.train.3.temp, Age ~ Pclass+Sex+SibSp+Parch+Fare+Embarked+Level,pch=18,cex=1)
xyplot(data.train.3.temp, Age2 ~ Pclass+Sex+SibSp+Parch+Fare+Embarked+Level,pch=18,cex=1)
xyplot(data.train.3.temp, Age3 ~ Pclass+Sex+SibSp+Parch+Fare+Embarked+Level,pch=18,cex=1)
densityplot(data.train.3.temp)

data.train.3.temp$data$Level <- relevel(data.train.3.temp$data$Level, ref = "N")
completedData1 <- complete(data.train.3.temp, 1)

attach(completedData1)

# Model Buliding and variable selection
# First on data.train.1 
# perform univariate analysis
PartialLogit(completedData1, 2)
# Clearly SibSp is the only variable that has no moderately significant relationship with Survived. Thus,
# we shall only consider it with interaction terms?

# Function to calculate average deviance of the pooled MICE object, where mod is the series of fitted
# models, and m is the number of multiple imputations
AvDev <- function(mod, m) {
  dev <- 0
  for (i in 1:m){
    dev <- dev + mod$analyses[[i]]$deviance
  }
  return(dev/m)
}

# Fit the full model
fullMod.3 <- with(data.train.3.temp, glm(Survived~ Pclass+Sex+Age+Parch+Fare+Embarked+Level+SibSp, family = "binomial"))
summary(pool(fullMod.3))  
AvDev(fullMod.3, 5)
# Here we see Fare and Parch seem insignificant given the other variables are in the model. Embarked is
# quasi-significant. Let's try fitting a reduced model.

reduced.3.1 <- with(data.train.3.temp, glm(Survived~ Pclass+Sex+Age+Embarked+Level+SibSp, family = "binomial"))
summary(pool(reduced.3.1))
AvDev(reduced.3.1, 5)
# Looks better. Adds significance to Embarked. Let's observe the change in coefficients

delta.coef3 <- abs((summary(pool(reduced.3.1))$estimate - summary(pool(fullMod.3))$estimate[-c(6,7)])/summary(pool(fullMod.3))$estimate[-c(6,7)])
round(delta.coef3, 3)
# Only Pclass2 changes over 20% . Because this sub-variables was not
# statistically significant in the first place, we conclude that the change in coefficients is within
# reason

# Use difference in deviance test to check for significant difference in the models
DiffDev <- function(modfull, modred, m, r) {
  dif <- AvDev(modred, m) - AvDev(modfull, m)
  pval <- 1 - pchisq(dif, r)
  return(pval)
}

DiffDev(fullMod.3, reduced.3.1, 5, 2)
# This suggests that there is no significant difference between the full model and the reduced model

# Logliklihood Ratio Test
fit1 = with(data = data.train.3.temp, expr = glm(Survived~ Pclass+Sex+Age+Embarked+Level+SibSp, family = "binomial"))
fit2 = with(data = data.train.3.temp, expr = glm(Survived~ Pclass+Sex+Age+Parch+Fare+Embarked+Level+SibSp, family = "binomial"))
# Wald test
stat = pool.compare(fit2, fit1, method = "likelihood")
# P-value of the test
stat$p

# Check for linearity of the regressor variables
z4 <- summary(pool(reduced.3.1))$estimate[1] + summary(pool(reduced.3.1))$estimate[2]*(as.integer(Pclass==2)) + summary(pool(reduced.3.1))$estimate[3]*(as.integer(Pclass==3)) + summary(pool(reduced.3.1))$estimate[4]*(as.integer(Sex=='male')) + summary(pool(reduced.3.1))$estimate[5]*Age + summary(pool(reduced.3.1))$estimate[6]*(as.integer(Embarked=='C')) + summary(pool(reduced.3.1))$estimate[7]*(as.integer(Embarked=='Q')) + summary(pool(reduced.3.1))$estimate[8]*(as.integer(Level=='A')) + summary(pool(reduced.3.1))$estimate[9]*(as.integer(Level=='B')) + summary(pool(reduced.3.1))$estimate[10]*(as.integer(Level=='C')) + summary(pool(reduced.3.1))$estimate[11]*(as.integer(Level=='D')) +summary(pool(reduced.3.1))$estimate[12]*(as.integer(Level=='E')) +summary(pool(reduced.3.1))$estimate[13]*(as.integer(Level=='F')) + summary(pool(reduced.3.1))$estimate[14]*SibSp
pr4 <- 1/(1+exp(-z4))
scatter.smooth(Age, log(pr4/(1-pr4)), cex = 0.2)
# Transforming this may help
hist(Age)
# Skewed right


# Try a log transformation of Age
reduced.3.2 <- with(data.train.3.temp, glm(Survived~ Pclass+Sex+Age2+Embarked+Level+SibSp, family = "binomial"))
summary(pool(reduced.3.2))
AvDev(reduced.3.2, 5)
# 763.55 - Significant improvement
z5 <- summary(pool(reduced.3.2))$estimate[1] + summary(pool(reduced.3.2))$estimate[2]*(as.integer(Pclass==2)) + summary(pool(reduced.3.2))$estimate[3]*(as.integer(Pclass==3)) + summary(pool(reduced.3.2))$estimate[4]*(as.integer(Sex=='male')) + summary(pool(reduced.3.2))$estimate[5]*Age2 + summary(pool(reduced.3.2))$estimate[6]*(as.integer(Embarked=='C')) + summary(pool(reduced.3.2))$estimate[7]*(as.integer(Embarked=='Q')) + summary(pool(reduced.3.2))$estimate[8]*(as.integer(Level=='A')) + summary(pool(reduced.3.2))$estimate[9]*(as.integer(Level=='B')) + summary(pool(reduced.3.2))$estimate[10]*(as.integer(Level=='C')) + summary(pool(reduced.3.2))$estimate[11]*(as.integer(Level=='D')) +summary(pool(reduced.3.2))$estimate[12]*(as.integer(Level=='E')) +summary(pool(reduced.3.2))$estimate[13]*(as.integer(Level=='F')) + summary(pool(reduced.3.2))$estimate[14]*SibSp
pr5 <- 1/(1+exp(-z5))
scatter.smooth(Age2, log(pr5/(1-pr5)), cex = 0.75)
# Transforming this may help
hist(Age2)


# Try a exponential transformation of Age
reduced.3.3 <- with(data.train.3.temp, glm(Survived~ Pclass+Sex+Age3+Embarked+Level+SibSp, family = "binomial"))
summary(pool(reduced.3.3))
AvDev(reduced.3.3, 5)
# 765.22 - Modest improvement
z6 <- summary(pool(reduced.3.3))$estimate[1] + summary(pool(reduced.3.3))$estimate[2]*(as.integer(Pclass==2)) + summary(pool(reduced.3.3))$estimate[3]*(as.integer(Pclass==3)) + summary(pool(reduced.3.3))$estimate[4]*(as.integer(Sex=='male')) + summary(pool(reduced.3.3))$estimate[5]*Age3 + summary(pool(reduced.3.3))$estimate[6]*(as.integer(Embarked=='C')) + summary(pool(reduced.3.3))$estimate[7]*(as.integer(Embarked=='Q')) + summary(pool(reduced.3.3))$estimate[8]*(as.integer(Level=='A')) + summary(pool(reduced.3.3))$estimate[9]*(as.integer(Level=='B')) + summary(pool(reduced.3.3))$estimate[10]*(as.integer(Level=='C')) + summary(pool(reduced.3.3))$estimate[11]*(as.integer(Level=='D')) +summary(pool(reduced.3.3))$estimate[12]*(as.integer(Level=='E')) +summary(pool(reduced.3.3))$estimate[13]*(as.integer(Level=='F')) + summary(pool(reduced.3.3))$estimate[14]*SibSp
pr6 <- 1/(1+exp(-z6))
scatter.smooth(Age3, log(pr6/(1-pr6)), cex = 0.2)
hist(Age3)


# Check interaction terms
reduced.3.4 <- with(data.train.3.temp, glm(Survived~ Pclass+Sex*Age+Embarked+Level+SibSp+Age:Parch, family = "binomial"))
summary(pool(reduced.3.4))
AvDev(reduced.3.4, 5)
# Res Dev: 758.85

reduced.3.5 <- with(data.train.3.temp, glm(Survived~ Pclass*Sex*Age2+Embarked+Level+Age:Parch+Pclass:SibSp, family = "binomial"))
summary(pool(reduced.3.5))
AvDev(reduced.3.5, 5)
# Res Dev: 703.93 - big improvement
# Lowest Deviance
DiffDev(reduced.3.5, reduced.3.2, 5, 8)
# Logliklihood Ratio Test
#stat1 = pool.compare(reduced.3.5, reduced.3.2, method = "likelihood")
# P-value of the test
#stat1$pvalue


reduced.3.6 <- with(data.train.3.temp, glm(Survived~ Pclass*Sex*Age3+Embarked+Level+SibSp:Age3+Age:Parch, family = "binomial"))
summary(pool(reduced.3.6))
AvDev(reduced.3.6, 5)
# Res Dev: 706.95

#hoslem.test(reduced.3.6$y, fitted(reduced.3.6))


#PredProb <- predict(reduced.1.9, type = "response")
#plot(PredProb, jitter(as.numeric(Survived), 0.5), cex = 0.5)
# rocplot(reduced.1.9)
#histogram(~PredProb|Survived)

#preds <- predict(reduced.1.9, newdata = data.test, type = "response")

KfoldsMICE <- function(mod, dataset, m, k, costf) {
  err1 <- 0
  err2 <- 0
  for (i in 1:m) {
    dat <- complete(dataset, i)
    err1 <- err1 + cv.glm(dat, mod, K=k)$delta[1] 
    err2 <- err2 + cv.glm(dat, mod, K=k, cost = costf)$delta[1]
  }
  return(list(MSE = err1/m, Cost10 = err2/m))
}

# reduced.3.5
KfoldsMICE(mod = glm(Survived~ Pclass*Sex*Age2+Embarked+Level+Age:Parch+Pclass:SibSp, family = "binomial"), dataset = data.train.3.temp, m = 5, k = 10, costf = cost)
# best 

# reduced.3.6
KfoldsMICE(mod = glm(Survived~ Pclass*Sex*Age3+Embarked+Level+SibSp:Age3+Age:Parch, family = "binomial"), dataset = data.train.3.temp, m = 5, k = 10, costf = cost)
# close 2nd

# reduced.3.4
KfoldsMICE(mod = glm(Survived~ Pclass+Sex*Age+Embarked+Level+SibSp+Age:Parch, family = "binomial"), dataset = data.train.3.temp, m = 5, k = 10, costf = cost)
# don't use


# See how we performed on the imputed values
sample3 <- sample.int(nrow(completedData1), floor(.85*nrow(completedData1)), replace = F)
NAs <- which(is.na(data.train$Age))
samp.train3 <- completedData1[sample3, ]
samp.test3 <- completedData1[NAs, ]

# Create ROC curve
# 3.5
reduced.roc.6 <- glm(Survived~ Pclass*Sex*Age2+Embarked+Level+Age:Parch+Pclass:SibSp, family = "binomial", data = samp.train3)
summary(reduced.roc.6)
# 3.6
reduced.roc.7<- glm(Survived~ Pclass*Sex*Age3+Embarked+Level+SibSp:Age3+Age:Parch, family = "binomial", data = samp.train3)
summary(reduced.roc.7)

ROCcalc(reduced.roc.6, samp.test3, 2)
ROCcalc(reduced.roc.7, samp.test3, 2)

detach(completedData1)
####################
# Emsemble Methods #
####################

# Given we know the Age of the individual, we should use an ensemble from 1.8, 1.9, and 1.11, and
# POTENTIALLY 3.5 and 3.6.

# Given we do NOT know the Age of the individual, we should use an ensemble from 2.2, 2.3, and
# POTENTIALLY 3.5 and 3.6 (with the age imputed using pmm). During this imputation, include the data
# from training (with known ages only), in order to aid in the prediction.


# Create sample for ensemble testing
# Consider writing this as a function and testing repeatedly
sample4 <- sample.int(nrow(data.train), floor(.8*nrow(data.train)), replace = F)
samp.train4 <- data.train[sample4, ]
train.NA <- which(is.na(samp.train4$Age))
samp.test4 <- data.train[-sample4, ]
test.NA <- which(is.na(samp.test4$Age))

# Change reference class for Level
samp.train4$Level <- relevel(samp.train4$Level, ref = "N")
samp.test4$Level <- relevel(samp.test4$Level, ref = "N")

# Create a cleaned version of samp.train4
samp.train4.clean <- na.omit(samp.train4)

# Create parallel samples for the imputed data
x <- c(1:dim(data.train)[1])
y <- x[-sample4]
samp.clean <- as.integer(row.names(samp.train4.clean))
imp.train <- miceadds::subset_datlist(data.train.3.temp, subset = sample4)
imp.test <- miceadds::subset_datlist(data.train.3.temp, subset = y)
imp.train.clean <- miceadds::subset_datlist(data.train.3.temp, subset = samp.clean)

# Now show predictions with each model
# Models that learned from the data with true ages
mod1.8 <- glm(Survived~Pclass*Sex*Age2+Embarked+Level+Age2:SibSp, data = samp.train4, family = "binomial")
mod1.9 <- glm(Survived~Pclass*Sex*Age3+Level+Age3:SibSp, data = samp.train4, family = "binomial")
mod1.11 <- glm(Survived~Pclass*Sex*Age3+Level+Age3:SibSp+SibSp, data = samp.train4, family = "binomial")
samp.test4$pred1.8 <- predict.glm(mod1.8, newdata = samp.test4, type="response")
samp.test4$pred1.9 <- predict.glm(mod1.9, newdata = samp.test4, type="response")
samp.test4$pred1.11 <- predict.glm(mod1.11, newdata = samp.test4, type="response")

# Models that learned from the data with no Age variable
mod2.2 <- glm(Survived~Pclass+Sex+Embarked+Level+Sex:Pclass+Sex:SibSp, data = samp.train4, family = "binomial")
mod2.3 <- glm(Survived~Pclass+Sex+Embarked+Level+Sex:Pclass+Sex:SibSp+Sex:Parch+Pclass:Parch, data = samp.train4, family = "binomial")
mod2.2.clean <- glm(Survived~Pclass+Sex+Embarked+Level+Sex:Pclass+Sex:SibSp, data = samp.train4.clean, family = "binomial")
mod2.3.clean <- glm(Survived~Pclass+Sex+Embarked+Level+Sex:Pclass+Sex:SibSp+Sex:Parch+Pclass:Parch, data = samp.train4.clean, family = "binomial")
samp.test4$pred2.2 <- predict.glm(mod2.2, newdata = samp.test4, type="response")
samp.test4$pred2.3 <- predict.glm(mod2.3, newdata = samp.test4, type="response")

# Function to use the pooled prediction with test data
imp.Pred <- function(imp.data, imp.mod, m) {
  pred <- numeric(dim(imp.data[[1]])[1])
  pooled <- pool(imp.mod)
  pooled_lm = imp.mod[[1]]
  pooled_lm$coefficients = summary(pooled)$estimate
  for (i in 1:m) {
    comp <- imp.data[[i]]
    pred <- pred + predict.glm(pooled_lm, newdata = comp, type="response")
  }
  return(pred/m)
}

# Models that learned from the data with missing Ages imputed
fit3.5 <- with(data = imp.train, exp = glm(Survived~ Pclass*Sex*Age2+Embarked+Level+Age:Parch+Pclass:SibSp, family = "binomial"))
fit3.6 <- with(data = imp.train, exp = glm(Survived~ Pclass*Sex*Age3+Embarked+Level+SibSp:Age3+Age:Parch, family = "binomial"))
samp.test4$pred3.5 <- imp.Pred(imp.test, fit3.5, 5)
samp.test4$pred3.6 <- imp.Pred(imp.test, fit3.6, 5)

# Ensemble the prediction
# Simple ensemble
attach(samp.test4)
samp.test4$ensemble.1 <- rowMeans(samp.test4[, c('pred1.8', 'pred1.9', 'pred1.11', 'pred2.2', 'pred2.3', 'pred3.5', 'pred3.6')], na.rm=TRUE)
detach(samp.test4)


KfoldImp <-function(dataset, k, m) {
  TotErr <- 0
  TotErr2 <- 0
  n <- floor(dim(dataset[[1]])[1]/k)
  N <- dim(dataset[[1]])[1]
  allRows <- c(1:N)
  # First fit the model on each k-subset of data 
  # Then predict the results of the remaining data and calculate to 0-1 loss
  for (i in 1:k) {
    testRows <- ((i-1)*n+1):(i*n)
    holdout <- miceadds::subset_datlist(dataset, subset = testRows)
    trainRows <- allRows[-testRows]
    Rdata <-  miceadds::subset_datlist(dataset, subset = trainRows)
    Rresult <- with(data = Rdata, exp = glm(Survived~ Pclass*Sex*Age2+Embarked+Level+Age:Parch+Pclass:SibSp, family = "binomial"))
    Rresult2 <- with(data = Rdata, exp = glm(Survived~ Pclass*Sex*Age3+Embarked+Level+SibSp:Age3+Age:Parch, family = "binomial"))
    pred <- imp.Pred(holdout, Rresult, m)
    pred2 <- imp.Pred(holdout, Rresult2, m)
    correct <- with(holdout[[1]], expr = as.numeric(pred > 0.5) == holdout[[1]]$Survived)
    correct2 <- with(holdout[[1]], expr = as.numeric(pred2 > 0.5) == holdout[[1]]$Survived)
    accu <- sum(correct)/length(correct)
    accu2 <- sum(correct2)/length(correct2)
    TotErr <- TotErr + (1 - accu)
    TotErr2 <- TotErr2 + (1 - accu2)
  }
  return(list(err3.5 = TotErr/k, err3.6 = TotErr2/k))
}


mod1.8.err <- cv.glm(samp.train4.clean, mod1.8, K=20, cost = cost)$delta[1]
mod1.9.err <- cv.glm(samp.train4.clean, mod1.9, K=20, cost = cost)$delta[1]
mod1.11.err <- cv.glm(samp.train4.clean, mod1.11, K=20, cost = cost)$delta[1]
mod2.2.err <- cv.glm(samp.train4, mod2.2, K=20, cost = cost)$delta[1]
mod2.3.err <- cv.glm(samp.train4, mod2.3, K=20, cost = cost)$delta[1]
mod2.2.err.clean <- cv.glm(samp.train4.clean, mod2.2.clean, K=20, cost = cost)$delta[1]
mod2.3.err.clean <- cv.glm(samp.train4.clean, mod2.3.clean, K=20, cost = cost)$delta[1]
mod3.5.err <- KfoldImp(dataset = imp.train, k=20, m=5)$err3.5
mod3.6.err <- KfoldImp(dataset = imp.train, k=20, m=5)$err3.6
mod3.5.err.clean <- KfoldImp(dataset = imp.train.clean, k=20, m=5)$err3.5
mod3.6.err.clean <- KfoldImp(dataset = imp.train.clean, k=20, m=5)$err3.6


wEnsemble <- function(a, b, c, d, e, f, g, h) {
  if (is.na(a) == TRUE) {
    invErr <- (1-mod2.2.err) + (1-mod2.3.err) + (1-mod3.5.err) + (1-mod3.6.err)
    pred <- with(samp.test4, expr = e*((1-mod2.2.err)/invErr) + f*((1-mod2.3.err)/invErr) + g*((1-mod3.5.err)/invErr) + h*((1-mod3.6.err)/invErr))
  }
  else{
    invErr <- (1-mod2.2.err.clean) + (1-mod2.3.err.clean) + (1-mod3.5.err.clean) + (1-mod3.6.err.clean) + (1-mod1.8.err) + (1-mod1.9.err) + (1-mod1.11.err)
    pred <- with(samp.test4, expr = e*((1-mod2.2.err.clean)/invErr) + f*((1-mod2.3.err.clean)/invErr) + g*((1-mod3.5.err.clean)/invErr) + h*((1-mod3.6.err.clean)/invErr) + b*((1-mod1.8.err)/invErr) + c*((1-mod1.9.err)/invErr) + d*((1-mod1.11.err)/invErr))
  }
  return(pred)
}


wEnsemble2 <- function(a, b, c, d, e, f, g, h) {
  if (is.na(a) == TRUE) {
    invErr <- (1-mod2.2.err) + (1-mod2.3.err) + (1-mod3.5.err) + (1-mod3.6.err)
    pred <- with(samp.test4, expr = e*((1-mod2.2.err)/invErr) + f*((1-mod2.3.err)/invErr) + g*((1-mod3.5.err)/invErr) + h*((1-mod3.6.err)/invErr))
  }
  else{
    invErr <- (1-mod1.8.err) + (1-mod1.9.err) + (1-mod1.11.err)
    pred <- with(samp.test4, expr = b*((1-mod1.8.err)/invErr) + c*((1-mod1.9.err)/invErr) + d*((1-mod1.11.err)/invErr))
  }
  return(pred)
}

samp.test4$ensemble.2 <- apply(samp.test4[, c('Age', "pred1.8", "pred1.9", "pred1.11", 'pred2.2', "pred2.3", 'pred3.5', "pred3.6")], 1, function(x) wEnsemble(x['Age'], x['pred1.8'], x['pred1.9'], x['pred1.11'], x['pred2.2'], x['pred2.3'], x['pred3.5'], x['pred3.6']))
samp.test4$ensemble.3 <- apply(samp.test4[, c('Age', "pred1.8", "pred1.9", "pred1.11", 'pred2.2', "pred2.3", 'pred3.5', "pred3.6")], 1, function(x) wEnsemble2(x['Age'], x['pred1.8'], x['pred1.9'], x['pred1.11'], x['pred2.2'], x['pred2.3'], x['pred3.5'], x['pred3.6']))

Acc <- function(mod, dataset) {
  if (any(is.na(mod)) == TRUE) {
    correct <- with(dataset, expr = na.omit(as.numeric(mod > 0.5)) == na.omit(dataset)$Survived)
    accu <- sum(correct)/length(correct)
  }
  else {
    correct <- with(dataset, expr = as.numeric(mod > 0.5) == dataset$Survived)
    accu <- sum(correct)/length(correct)
  }
  return(accu)
}

Acc(samp.test4$pred1.8, samp.test4)
Acc(samp.test4$pred1.9, samp.test4)
Acc(samp.test4$pred1.11, samp.test4)
Acc(samp.test4$pred2.2, samp.test4)
Acc(samp.test4$pred2.3, samp.test4)
Acc(samp.test4$pred3.5, samp.test4)
Acc(samp.test4$pred3.6, samp.test4)
Acc(samp.test4$ensemble.1, samp.test4)
Acc(samp.test4$ensemble.2, samp.test4)
Acc(samp.test4$ensemble.3, samp.test4)



# Finally, make predictions on the original test set
data.test$Survived <- NA
data.test$Age2 <- log(data.test$Age)
data.test$Age3 <- data.test$Age^(-1/4)

merDat <- rbind(data.train, data.test)
imp.full <- mice(merDat, m = 5, meth= c('pmm'))
meth = imp.full$method
meth[c("Survived")] = ""
predM1 = imp.full$predictorMatrix
predM1[, c("Survived")]=0
imp.full = mice(merDat, method=meth, predictorMatrix=predM1, m=5)
imp.full$data$Level <- relevel(imp.full$data$Level, ref = "N")


x.full <- c(1:dim(merDat)[1])
x.train <- x.full[1:dim(data.train)[1]]
x.test <- x.full[-(1:dim(data.train)[1])]

mice.train <- miceadds::subset_datlist(imp.full, subset = x.train)
mice.test <- miceadds::subset_datlist(imp.full, subset = x.test)


red3.5 <- with(data = mice.train, exp = glm(Survived~ Pclass*Sex*Age2+Embarked+Level+Age:Parch+Pclass:SibSp, family = "binomial"))
red3.6 <- with(data = mice.train, exp = glm(Survived~ Pclass*Sex*Age3+Embarked+Level+SibSp:Age3+Age:Parch, family = "binomial"))

data.test$pred1.8 <- predict.glm(reduced.1.8, newdata = data.test, type="response")
data.test$pred1.9 <- predict.glm(reduced.1.9, newdata = data.test, type="response")
data.test$pred1.11 <- predict.glm(reduced.1.11, newdata = data.test, type="response")
data.test$pred2.2 <- predict.glm(reduced.2.2, newdata = data.test, type="response")
data.test$pred2.3 <- predict.glm(reduced.2.3, newdata = data.test, type="response")
data.test$pred3.5 <- imp.Pred(mice.test, red3.5, 5)
data.test$pred3.6 <- imp.Pred(mice.test, red3.6, 5)
data.test$ensemble.1 <- rowMeans(data.test[, c('pred1.8', 'pred1.9', 'pred1.11', 'pred2.2', 'pred2.3', 'pred3.5', 'pred3.6')], na.rm=TRUE)
data.test$ensemble.2 <- apply(data.test[, c('Age', "pred1.8", "pred1.9", "pred1.11", 'pred2.2', "pred2.3", 'pred3.5', "pred3.6")], 1, function(x) wEnsemble(x['Age'], x['pred1.8'], x['pred1.9'], x['pred1.11'], x['pred2.2'], x['pred2.3'], x['pred3.5'], x['pred3.6']))
data.test$ensemble.3 <- apply(data.test[, c('Age', "pred1.8", "pred1.9", "pred1.11", 'pred2.2', "pred2.3", 'pred3.5', "pred3.6")], 1, function(x) wEnsemble2(x['Age'], x['pred1.8'], x['pred1.9'], x['pred1.11'], x['pred2.2'], x['pred2.3'], x['pred3.5'], x['pred3.6']))
data.test$Survived <- as.numeric(data.test$ensemble.3 > 0.5)

write.csv(data.test[c('PassengerId', 'Survived')], '/Users/cullenbaker/Library/Mobile Documents/com~apple~CloudDocs/R-Projects/titanic/Survivors.csv', row.names = FALSE)



data.test$pred1.8 <- predict.glm(mod1.8, newdata = data.test, type="response")
data.test$pred1.9 <- predict.glm(mod1.9, newdata = data.test, type="response")
data.test$pred1.11 <- predict.glm(mod1.11, newdata = data.test, type="response")
data.test$pred2.2 <- predict.glm(mod2.2, newdata = data.test, type="response")
data.test$pred2.3 <- predict.glm(mod2.3, newdata = data.test, type="response")
data.test$pred3.5 <- imp.Pred(mice.test, fit3.5, 5)
data.test$pred3.6 <- imp.Pred(mice.test, fit3.6, 5)
data.test$ensemble.1 <- rowMeans(data.test[, c('pred1.8', 'pred1.9', 'pred1.11', 'pred2.2', 'pred2.3', 'pred3.5', 'pred3.6')], na.rm=TRUE)
data.test$ensemble.2 <- apply(data.test[, c('Age', "pred1.8", "pred1.9", "pred1.11", 'pred2.2', "pred2.3", 'pred3.5', "pred3.6")], 1, function(x) wEnsemble(x['Age'], x['pred1.8'], x['pred1.9'], x['pred1.11'], x['pred2.2'], x['pred2.3'], x['pred3.5'], x['pred3.6']))
data.test$ensemble.3 <- apply(data.test[, c('Age', "pred1.8", "pred1.9", "pred1.11", 'pred2.2', "pred2.3", 'pred3.5', "pred3.6")], 1, function(x) wEnsemble2(x['Age'], x['pred1.8'], x['pred1.9'], x['pred1.11'], x['pred2.2'], x['pred2.3'], x['pred3.5'], x['pred3.6']))
data.test$Survived <- as.numeric(data.test$ensemble.3 > 0.5)

write.csv(data.test[c('PassengerId', 'Survived')], '/Users/cullenbaker/Library/Mobile Documents/com~apple~CloudDocs/R-Projects/titanic/Survivors.csv', row.names = FALSE)




# Alternative K folds method for imp data
#KfoldsImp <- function(dataset, m, k) {
 # err1 <- 0
  #err2 <- 0
  #for (i in 1:m) {
   # dat <- dataset[[i]]
    #err1 <- err1 + cv.glm(data = dat, glmfit = with(data = dat, expr = glm(Survived~ Pclass*Sex*Age2+Embarked+Level+Age:Parch+Pclass:SibSp, family = "binomial")), cost = cost, K = k)$delta[1]
    #err2 <- err2 + cv.glm(data = dat, glmfit = with(data = dat, expr = glm(Survived~ Pclass*Sex*Age3+Embarked+Level+SibSp:Age3+Age:Parch, family = "binomial")), cost = cost, K = k)$delta[1]
  #}
  #return(list(err3.5 = err1/m, err3.6 = err2/m))
#}
