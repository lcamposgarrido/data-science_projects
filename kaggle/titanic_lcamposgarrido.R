# Kaggle Titanic Competition
# Luis Campos Garrido 16/04/2016
# Ranked 709th/4024 (best 20%)

# Set working directory, import datafiles and load libraries
setwd("~/kaggle/titanic")
train <- read.csv("~/kaggle/titanic/train.csv")
test <- read.csv("~/kaggle/titanic/test.csv")

# Loading libraries
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)
library(party)

# Combine train and test datasets
test$Survived <- NA
alldata <- rbind(train, test)

# Add new column Title and clean it up
alldata$Name <- as.character(alldata$Name)
alldata$Title <- sapply(alldata$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
alldata$Title <- sub(' ', '', alldata$Title)
alldata$Title[alldata$Title %in% c('Mlle', 'Mme', 'Ms')] <- 'Miss'
alldata$Title[alldata$Title %in% c('Lady', 'the Countess', 'Dona')] <- 'Mrs'
alldata$Title[alldata$Title %in% c('Capt', 'Don', 'Major', 'Sir', 'Jonkheer')] <- 'Sir'
alldata$Title <- factor(alldata$Title)

# Add new column FamilySize
alldata$FamilySize <- alldata$SibSp + alldata$Parch + 1

# Cleanning up de data for forest model
#Fixing NA Age values
agefit <- rpart(Age ~ Pclass + Sex + Fare + Embarked + Title + FamilySize, data = alldata[!is.na(alldata$Age), ], method = "anova")
alldata$Age[is.na(alldata$Age)] <- predict(agefit, alldata[is.na(alldata$Age), ])
#Fixing empty Embarked values
alldata$Embarked[which(alldata$Embarked == '')] = "S"
alldata$Embarked <- factor(alldata$Embarked)
#Fixing NA Fare value
alldata$Fare[which(is.na(alldata$Fare))] <- median(alldata$Fare, na.rm = TRUE) 

# Separate data into train and test sets
train_new <- alldata[1:891, ]
test_new <- alldata[892: 1309, ]

# Decision tree, plot, prediction and export
set.seed(1358)
forest <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + Fare + Embarked + Title + FamilySize, data = train_new, controls = cforest_unbiased(ntree = 2000, mtry = 3))
prediction <- predict(forest, test_new, OOB = TRUE, type = "response")
solution <- data.frame("PassengerId" = test_new$PassengerId, "Survived" = prediction)
write.csv(solution, "titanic_solution_lcamposgarrido.csv", row.names = FALSE)
