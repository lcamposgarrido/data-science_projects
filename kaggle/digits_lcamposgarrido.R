# Kaggle Digits Recognizer
# Luis Campos Garrido 23/04/2016
# Ranked 918th/1355

# Set working directory, import datafiles and load libraries
setwd("~/kaggle/digits")
train <- read.csv("~/kaggle/digits/train.csv")
test <- read.csv("~/kaggle/digits/test.csv")
head(train[1:10])

# Plot some 28*28 matrix with pixel color values
rotate <- function(x) t(apply(x, 2, rev))
par(mfrow=c(2,3))
lapply(5:10, function(x) image(
      rotate(matrix(unlist(train[x,-1]),nrow = 28,byrow = T)),
      col=grey.colors(255),
      xlab=train[x,1]))

# Random forest benchmark
library(randomForest)
set.seed(1358)
numTrees <- 25
forest <- randomForest(train[ , 2:ncol(train)], y = as.factor(train$label), xtest = test, ntree = numTrees)
# Output solution
predictions <- data.frame(ImageId = 1:nrow(test), Label = forest$test$predicted)
write.csv(predictions, "digits_solution_lcamposgarrido.csv", row.names = FALSE)

# Conditional inference trees random forest benchmark
library(party)
set.seed(1358)
numTrees <- 100
forest <- cforest(label ~ ., data = train, control = cforest_unbiased(ntree = numTrees))
predictions <- predict(forest, test, OOB = TRUE)
solution <- data.frame(ImageId = 1:nrow(test), Label = predictions)
write.csv(predictions, "digits_solution_lcamposgarrido.csv", row.names = FALSE)
