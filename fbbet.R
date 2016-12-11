# Football Project
# Luis Campos Garrido 05/08/2016

# Data load
{
esp11 <- read.csv('http://www.football-data.co.uk/mmz4281/1112/SP1.csv')
esp12 <- read.csv('http://www.football-data.co.uk/mmz4281/1213/SP1.csv')
esp13 <- read.csv('http://www.football-data.co.uk/mmz4281/1314/SP1.csv')
esp14 <- read.csv('http://www.football-data.co.uk/mmz4281/1415/SP1.csv')
esp15 <- read.csv('http://www.football-data.co.uk/mmz4281/1516/SP1.csv')
uk11 <- read.csv('http://www.football-data.co.uk/mmz4281/1112/E0.csv')
uk12 <- read.csv('http://www.football-data.co.uk/mmz4281/1213/E0.csv')
uk13 <- read.csv('http://www.football-data.co.uk/mmz4281/1314/E0.csv')
uk14 <- read.csv('http://www.football-data.co.uk/mmz4281/1415/E0.csv')
uk15 <- read.csv('http://www.football-data.co.uk/mmz4281/1516/E0.csv')
deu11 <- read.csv('http://www.football-data.co.uk/mmz4281/1112/D1.csv')
deu12 <- read.csv('http://www.football-data.co.uk/mmz4281/1213/D1.csv')
deu13 <- read.csv('http://www.football-data.co.uk/mmz4281/1314/D1.csv')
deu14 <- read.csv('http://www.football-data.co.uk/mmz4281/1415/D1.csv')
deu15 <- read.csv('http://www.football-data.co.uk/mmz4281/1516/D1.csv')
}

# Test set
{
test <- esp15[,3:7]
test$OverUnder <- NULL
test$OverUnderPred <- NULL
test$OverUnderProb <- NULL
test$ResultPred <- NULL
test$ResultProb <- NULL
}

# Cleanning teams in Test set that are not in Train
train <- list(esp12[,3:6], esp13[,3:6], esp14[,3:6])
train <- do.call(rbind, train)
cleanData <- function(testset, trainset) {
  # Home teams
  i <- 0
  testset$Delete <- FALSE
  for(team1 in testset$HomeTeam) {
    flag <- 0
    i <- i + 1
    for(team2 in trainset$HomeTeam) {
      if(identical(team1, team2)) {
        flag <- 1
        break
      }
    }
    if(flag==0) testset$Delete[i] <- TRUE
  }
  testset <- testset[-c(which(testset$Delete==TRUE, arr.ind = TRUE)),]
  # Away teams
  i <- 0
  testset$Delete <- FALSE
  for(team1 in testset$AwayTeam) {
    flag <- 0
    i <- i + 1
    for(team2 in trainset$AwayTeam) {
      if(identical(team1, team2)) {
        flag <- 1
        break
      }
    }
    if(flag==0) testset$Delete[i] <- TRUE
  }
  testset <- testset[-c(which(testset$Delete==TRUE, arr.ind = TRUE)),]
  testset$Delete <- NULL
  return(testset)
}
test <- cleanData(test, train)

# Trainning set
train <- apply(train, 1, function(row){
  data.frame(team=c(row['HomeTeam'], row['AwayTeam']),
             opponent=c(row['AwayTeam'], row['HomeTeam']),
             goals=as.numeric(c(row['FTHG'], row['FTAG'])),
             home=c(1, 0),
             row.names = NULL)
})
train <- do.call(rbind, train)

# Fit the model
model <- glm(goals ~ home + team + opponent, family = "poisson", data = train)
#summary(model)

# Dixon-Coles adjustment
#
expected <- fitted(model)
home.expected <- expected[1:nrow(train)]
away.expected <- expected[(nrow(train)+1):(nrow(train)*2)]
#
DClogLik <- function(y1, y2, lambda, mu, rho=0){
  #rho=0, independence
  sum(log(tau(y1, y2, lambda, mu, rho)) + log(dpois(y1, lambda)) + log(dpois(y2, mu)))
}
#
DCoptimRhoFn <- function(par){
  rho <- par[1]
  DClogLik(train$FTHG, train$FTAG, home.expected, away.expected, rho)
}
#model <- optim(par=c(0.1), fn=DCoptimRhoFn, control=list(fnscale=-1), method='BFGS')

# Predictions
for (n in 1:nrow(test)) {
  # Expected goals home
  lambda <- predict(model, data.frame(home=1, team=test$HomeTeam[n], opponent=test$AwayTeam[n]), type="response")
  # Expected goals away
  mu <- predict(model, data.frame(home=0, team=test$AwayTeam[n], opponent=test$HomeTeam[n]), type="response")
  # Calculate probability of scores
  maxgoal <- 5
  prob_matrix <- dpois(0:maxgoal, lambda) %*% t(dpois(0:maxgoal, mu))
  #scaling_matrix <- matrix(tau(c(0,1,0,1), c(0,0,1,1), lambda, mu, model$par['RHO']), nrow=2)
  #prob_matrix[1:2, 1:2] <- prob_matrix[1:2, 1:2] * scaling_matrix
  # Win / Loss / Draw
  draw <- sum(diag(prob_matrix))
  away <- sum(prob_matrix[upper.tri(prob_matrix)])
  home <- sum(prob_matrix[lower.tri(prob_matrix)])
  if(draw > away & draw > home) test$ResultPred[n] <- "D"
  else if(away > draw & away > home) test$ResultPred[n] <- "A"
  else if(home > away & home > draw) test$ResultPred[n] <- "H"
  # Over / Under
  market <- 2.5
  prob <- 0
  for(i in as.numeric(rownames(prob_matrix))) {
    for(j in as.numeric(colnames(m))) {
      if(i + j > market) prob <- prob + m[i+1, j+1]
    }
  }
  test$OverUnder[n] <- (test$FTHG[n] + test$FTAG[n] > 2.5)
  test$OverUnderPred[n] <- (prob > 0.5)
  test$OverUnderProb[n] <- prob
}

# Test hits
{
test$FTHG <- NULL
test$FTAG <- NULL
test$ResultHit <- test$FTR == test$ResultPred
test$OverUnderHit <- test$OverUnder == test$OverUnderPred
resultHits <- sum(test$ResultHit==TRUE)
overUnderHits <- sum(test$OverUnderHit==TRUE)
cat('Result hits: ', resultHits/nrow(test)*100, '%')
cat('Over/Under hits: ', overUnderHits/nrow(test)*100, '%')
}
