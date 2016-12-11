# Football project with Dixon-Coles model
# Luis Campos Garrido 05/08/2016

# Data load
{
esp12 <- read.csv('http://www.football-data.co.uk/mmz4281/1213/SP1.csv')
esp13 <- read.csv('http://www.football-data.co.uk/mmz4281/1314/SP1.csv')
esp14 <- read.csv('http://www.football-data.co.uk/mmz4281/1415/SP1.csv')
esp15 <- read.csv('http://www.football-data.co.uk/mmz4281/1516/SP1.csv')
}

# 1. Building the model

# Tau function: computes the degree in which the probability for the low scoring goals changes
tau <- Vectorize(function(xx, yy, lambda, mu, rho){
  if (xx==0 & yy==0) return(1 - (lambda*mu*rho))
  else if (xx==0 & yy==1) return(1 + (lambda*rho))
  else if (xx==1 & yy==0) return(1 + (mu*rho))
  else if (xx==1 & yy==1) return(1 - rho)
  else return(1)
})

# Dixon-Coles likelihood function:
# takes vectors mu-lambda (expected home/away goals), rho and vectors y1-y2 (observed home/away goals) and
# computes the log-likelihood
DClogLik <- function(y1, y2, lambda, mu, rho=0){
  #rho=0, independence
  sum(log(tau(y1, y2, lambda, mu, rho)) + log(dpois(y1, lambda)) + log(dpois(y2, mu)))
}

# Data wrangling function: takes data.frame from 'football-data.co.uk' and returns a list with matrices and vectors with the match results
DCmodelData <- function(df){
  hm <- model.matrix(~ HomeTeam - 1, data=df, contrasts.arg=list(HomeTeam='contr.treatment'))
  am <- model.matrix(~ AwayTeam -1, data=df)
  team.names <- unique(c(levels(df$HomeTeam), levels(df$AwayTeam)))
  return(list(
    homeTeamDM=hm,
    awayTeamDM=am,
    homeGoals=df$FTHG,
    awayGoals=df$FTAG,
    teams=team.names
  )) 
}

# Dixon-Coles optimization function:
# calculates the log-likelihood from a set of parameters and the data we have
# first calculates lambda-mu for each match, then passes it and n goals per match to the log-likelihood function
DCoptimFn <- function(params, DCm){
  home.p <- params[1]
  rho.p <- params[2]
  nteams <- length(DCm$teams)
  attack.p <- matrix(params[3:(nteams+2)], ncol=1)
  defence.p <- matrix(params[(nteams+3):length(params)], ncol=1)
  lambda <- exp(DCm$homeTeamDM %*% attack.p + DCm$awayTeamDM %*% defence.p + home.p)
  mu <- exp(DCm$awayTeamDM %*% attack.p + DCm$homeTeamDM %*% defence.p)
  return(DClogLik(y1=DCm$homeGoals, y2=DCm$awayGoals, lambda, mu, rho.p) * -1)
}

# Dixon-Coles attack parameters constraint function:
# helps the optimizer handle the constraint that all the attack parameters must sum to 1
DCattackConstr <- function(params, DCm, ...){
  nteams <- length(DCm$teams)
  attack.p <- matrix(params[3:(nteams+2)], ncol=1)
  return((sum(attack.p) / nteams) - 1)
}

# Data load
train <- list(esp12[,3:6], esp13[,3:6], esp14[,3:6])
train <- do.call(rbind, train)
dcm <- DCmodelData(train)

# Initial parameter estimates
attack.params <- rep(.01, times=nlevels(train$HomeTeam))
defence.params <- rep(-0.08, times=nlevels(train$HomeTeam))
home.param <- 0.06
rho.init <- 0.03
par.inits <- c(home.param, rho.init, attack.params, defence.params)
names(par.inits) <- c('HOME', 'RHO', paste('Attack.', dcm$teams), paste('Defence.', dcm$teams))

# Optimization with equiality constraints
library(alabama)
res <- auglag(par=par.inits, fn=DCoptimFn, heq=DCattackConstr, DCm=dcm)

res$par


# 2. Test set

test <- esp15[,3:7]

# Clean teams in Test set that are not in Train
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
test$ResultPred <- NULL

# 3. Predictions making

# Predict function: pass model and home-away teams and returns prob matrix
DCpredict <- function(res, homeTeam, awayTeam) {
  homeAttack <- paste('Attack.', homeTeam)
  homeDefence <- paste('Defence.', homeTeam)
  awayAttack <- paste('Attack.', awayTeam)
  awayDefence <- paste('Defence.', awayTeam)
  # Expected goals home
  lambda <- exp(res$par['HOME'] + res$par[homeAttack] + res$par[awayDefence])
  # Expected goals away
  mu <- exp(res$par[awayAttack] + res$par[homeDefence])
  # Matrix based on independent Poisson distributions
  maxgoal <- 5
  probability_matrix <- dpois(0:maxgoal, lambda) %*% t(dpois(0:maxgoal, mu))
  # Rho to adjust low-scoring results 
  scaling_matrix <- matrix(tau(c(0,1,0,1), c(0,0,1,1), lambda, mu, res$par['RHO']), nrow=2)
  probability_matrix[1:2, 1:2] <- probability_matrix[1:2, 1:2] * scaling_matrix
  return(probability_matrix)
}

for (n in 1:nrow(test)) {
  m <- DCpredict(res, test$HomeTeam[n], test$AwayTeam[n])
  # Match result probabilities
  homeProb <- sum(m[lower.tri(m)])
  drawProb <- sum(diag(m))
  awayProb <- sum(m[upper.tri(m)])
  if(drawProb > awayProb & drawProb > homeProb) test$ResultPred[n] <- "D"
  else if(awayProb > drawProb & awayProb > homeProb) test$ResultPred[n] <- "A"
  else if(homeProb > awayProb & homeProb > drawProb) test$ResultPred[n] <- "H"
}

# Test hits
test$ResultHit <- test$FTR == test$ResultPred
#test$OverUnderHit <- test$OverUnder == test$OverUnderPred
resultHits <- sum(test$ResultHit==TRUE)
#overUnderHits <- sum(test$OverUnderHit==TRUE)
cat('Result hits: ', resultHits/nrow(test)*100, '%')
#cat('Over/Under hits: ', overUnderHits/nrow(test)*100, '%')
