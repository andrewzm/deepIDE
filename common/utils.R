nasa_palette <- c("#03006d","#02008f","#0000b6","#0001ef","#0000f6",
                  "#0428f6","#0b53f7","#0f81f3",
                  "#18b1f5","#1ff0f7","#27fada","#3efaa3","#5dfc7b",
                  "#85fd4e","#aefc2a","#e9fc0d","#f6da0c","#f5a009",
                  "#f6780a","#f34a09","#f2210a","#f50008","#d90009",
                  "#a80109","#730005")


Wendland1_R <- function(theta, D) {
  R <- D / theta
  W <- (R <= 1) * (1 - R)^4 * (1 + 4 * R)
}


RMSPE <- function(x, y) {
  sqrt(mean((x - y)^2))
}

coverage90 <- function(z,mu,se) {
  
  lower <- mu - 1.64*se
  upper <- mu + 1.64*se
  sum((z < upper) & (z > lower)) / length(z)
}

IS90 <- function(true, mu, se) {
  
  alpha = 0.1
  pred90l <- mu - 1.64*se
  pred90u <- mu + 1.64*se
  
  ISs <- (pred90u - pred90l) + 2/alpha * (pred90l - true) * (true < pred90l) +
    2/alpha * (true - pred90u) * (true > pred90u)
  mean(ISs)
}

summarystats <- function(true, mu, se, name, time, zone) {
  
  data.frame(method = name,
             time = time,
             zone = zone,
             RMSPE = RMSPE(true, mu),
             CRPS = crps(true, cbind(mu, se))$CRPS,
             IS90 = IS90(true, mu, se),
             Cov90 = coverage90(true, mu, se))               
  
}

