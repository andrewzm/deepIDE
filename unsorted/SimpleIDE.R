library(ggquiver)
library(fields)
library(ggplot2)

## Simple spatially-varying IDE
ds <- 1/64L
sgrid <- expand.grid(s1 = seq(0, 1, length.out = 64L),
                     s2 = seq(0, 1, length.out = 64L))

nbasis <- 6^2
kernellocs <- expand.grid(s1 = seq(0, 1, length.out = sqrt(nbasis)),
                       s2 = seq(0, 1, length.out = sqrt(nbasis)))
kernelsds <- 1/(1.5*sqrt(nbasis))

sqexp <- function(s, r, sd) {
  d <- fields::rdist(s, r)
  exp(- (d^2) / (2 * (sd^2)))
}
PHI <-sqexp(sgrid, kernellocs, kernelsds)
u <- PHI %*% rnorm(nbasis, sd = 1/64L)
v <- PHI %*% rnorm(nbasis, sd = 1/64L)
df <- sgrid %>% mutate(u = c(u), v = c(v))
ggplot(df) + geom_tile(aes(x = s1, y = s2, fill = u)) + theme_bw()
ggplot(df) + geom_quiver(aes(x = s1, y = s2, u = u, v =  v)) + theme_bw()

Kfun <- function(s, r, w, D) {
  d <- fields::rdist(s, r - w)
  1/(4 * pi * D) * exp(-d^2 / (4 *  D)) * ds^2
}

K <- Kfun(sgrid, sgrid, cbind(u, v), 0.00005)
df$y <- - scale(filter(spatgridsub, lonbox == 3 & latbox == 2)$thetao)
ggplot(df) + geom_tile(aes(x = s1, y = s2, fill = y)) + theme_bw() + myscale
ggplot(df) + geom_tile(aes(x = s1, y = s2, fill = K %*% y)) + theme_bw() + myscale
ggplot(df) + geom_tile(aes(x = s1, y = s2, fill = K %*% (K %*% y))) + theme_bw() + myscale
ggplot(df) + geom_tile(aes(x = s1, y = s2, fill = K %*% (K %*% (K %*% y)))) + theme_bw() + myscale


