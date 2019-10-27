## Copyright 2019 Andrew Zammit Mangion
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

library("tensorflow")

logdet_tf <- function (R) {
  diagR <- tf$linalg$diag_part(R)
  ldet <- tf$log(diagR) %>%
          tf$reduce_sum(axis = -1L) %>%
          tf$multiply(2)
  return(ldet)
}

ndim_tf <- function(x) {
  tf$size(tf$shape(x))
}

tr_tf <- function(A) {
  tf$trace(A)
}

safe_chol_tf <- function(A) {
  I <- tf$constant(1e-6 * diag(nrow(A)), name = "Imat", dtype = "float32")
  tf$cholesky_upper(tf$add(A, I))
}

atBa_tf <- function(a, B) {
  a %>%
    tf$transpose() %>%
    tf$matmul(B) %>%
    tf$matmul(a)
}

ABinvAt_tf <- function(A, cholB) {
  AcholB <- tf$matmul(A, tf$matrix_inverse(cholB))
  tf$matmul(AcholB, tf$transpose(AcholB))
}

AtBA_p_C_tf <- function(A, cholB, C) {
  cholBA <- tf$matmul(cholB, A)
  tf$matmul(tf$linalg$transpose(cholBA), cholBA) %>%
     tf$add(C)
}


entropy_tf <- function(s) {
  d <- ncol(s)
  s %>% tf$log() %>% tf$reduce_sum() %>% tf$multiply(0.5)
}

chol2inv_tf <- function(R) {
  Rinv <- tf$matrix_inverse(R)
  tf$matmul(Rinv, tf$transpose(Rinv))
}

tile_on_dim1 <- function(A, n) {
  m1 <- nrow(A)
  m2 <- ncol(A)
  X <- tf$tile(A, c(n, 1L)) %>%
       tf$reshape(c(n, m1, m2))
  X
}

pinvsolve_tf <- function(A, b, reltol = 1e-6) {
  # Compute the SVD of the input matrix A
  A_SVD = tf$svd(A)
  s <- A_SVD[[1]]
  u <- A_SVD[[2]]
  v <- A_SVD[[3]]

  # Invert s, clear entries lower than reltol*s[0].
  atol = tf$multiply(tf$reduce_max(s), reltol)
  s_mask = tf$boolean_mask(s, tf$greater_equal(s, atol))
  s_reciprocal <- tf$reciprocal(s_mask)
  s_inv = tf$diag(tf$concat(list(s_reciprocal,
                                 tf$zeros(tf$size(s) - tf$size(s_mask))), 0L))

  # Compute v * s_inv * u_t * b from the left to avoid forming large intermediate matrices.
  tf$matmul(v, tf$matmul(s_inv, tf$matmul(u, b, transpose_a = TRUE)))
}

scale_lims_tf <- function(s_tf) {
  smin_tf <- tf$reduce_min(s_tf, axis = -2L, keepdims = TRUE)
  smax_tf <- tf$reduce_max(s_tf, axis = -2L, keepdims = TRUE)

  list(min = smin_tf,
       max = smax_tf)
}

scale_0_5_tf <- function(s_tf, smin_tf, smax_tf) {
  s_tf <- (s_tf - smin_tf) /(smax_tf - smin_tf) -
          tf$constant(0.5, dtype = "float32")
}

KL_tf <- function(mu1, S1, mu2, S2) {
  R1 <- tf$cholesky_upper(S1)
  R2 <- tf$cholesky_upper(S2)
  Q2 <- chol2inv_tf(R2)
  k <- tf$shape(mu1)[1] %>% tf$to_float()

  Part1 <- tf$matmul(Q2, S1) %>% tf$trace()
  Part2 <- (tf$transpose(mu2 - mu1) %>%
           tf$matmul(Q2) %>%
           tf$matmul(mu2 - mu1))[1,1]
  Part3 <- -k
  Part4 <- logdet_tf(R2) - logdet_tf(R1)

  tf$constant(0.5, dtype = "float32") * (Part1 + Part2 + Part3 + Part4)
}

k_tf1 <- function(s, r, w, D) {
  d1 <- tf$square(s[, 1] - w[, 1, drop = FALSE] - r[, 1, drop = FALSE])
  d2 <- tf$square(s[, 2] - w[, 2, drop = FALSE] - r[, 2, drop = FALSE])
  1/(4 * pi * D[, 1, drop = FALSE]) * tf$exp(-(d1 + d2) / (4 *  D[, 1, drop = FALSE])) * tfconst(ds^2)
}

k_tf2 <- function(s, r, w, D, ds) {
  d1 <- tf$square(s[,, 1] - w[,, 1, drop = FALSE] - r[,, 1, drop = FALSE])
  d2 <- tf$square(s[,, 2] - w[,, 2, drop = FALSE] - r[,, 2, drop = FALSE])
  1/(4 * pi * D[,, 1, drop = FALSE]) * tf$exp(-(d1 + d2) / (4 *  D[,, 1, drop = FALSE])) * tfconst(ds^2)
}

sqexp <- function(s, r, sd) {
  d1 <- tf$square(r[, 1] - s[, 1, drop = FALSE])
  d2 <- tf$square(r[, 2] - s[, 2, drop = FALSE])
  tf$exp(-(d1 + d2) / (2 * tf$square(sd)))
}

Matern32 <- function(sigma2, rho, D) {
  s3 <- tfconst(sqrt(3))
  sigma2*(1 + s3*D/rho)*tf$exp(-s3*D/rho)
}

Matern12 <- function(sigma2, rho, D) {
  sigma2*tf$exp(-D/rho)
}

## Function to compute the empirical covariance matrix of x
## tf_cov <- function(x, thisN_Batch) {
##  mean_x <- tf$reduce_mean(x, axis = 0L, keepdims = TRUE)
##  mx <- tf$tile(mean_x, c(thisN_Batch, 1L))
##  diffx <- x - mx
## vx = tf$matmul(tf$transpose(diffx), diffx) / (tf$cast(thisN_Batch, tf$float32) - 1)
## }

## Function to compute the empirical covariance matrix of x
tf_cov <- function(x, y = x, thisN_Batch) {
    mean_x <- tf$reduce_mean(x, axis = 0L, keepdims = TRUE)
    mean_y <- tf$reduce_mean(y, axis = 0L, keepdims = TRUE)
    mx <- tf$tile(mean_x, c(thisN_Batch, 1L))
    my <- tf$tile(mean_y, c(thisN_Batch, 1L))
    diffx <- x - mx
    diffy <- y - my
    vx <- tf$matmul(tf$transpose(diffx), diffy) / (tf$cast(thisN_Batch, tf$float32) - 1)
}

## Set tensorflow seed
set_tf_seed <- function(seed = 1L) {
  tf$set_random_seed(seed)
  invisible()
}


tfconst <- function(x, name = NULL) tf$constant(x, dtype = "float32", name = name)
tfVar <- function(x, name = NULL) tf$Variable(x, dtype = "float32", name = name)

## Helper Functions
run <- tf$Session()$run
init <- tf$global_variables_initializer()
