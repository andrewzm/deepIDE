createCNNgraph <- function(W = 64L, H = W, N_Channels = 3L, N_Filters = 64L,
                           sqrtN_Basis = 8L, patch_size_flow = 5L,
                           patch_size_diff = 5L, border_mask = 0.1) {

  ### Initialise spatial grid and mask

  s1 <- seq(0, 1, length.out = W)  # grid points for s1
  s2 <- seq(0, 1, length.out = H)  # grid points for s2
  ds1 <- mean(diff(s1))            # s1 spacings
  ds2 <- mean(diff(s2))            # s2 spacings
  ds <- sqrt(ds1 * ds2)            # area of pixel
  sgrid <- expand.grid(s1 = s1, s2 = s2)  # spatial grid in long format
  maskidx <- which(sgrid$s1 > border_mask & sgrid$s1 < (1 - border_mask) &  # we train on CNN only ..
                   sgrid$s2 > border_mask & sgrid$s2 < (1 - border_mask))   # inside this square box ..

  mask <- rep(1, W*H)                                 # to avoid boundary effects
  mask[-maskidx] <- 0
    
  ### 3.1: Data placeholders
  
  ## An input "data point" consisting of N_Batches of three frames 
  data_in <- tf$placeholder(dtype = "float32", shape = list(NULL, W, H, N_Channels))
  
  ## The final data point of data_in (for N_Batches)
  data_current <- tf$placeholder(dtype = "float32", shape = list(NULL, W, H))
  
  ## The future data point for data_in (for N_Batches)
  data_future <- tf$placeholder(dtype = "float32", shape = list(NULL, W, H))
  
  ## Number of batches
  thisN_Batch <- tf$shape(data_in)[1]
  
  ### 3.2: Kernel setup
  sgrid_tf <- tfconst(as.matrix(sgrid))   # spatial grid
  kernelside <- seq(0, 1, length.out = sqrtN_Basis)      # locs of kernel basis ..
  kernellocs <- as.matrix(expand.grid(s1 = kernelside,   # functions on grid
                                      s2 = kernelside))
  kernelsds <- 1/(1.5*sqrtN_Basis)       # width of keel basis functions
  kernellocs_tf <- tfconst(kernellocs)   # put as TF variable
  kernelsds_tf <- tfconst(kernelsds)     # put as TF variable
  PHI_tf <- sqexp(sgrid_tf, 
                  kernellocs_tf, 
                  kernelsds)             # basis function matrix (4096 x 64)
  PHI2_tf <- tf$expand_dims(PHI_tf, 0L)  # make (1 x 4096 x 64)      
  PHI3_tf <- tf$tile(PHI2_tf, c(thisN_Batch, 1L, 1L)) # tile for N_Batches
  
  ### 3.3: CNN architecture for the IDE parameters
  
  ### 3.3.1: CNN for the horizontal and vertical parameters
  
  ## Construct convnet that takes in the data (with N_Batches) and outputs
  ## a matrix of dimension N_Batches x 128 (the first N_Batches x 64 are for u
  ## while the second N_Batches x 64 are for v)
  nnet_flow <- ConvIDEnetsemiV2(data_in = data_in, 
                                N_Channels = N_Channels, 
                                N_Filters = N_Filters, 
                                sqrtN_Basis = sqrtN_Basis,
                                kernel_size = patch_size_flow)
  
  ## Make size N_Batches x 128 x 1
  flow_coef <- tf$expand_dims(nnet_flow$flow_coef, 2L)
  
  ## Multiply PHI with weights to get images of u and v
  u_long <- tf$matmul(PHI3_tf, flow_coef[, 1:(sqrtN_Basis^2),])
  v_long <- tf$matmul(PHI3_tf, flow_coef[, sqrtN_Basis^2 + 1:(sqrtN_Basis^2),])
  
  ### 3.3.2: CNN for the diffusion coefficients
  
  ## Construct convnet that takes in the data (with N_Batches) and outputs
  ## a matrix of dimension N_Batches x 64 
  nnet_D <- ConvIDEnet_diffusion(data_in = data_in, 
                                 N_Channels = N_Channels, 
                                 N_Filters = N_Filters, 
                                 sqrtN_Basis = sqrtN_Basis,
                                 kernel_size = patch_size_diff)
  
  ## Make size N_Batches x 64 x 1
  D_coef <- tf$expand_dims(nnet_D$D_coef, 2L)
  
  ## Multiply PHI with weights to get image of D
  D_long <- tf$matmul(PHI3_tf, D_coef[, 1:(sqrtN_Basis^2),])
  
  
  ### 3.3.3: The top layer
  w <- tf$concat(list(u_long, v_long), axis = 2L)  # Put into size (N_Batches x 4096 x 2)
  sgridlong_tf <- tf$expand_dims(sgrid_tf, 0L) # Spatial grid as (1 x 4096 x 2)
  K_IDE <- k_tf2(s = sgridlong_tf,             # Construct kernel which is ..
                 r = sgridlong_tf,             # of size N_Batches x 4096 x 4096
                 w = w,
                 D = D_long,
                 ds = ds)
  
  ## K_IDE_Long is in the right format -- we now just need
  ## to just multiply it by data_current for prediction
  ## But we need to make the 64 x 64 data into 4096 x 1 format
  ## We cannot just use tf$reshape, as the reordering is different than
  ## what R does when reshaping! We therefore have another function 
  ## reshape_3d_to_2d() to do the reordering.
  ## E.g., tf$reshape(data_current, c(-1L, W*H, 1L)) would be wrong
  data_current_long <- reshape_3d_to_2d(data_current, W*H)
  data_future_long <- reshape_3d_to_2d(data_future, W*H)
  Ypred <- tf$matmul(K_IDE, data_current_long)
  
  ## Now we mask when comparing, to reduce boundary effects
  mask_tf <- tfconst(array(mask, dim = c(1L, W*H, 1L)))
  mask_tf <- tf$tile(mask_tf, c(thisN_Batch, 1L, 1L))
  Ypred_masked <- tf$multiply(Ypred, mask_tf)
  data_future_masked <- tf$multiply(data_future_long, mask_tf)
  
  ## The basic cost function is then just the mean squared error 
  Cost1 <- tf$reduce_mean(tf$square(data_future_masked - Ypred_masked))
  
  ### 3.3.4: The top layer with Matern covariance function
  
  ## Create the variables for the Matern
  log_sigma2 <- tfVar(log(0.4), "log_sigma2")
  log_rho <- tfVar(log(0.1), "log_rho")
  
  sigma2 <- tf$exp(log_sigma2)
  rho <- tf$exp(log_rho)
  
  ## Create the distance matrix and covariance matrix
  d1 <- tf$square(sgrid_tf[, 1] - sgrid_tf[, 1, drop = FALSE])
  d2 <- tf$square(sgrid_tf[, 2] - sgrid_tf[, 2, drop = FALSE])
  D <- tf$sqrt(d1 + d2)
  SIGMA <- Matern32(sigma2, rho, D)
  
  ## Now create a mask matrix of size 2500 x 4096, so that only the 
  ## 2500 elements not masked out are returned when premultiplying
  ## So this is a bit differen from mask_tf which just zeroed out
  maskmat <- matrix(0, sum(mask), W*H)
  for(i in seq_along(maskidx)) {
    maskmat[i,maskidx[i]] <- 1
  }
  maskmat_tf <- tfconst(maskmat)
  
  ## Expand it for the number of batches we have
  maskmat_tiled <- tf$expand_dims(maskmat_tf, 0L)
  maskmat_tiled <- tf$tile(maskmat_tiled, c(thisN_Batch, 1L, 1L))
  
  ## Now extract the subset of covariance matrix we are interested in
  SIGMA_sub <- tf$matmul(tf$matmul(maskmat_tf, SIGMA), tf$transpose(maskmat_tf))
  
  ## And compute the likelihood based on this
  ## Compute the upper Cholesky and tile it for all the batches
  R <- tf$transpose(tf$cholesky(SIGMA_sub))
  Rinv <- tf$matrix_inverse(R)
  Rinv_tiled <- tf$expand_dims(Rinv, 0L)
  Rinv_tiled <- tf$tile(Rinv_tiled, c(thisN_Batch, 1L, 1L))
  
  ## Extract the part of data and the predictions that we need (all batches)
  data_future_sub <- tf$linalg$matmul(maskmat_tiled, data_future_long)
  Ypred_sub <- tf$linalg$matmul(maskmat_tiled, Ypred) 
  
  ## Find the difference Ytilde^T
  Ydiff <- tf$linalg$transpose(data_future_sub - Ypred_sub)
  
  ## Compute log determinant * number of batches (as it's the same for each batch)
  logdet_part <- -0.5 * tf$to_float(thisN_Batch)*logdet_tf(R)
  
  ## Compute quadratic part
  YdiffRinv <- tf$matmul(Ydiff, Rinv_tiled)
  squared_part_Batch <- -0.5 * tf$matmul(YdiffRinv, tf$linalg$transpose(YdiffRinv))
  squared_part <- tf$reduce_sum(squared_part_Batch)
  
  ## Negative log-likelihood
  Cost2 <- -(logdet_part + squared_part)

   
  list(Cost1 = Cost1,
       Cost2 = Cost2,
       data_in = data_in,
       data_current = data_current,
       data_current_long = data_current_long,
       data_future = data_future,
       data_future_masked = data_future_masked,
       data_future_long = data_future_long,
       D = D,
       D_coef = D_coef,
       D_long = D_long,
       flow_coef = flow_coef,
       K_IDE = K_IDE,
       log_sigma2 = log_sigma2,
       log_rho = log_rho,u_long = u_long,
       SIGMA = SIGMA,
       thisN_Batch = thisN_Batch,
       v_long = v_long,
       Ypred = Ypred,
       Ypred_masked = Ypred_masked)
  
}
