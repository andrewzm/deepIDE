Conv2net <- function(data_in) {

  conv1wts <- tfVar(array(0.01*rnorm(5*5*N_Channels*N_Filters),
                          dim = c(5, 5, N_Channels, N_Filters)), "conv1wts")
  conv1out <- tf$nn$conv2d(input = data_in,
                           filter = conv1wts,
                           strides = c(1L, 1L, 1L, 1L),
                           padding = "SAME")
  biases1 <- tfconst(0.01*rnorm(N_Filters), 'biases')
  
  
  relu1out <- tf$nn$relu(tf$nn$bias_add(conv1out, biases1))
  pool1out = tf$nn$max_pool(relu1out, ksize = c(1L, 3L, 3L, 1L),
                            strides = c(1L, 2L, 2L, 1L),
                            padding = 'SAME', name = 'pool1out')
  norm1out = tf$nn$lrn(pool1out, depth_radius = 4, bias = 1.0, alpha = 0.001 / 9.0, beta=0.75,
                       name='norm1out')
  
  
  conv2wts <- tfVar(array(0.01*rnorm(5*5*N_Filters*N_Filters*2),
                          dim = c(5, 5, N_Filters, N_Filters*2)), "conv2wts")
  conv2out <- tf$nn$conv2d(input = norm1out,
                           filter = conv2wts,
                           strides = c(1L, 1L, 1L, 1L),
                           padding = "SAME")
  
  biases2 <- tfconst(0.01*rnorm(N_Filters*2), 'biases')
  relu2out <- tf$nn$relu(tf$nn$bias_add(conv2out, biases2))
  ## relu2out <- tf$nn$relu(conv2out)
  pool2out = tf$nn$max_pool(relu2out, ksize = c(1L, 3L, 3L, 1L),
                            strides = c(1L, 2L, 2L, 1L),
                            padding = 'SAME', name = 'pool2out')
  norm2outA = tf$nn$lrn(pool2out, depth_radius = 4, bias = 1.0, alpha = 0.001 / 9.0, beta=0.75,
                        name='norm2outA')
  
  
  ## norm2outB <- tf$squeeze(tf$reshape(norm2outA, shape =c(N_Batch,
  ##                                                       as.integer(prod(dim(norm2outA)[-1])), 1L, 1L)))
  norm2outB <- tf$reshape(norm2outA, shape =shape(-1L, prod(unlist(dim(norm2outA)))))
  weights_final <- tfVar(matrix(0.01*rnorm(dim(norm2outB)[[2]]*4), ncol = 4), "wts_final")
  biases <- tfconst(0.01*rnorm(4), 'biases')
  theta_tf = tf$matmul(norm2outB, weights_final) + biases
  
  dilation_par <- tf$exp(theta_tf[,2])
  theta_tf <- tf$stack(list(theta_tf[,1], dilation_par, theta_tf[,3], theta_tf[,4]),
                       axis =1L)
  list(conv1wts = conv1wts,
       conv2wts = conv2wts,
       relu1out = relu1out,
       norm1out = norm1out,
       theta_tf = theta_tf)
}

conv <- function(dd, inchannels, outchannels, kernel_size = 5L, stride = 2L, 
                 bias = FALSE, relu = TRUE, pool = TRUE, norm = TRUE,
                 convwts = NULL, returnwts = FALSE) {
  if(is.null(convwts))
     convwts <- tfVar(array(0.01*rnorm(kernel_size*kernel_size*inchannels*outchannels),
                            dim = c(kernel_size, kernel_size, inchannels, outchannels)), "conv1wts")
  convout <- tf$nn$conv2d(input = dd,
                          filter = convwts,
                          strides = c(1L, 1L, 1L, 1L),
                          padding = "SAME")

  if(norm) {
    normout <- tf$nn$lrn(convout, depth_radius = 4, bias = 1.0, alpha = 0.001 / 9.0, beta=0.75)
  } else {
    normout <- convout
  }
  
  if(bias)  {
      biases <- tfVar(0.01*rnorm(outchannels))
  } else {
      biases <- tfconst(rep(0, outchannels))
  }
 
  if(relu) {
    reluout <- tf$nn$leaky_relu(tf$nn$bias_add(normout, biases))
  } else {
    reluout <- tf$nn$bias_add(normout, biases)
  }
  
  if(pool) {
    poolout <- tf$nn$max_pool(reluout, ksize = c(1L, 3L, 3L, 1L),
                              strides = c(1L, stride, stride, 1L),
                              padding = 'SAME')
  } else {
    poolout <- reluout
  }

  if(!returnwts) {
      poolout
  } else {
      list(out = poolout,
           convwts = convwts)
  }
}

deconv <- function(dd, inchannels, outchannels) {
  newdim <- unlist(dim(dd)[c(2,3)]) * 2L
  conv(dd = tf$image$resize_bilinear(images = dd, size = newdim),
       inchannels, outchannels, kernel_size = 5L, stride = 1L, pool = FALSE)
}


ConvIDEnet <- function(data_in, N_Channels = 3L, N_Filters = 16L, sqrtN_Basis = 6L) {
  
  N_out <- (sqrtN_Basis^2)*2

  conv1 <- conv(data_in, N_Channels, N_Filters) # 32 x 32 x 16
  conv2 <- conv(conv1, N_Filters, N_Filters * 2) # 16 x 16 x 32
  convf <- conv(conv2, N_Filters* 2, N_Filters * 4) # 8 x 8 x 64
  #convf <- conv(conv3, N_Filters* 4, N_Filters * 8) # 4 x 4 x 128


  convf_long <- tf$reshape(convf, shape =shape(-1L, prod(unlist(dim(convf)))))
  convf_long <- tf$nn$dropout(convf_long, keep_prob = 0.5)  
  weights_final <- tfVar(matrix(0.01*rnorm(dim(convf_long)[[2]]*N_out), ncol = N_out), "wts_final")
  flow_coef_unnorm = tf$matmul(convf_long, weights_final)

  ## weights_final <- tfVar(matrix(0.01*rnorm(dim(convf_long)[[2]]), nrow = 1L), "wts_final")
  ## weights_final <- tf$tile(weights_final, c(N_Batch, 1L))
  ## flow_coef_unnorm <- tf$multiply(convf_long, weights_final)

  flow_coef <- ((tf$sigmoid(flow_coef_unnorm) - 0.5) * 2)*5/64 # max/min = 5 pixels  
  list(flow_coef = flow_coef)
}

ConvIDEnetsemi <- function(data_in, N_Channels = 3L, N_Filters = 16L, sqrtN_Basis = 6L) {
  
  N_out <- (sqrtN_Basis^2)

  conv1 <- conv(data_in, N_Channels, N_Filters, returnwts = TRUE) # 32 x 32 x 16
  conv2 <- conv(conv1$out, N_Filters, N_Filters * 2, returnwts = TRUE) # 16 x 16 x 32
  convf <- conv(conv2$out, N_Filters* 2, N_Filters * 4, returnwts = TRUE) # 8 x 8 x 64

  convf_long <- tf$reshape(convf$out, shape =shape(-1L, prod(unlist(dim(convf$out)))))
  convf_long <- tf$nn$dropout(convf_long, keep_prob = 0.3)  
  weights_final <- tfVar(matrix(0.01*rnorm(dim(convf_long)[[2]]*N_out), ncol = N_out), "wts_final")
  u_unnorm = tf$matmul(convf_long, weights_final)

  wts1T <- tf$transpose(conv1$convwts, perm = c(1L, 0L, 2L, 3L))
  conv1T <- conv(data_in, N_Channels, N_Filters, convwts = wts1T) # 32 x 32 x 16
  conv2T <- conv(conv1T, N_Filters, N_Filters * 2, convwts = conv2$convwts) # 16 x 16 x 32
  convfT <- conv(conv2T, N_Filters* 2, N_Filters * 4, convwts = convf$convwts) # 8 x 8 x 64

  convfT_long <- tf$reshape(convfT, shape =shape(-1L, prod(unlist(dim(convfT)))))
  convfT_long <- tf$nn$dropout(convfT_long, keep_prob = 0.3)  
  v_unnorm = tf$matmul(convfT_long, weights_final)
   
  flow_coef_unnorm = tf$concat(list(u_unnorm, v_unnorm), axis = 1L)

  flow_coef <- ((tf$sigmoid(flow_coef_unnorm) - 0.5) * 2)*5/64 # max/min = 5 pixels  
  list(flow_coef = flow_coef)
}

ConvIDEnetsemiV2 <- function(data_in, N_Channels = 3L, N_Filters = 16L,
                             sqrtN_Basis = 6L, kernel_size = 5L) {
  ## Trying to reduce weights to one set per basis function
  N_out <- (sqrtN_Basis^2)
  thisN_Batch <- tf$shape(data_in)[1]
    
  conv1 <- conv(data_in, N_Channels, N_Filters, returnwts = TRUE, kernel_size = kernel_size) # 32 x 32 x 16
  conv2 <- conv(conv1$out, N_Filters, N_Filters * 2, returnwts = TRUE, kernel_size = kernel_size) # 16 x 16 x 32
  convf <- conv(conv2$out, N_Filters* 2, N_Filters * 4, returnwts = TRUE, kernel_size = kernel_size) # 8 x 8 x 64

  convf_long <- tf$reshape(convf$out, shape = shape(-1L, 64L, N_Filters * 4))
  weights_final <- tfVar(matrix(0.01*rnorm(N_Filters * 4), ncol = 1L), "wts_final")
  weights_final <- tf$expand_dims(weights_final, 0L)
  weights_final <- tf$tile(weights_final, c(thisN_Batch, 1L, 1L))
  u_unnorm_mixed = tf$matmul(convf_long, weights_final)
  
  demixer <- tfVar(array(0.01*rnorm(N_out^2), dim = c(1L, N_out, N_out)))
  demixer <- tf$tile(demixer, c(thisN_Batch, 1L, 1L))
  u_unnorm = tf$matmul(demixer, u_unnorm_mixed)
  
  wts1T <- tf$transpose(conv1$convwts, perm = c(1L, 0L, 2L, 3L))
  conv1T <- conv(data_in, N_Channels, N_Filters, convwts = wts1T) # 32 x 32 x 16
  conv2T <- conv(conv1T, N_Filters, N_Filters * 2, convwts = conv2$convwts) # 16 x 16 x 32
  convfT <- conv(conv2T, N_Filters* 2, N_Filters * 4, convwts = convf$convwts) # 8 x 8 x 64
  
  convfT_long <- tf$reshape(convfT,  shape =shape(-1L, 64L, N_Filters * 4))
  v_unnorm_mixed = tf$matmul(convfT_long, weights_final)
  v_unnorm <- tf$matmul(demixer, v_unnorm_mixed)
   
  flow_coef_unnorm = tf$concat(list(u_unnorm, v_unnorm), axis = 1L)
  flow_coef_unnorm <- tf$squeeze(flow_coef_unnorm, axis = 2L)
  flow_coef <- ((tf$sigmoid(flow_coef_unnorm) - 0.5) * 2)*5/64 # max/min = 5 pixels  
  list(flow_coef = flow_coef)
}


ConvDeconvnet <- function(data_in, N_Channels, N_Filters) {
  
  
  conv_chain <- list(conv(data_in, N_Channels, N_Filters))
  i <- 1
  while(!(dim(conv_chain[[i]])[[2]] == 2L)) {
    i <- i + 1
    conv_chain[[i]] <- conv(conv_chain[[i - 1]], N_Filters*(2^(i - 2)), N_Filters*2^(i - 1))
  }
  
  deconv_chain <- catimages <- list()
  deconv_chain[[i]] <- deconv(conv_chain[[i]], N_Filters*2^(i - 1), N_Filters*(2^(i - 3)))
  while(i >= 2) {
    i <- i - 1
    catimages[[i]] <- tf$concat(list(conv_chain[[i]], deconv_chain[[i + 1]]), axis = 3L)
    deconv_chain[[i]] <- deconv(catimages[[i]], N_Filters*2^(i - 1) + N_Filters*(2^(i - 2)), N_Filters*(2^(i - 3)))
  }
  
  finalcat <- tf$concat(list(data_in, deconv_chain[[i]]), axis = 3L)
  flow <- conv(dd = finalcat, inchannels = N_Filters*(2^(i - 3)) + N_Channels, outchannels = 4L,
               kernel_size = 5L, stride = 1L, relu = FALSE, pool =  FALSE, norm = FALSE)
  #biases <- tfconst(0.01*rnorm(4), 'biases')
  #theta_tf = tf$matmul(norm2outB, weights_final) + biases
  
  dilation_par <- tf$exp(flow[,,,2,drop=FALSE])
  flow <- tf$stack(list(flow[,,,1,drop=FALSE], dilation_par, 
                        flow[,,,3,drop=FALSE], flow[,,,4,drop=FALSE]),
                   axis = 3L) %>% tf$squeeze(4L)
  list(conv_chain = conv_chain,
       deconv_chain = deconv_chain,
       flow = flow)
}

ConvIDEnet_diffusion <- function(data_in, N_Channels = 3L, N_Filters = 16L,
                                 sqrtN_Basis = 6L, kernel_size = 5L) {
  
  N_out <- (sqrtN_Basis^2)
  thisN_Batch <- tf$shape(data_in)[1]
  
  conv1 <- conv(data_in, N_Channels, N_Filters, kernel_size = kernel_size) # 32 x 32 x 16
  conv2 <- conv(conv1, N_Filters, N_Filters * 2, kernel_size = kernel_size) # 16 x 16 x 32
  convf <- conv(conv2, N_Filters* 2, N_Filters * 4, kernel_size = kernel_size) # 8 x 8 x 64
  
  convf_long <- tf$reshape(convf, shape = shape(-1L, 64L, N_Filters * 4))
  weights_final <- tfVar(matrix(0.01*rnorm(N_Filters * 4), ncol = 1L), "wts_final")
  weights_final <- tf$expand_dims(weights_final, 0L)
  weights_final <- tf$tile(weights_final, c(thisN_Batch, 1L, 1L))
  D_coef_unnorm_mixed = tf$matmul(convf_long, weights_final)
  
  demixer <- tfVar(array(0.01*rnorm(N_out^2), dim = c(1L, N_out, N_out)))
  demixer <- tf$tile(demixer, c(thisN_Batch, 1L, 1L))
  D_coef_unnorm = tf$matmul(demixer, D_coef_unnorm_mixed)
  
  D_coef_unnorm <- tf$squeeze(D_coef_unnorm, axis = 2L)
  D_coef <- tf$sigmoid(D_coef_unnorm)*tfconst(0.0001)
  
  list(D_coef = D_coef)
}

ConvIDEnet_sigmapars <- function(data_in, N_Channels = 3L, N_Filters = 16L,
                                 sqrtN_Basis = 6L, kernel_size = 5L) {
  
  N_out <- 1L
  thisN_Batch <- tf$shape(data_in)[1]

  conv1 <- conv(data_in, N_Channels, N_Filters, kernel_size = kernel_size, returnwts = TRUE) # 32 x 32 x 16
  conv2 <- conv(conv1$out, N_Filters, N_Filters * 2, kernel_size = kernel_size, returnwts = TRUE) # 16 x 16 x 32
  conv3 <- conv(conv2$out, N_Filters* 2, N_Filters * 4, kernel_size = kernel_size, returnwts = TRUE) # 8 x 8 x 64
  conv4 <- conv(conv3$out, N_Filters* 4, N_Filters * 8, kernel_size = kernel_size, returnwts = TRUE) # 4 x 4 x 128
  conv5 <- conv(conv4$out, N_Filters* 8, N_Filters * 16, kernel_size = kernel_size, returnwts = TRUE) # 2 x 2 x 256
  conv6 <- conv(conv5$out, N_Filters* 16, N_Filters * 32, kernel_size = kernel_size, returnwts = TRUE) # 1 x 1 x 512

  conv6_long <- tf$reshape(conv6$out, shape = shape(-1L, 1L, N_Filters * 32))
  weights_final_var <- tfVar(matrix(0.01*rnorm(N_Filters * 32), ncol = 1L), "wts_final")
  weights_final <- tf$expand_dims(weights_final_var, 0L)
  weights_final <- tf$tile(weights_final, c(thisN_Batch, 1L, 1L))
  LogCovPar <- tf$matmul(conv6_long, weights_final) - 2.0

  list(LogCovPar = LogCovPar,
         wts = list(conv1$convwts, conv2$convwts, conv3$convwts,
                    conv4$convwts, conv5$convwts, conv6$convwts,
                    weights_final_var))
}


reshape_4d_to_3d <- function(x, mid_dim) {
  tf$reshape(tf$transpose(x, perm = c(0L, 2L, 1L, 3L)), c(-1L, mid_dim, 1L))
}

reshape_3d_to_2d <- function(x, mid_dim) {
  tf$reshape(tf$transpose(x, perm = c(0L, 2L, 1L)), c(-1L, mid_dim, 1L))
}
