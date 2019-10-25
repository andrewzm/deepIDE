W <- 64L                    # 64
H <- 64L                    # 64
tau <- N_Channels <- 3L     # 3 channels
patch_size_flow <- 5L       # kernel size for flow convs
patch_size_diff <- 5L       # kernel size for diffusion convs
sqrtN_Basis <- 8L           # sqrt of number of basis functions for IDE vector fields
border_mask <- 0.1          # percentage buffer around domain
N_Filters <- 64L         # number of convolution filters in first layer
s1 <- seq(0, 1, length.out = W)  # grid points for s1
s2 <- seq(0, 1, length.out = H)  # grid points for s2
sgrid <- expand.grid(s1 = s1, s2 = s2)  # spatial grid in long format
