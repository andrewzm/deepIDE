library(tensorflow)

## Kernel -- 2x2 filters with 2 input channels and 3 output channels
## Think of it as 3 2x2x2 filters
k <- array(c(rep(1,4), rep(2, 4), rep(3, 4),
             rep(4, 4), rep(5, 4), rep(6, 4)), dim =  c(2,2,2,3))
k_tf <- tf$constant(k, dtype = "float32")

## Image -- 3x3 with 2 input channels
im <- array(rnorm(18), dim = c(1,3,3,2))
im_tf <- tf$constant(im, dtype = "float32")

## Convolving gives 3 2x2 images
out <- tf$nn$conv2d(input = im_tf, filter = k_tf,
             strides = c(1L, 1L, 1L, 1L), padding = "VALID")

run <- tf$Session()$run
run(out)

## Is this the same as as convolving the first channel with the first set of
## kernels, the second with the second, and then adding?

k1 <- k[,,1,1:3, drop = FALSE]
k2 <- k[,,2,1:3, drop = FALSE]
k1_tf <- tf$constant(k1, dtype = "float32")
k2_tf <- tf$constant(k2, dtype = "float32")

im1 <- im[1,,,1, drop = FALSE]
im2 <- im[1,,,2, drop = FALSE]
im1_tf <- tf$constant(im1, dtype = "float32")
im2_tf <- tf$constant(im2, dtype = "float32")

out2 <- tf$nn$conv2d(input = im1_tf, filter = k1_tf,
                    strides = c(1L, 1L, 1L, 1L), padding = "VALID") +
  tf$nn$conv2d(input = im2_tf, filter = k2_tf,
               strides = c(1L, 1L, 1L, 1L), padding = "VALID")

## Yes, which is good
summary(run(out - out2))

## This means the we have a different kernel per channel, which is what I want
## (It's not the SAME kernel applied to each channel)
