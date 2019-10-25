###########################################################
## Title: Visualise dynamics by a moving GRBF in space/time
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
##########################################################
library("tensorflow")
library("dplyr")
library("ggplot2")
library("ggquiver")
library("R.utils")
sourceDirectory("../common") 

## Construct the graph
CNNgraph <- createCNNgraph(W = W,  H = H,
                           N_Channels = N_Channels,
                           N_Filters = N_Filters,
                           sqrtN_Basis = sqrtN_Basis, 
                           patch_size_flow = patch_size_flow,
                           patch_size_diff = patch_size_diff,
                           border_mask = border_mask)
list2env(CNNgraph, envir = .GlobalEnv)

## Load weights and assign to graph
load(file = "./intermediates/SSTIDE_weights_TF.rda")
run <- tf$Session()$run
All_Vars_tf <- tf$trainable_variables(scope = NULL)
for(i in 1:length(All_Vars_tf)) {
  run(tf$assign(All_Vars_tf[[i]], Trained_Vars[[i]]))
}

## Extract convolution output of first layer of v
convfilter_u <- tf$trainable_variables(scope = NULL)[[1]]
convfilter_v <- tf$transpose(convfilter_u, perm = c(1L, 0L, 2L, 3L))
conv1 <- conv(data_in, 3L, N_Filters, convwts = convfilter_v) # 32 x 32 x 16
set.seed(5)
for(i in 1:10) {
  
  ## Centres,directions, and SD of GRBF
  s1c <- runif(1, min = 0.25, max = 0.75)
  s2c <- runif(1, min = 0.25, max = 0.75)
  ds1 <- runif(1, max = 0.1) - 0.05
  ds2 <- runif(1, max = 0.1) - 0.05
  dD <- 1
  
  ## Create the space-time 
  ball <- mutate(sgrid, 
                 z1 = dnorm(s1, s1c, sqrt(0.001))*dnorm(s2, s2c, sqrt(0.001)),
                 z2 = dnorm(s1, s1c - ds1, sqrt(0.001*(dD^2)))*
                      dnorm(s2, s2c - ds2, sqrt(0.001*(dD^2)))/max(z1),
                 z3 = dnorm(s1, s1c - 2*ds1, sqrt(0.001*(dD^2)))*
                      dnorm(s2, s2c - 2*ds2, sqrt(0.001*(dD^2)))/max(z1),
                 z1 = z1 / max(z1))
  
  ## Put ball into 3D array
  ball_im <- array(c(ball$z1, ball$z2, ball$z3), dim = c(1, 64, 64, 3))
  
  ## Feed this ball to the CNN and look at the filter outputs of the first layer
  fd <- dict(data_in = ball_im)
  XX <- run(conv1, feed_dict = fd)
  filterout <- expand.grid(s1 = 1:32, s2 = 1:32, channel = 1:64) %>%
               mutate(value = as.numeric(XX))
  ggplot(filterout) + geom_tile(aes(s1, s2, fill = value)) + facet_wrap(~channel) +
      scale_fill_gradientn(colours = nasa_palette)
  
  sgrid$U <- as.numeric(run(u_long, feed_dict = fd))
  sgrid$V <- as.numeric(run(v_long, feed_dict = fd))
  sgrid$D <- as.numeric(run(D_long, feed_dict = fd))
  s1sub <- unique(sgrid$s1)[seq(1,64,by = 4)]
  s2sub <- unique(sgrid$s2)[seq(1,64,by = 4)]
  g1 <- ggplot(ball) + geom_contour(aes(s1, s2, z = z1), alpha = 0.2, colour = "black") +
    geom_contour(aes(s1, s2, z = z2), alpha = 0.5, colour = "black") +
    geom_contour(aes(s1, s2, z = z3), colour = "black") + theme_bw() +
    coord_fixed(xlim = c(0,1), ylim = c(0,1))
  g2 <- ggplot(filter(sgrid, s1 %in% s1sub & s2 %in% s2sub)) + 
    geom_contour(data = ball, aes(s1, s2, z = z1), alpha = 0.2, colour = "red") +
    geom_contour(data = ball, aes(s1, s2, z = z2), alpha = 0.5, colour = "red") +
    geom_contour(data = ball, aes(s1, s2, z = z3), colour = "red") +
    geom_quiver(aes(s1, s2, u = -U, v = -V,# colour = atan2(-V, -U), 
                    colour = (sqrt(U^2 + V^2))), vecsize = 1) + 
    scale_colour_distiller(palette = "Greys", trans = "reverse", name = "Amplitude") +
    theme_bw() + coord_fixed() + xlab(expression(s[1])) + 
    ylab(expression(s[2])) + theme(text = element_text(size = 18)) +
    #scale_colour_distiller(name = "Angle (deg)") +
    theme(legend.position = NULL)
  g3 <- ggplot(sgrid) + geom_tile(aes(s1, s2, fill = D)) + 
    scale_fill_gradientn(colours = nasa_palette) + coord_fixed() +
    theme_bw() 
  
  #gall <- grid.arrange(g1,g2,g3, nrow = 1, widths = c(0.3, 0.3, 0.37))
  #gall <- grid.arrange(g1,g2, nrow = 1, widths = c(0.45, 0.55))
  ggsave(g2, file = paste0("img/BallResults", i, ".png"), width = 8, height = 6)
}
