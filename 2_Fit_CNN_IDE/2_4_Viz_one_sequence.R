## Load libraries and functions
library("tensorflow")
library("dplyr")
library("ggplot2")
library("R.utils")
sourceDirectory("../common")

## Plot some sample images from the trained CNN
CNNgraph <- createCNNgraph(W = W,  H = H,
                           N_Channels = N_Channels,
                           N_Filters = N_Filters,
                           sqrtN_Basis = sqrtN_Basis, 
                           patch_size_flow = patch_size_flow,
                           patch_size_diff = patch_size_diff,
                           border_mask = border_mask)
list2env(CNNgraph, envir = .GlobalEnv)

load(file = "./intermediates/SSTIDE_weights_TF.rda")
run <- tf$Session()$run
All_Vars_tf <- tf$trainable_variables(scope = NULL)
for(i in 1:length(All_Vars_tf)) {
    run(tf$assign(All_Vars_tf[[i]], Trained_Vars[[i]]))
 }

## Load the data
load("../1_Preproc_data/intermediates/TrainingData3D.rda")
load("../1_Preproc_data/intermediates/TrainingDataFinals.rda")
load("../1_Preproc_data/intermediates/TrainingDataPreds.rda")

## Choose sequence number
idx <- 1000

## Create scale
myscale <- scale_fill_gradientn(colours = nasa_palette, name = "Z")

## Create dictionary
fd <- dict(data_in = d[idx,,,,drop = F],          
           data_current = dfinal[idx,,,drop=F],
           data_future = dpred[idx,,,drop=F])

## Add data to grid
sgrid$Y1 <- as.numeric(run(data_in[1,,,1], feed_dict = fd))
sgrid$Y2 <- as.numeric(run(data_in[1,,,2], feed_dict = fd))
sgrid$Y3 <- as.numeric(run(data_in[1,,,3], feed_dict = fd))
sgrid$Y4 <- as.numeric(run(data_future_long[1,,], feed_dict = fd))
sgrid$Y4hat <- as.numeric(run(Ypred[1,,], feed_dict = fd))

## Plot and save data
ggsave(ggplot(sgrid) + geom_tile(aes(s1,s2,fill=Y1)) + myscale + 
       coord_fixed(), file = "img/Y1.png", width = 8, height = 8)
ggsave(ggplot(sgrid) + geom_tile(aes(s1,s2,fill=Y2)) + myscale + 
       coord_fixed(), file = "img/Y2.png", width = 8, height = 8)
ggsave(ggplot(sgrid) + geom_tile(aes(s1,s2,fill=Y3)) + myscale + 
       coord_fixed(), file = "img/Y3.png", width = 8, height = 8)
ggsave(ggplot(sgrid) + geom_tile(aes(s1,s2,fill=Y4)) + myscale + 
       coord_fixed(), file = "img/Y4.png", width = 8, height = 8)
ggsave(ggplot(sgrid) + geom_tile(aes(s1,s2,fill=Y4hat)) + myscale + 
       coord_fixed(), file = "img/Y4hat.png", width = 8, height = 8)

## Illustrate one of the CNN layers
XX <- expand.grid(s1 = 1:5, s2 = 1:5, dim = 1:3, flt = 1:64)
XX$val <- as.numeric(run(All_Vars_tf[[1]]))
ggplot(filter(XX, flt  < 21)) + geom_tile(aes(s1, s2, fill = val)) +
  facet_grid(dim ~ flt) + scale_fill_distiller(palette = "Spectral")
