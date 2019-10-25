####################################################
## Title: Stage 1 fitting for the CNN-IDE (CNN pars)
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
####################################################

## Load packages and source files
library("tensorflow")
library("dplyr")
library("lubridate")
library("Matrix")
library("tidyr")
library("R.utils")
sourceDirectory("../common") 

############################
## Part 1: Construct the CNN
############################
CNNgraph <- createCNNgraph(W = W,  H = H,
                           N_Channels = N_Channels,
                           N_Filters = N_Filters,
                           sqrtN_Basis = sqrtN_Basis, 
                           patch_size_flow = patch_size_flow,
                           patch_size_diff = patch_size_diff,
                           border_mask = border_mask)
list2env(CNNgraph, envir = .GlobalEnv)

######################################################
## Part 2: Load data and massage into required formats
######################################################
load("../1_Preproc_data/intermediates/TrainingData3D.rda")
load("../1_Preproc_data/intermediates/TrainingDataFinals.rda")
load("../1_Preproc_data/intermediates/TrainingDataPreds.rda")

## Index notes: Note that first day in each zone is the day 2006-12-27
##   We index by first day of the 4-day (tau = 3) batch which is 2006-12-27
##   We have 4000 test/val data points (incl. 12/27), i.e., up to 2017-12-08
##   We have 456 test data points, i.e., up to 2019-03-09
##   Note that this includes data up to 2019-03-12 (since 2019-03-09 contains 4 images,
##   three used for analysis and one for forecast/traning)

## For the "current day", we store the third image, i.e., the first record
## in this data set is of 2006-12-29, and goes up to 2019-03-11
## For the "future day", we store the fourth image, i..e, the first record
## in this data set is of 2006-12-30, and goes up to 2019-03-12.


###############################
## Part 3.1: Training the graph
###############################
set.seed(1)                                         # set seed
source("../common/CNN_fit_vars.R")
nepochs <- 30                                       # 30 epochs
nsteps_per_epoch <- floor(N_Data_Train / N_Batch)   # number of steps per epoch (4275)
init_learning_rate <- 0.00005                       # learning rate for CNN weights
init_learning_rate_cov <- 0.0000005                 # learning rate for cov pars

## Optimiser for IID residuals
trainnet <- tf$train$AdamOptimizer(init_learning_rate)$minimize(Cost1)

## Optimiser for covariance function parameters
trainnetcov <- (tf$train$GradientDescentOptimizer(init_learning_rate_cov))$
    minimize(Cost2,
             var_list = list(log_sigma2, log_rho))

## Store cost function values in these lists
Objective <- Objective_val <- list()  

## Store summary stats of each epoch in this data frame
Epoch_train <- Epoch_val <- data.frame(mean = rep(0, nepochs),
                                       median = rep(0, nepochs),
                                       sd = rep(0, nepochs),
                                       mad = rep(0, nepochs))

## Initialise global variables
init <- tf$global_variables_initializer()
run(init)
options(scipen = 3, digits = 3)

## For each epoch
for(epoch in 1:nepochs) {
    
    ## Do all validation cases
    cat("Running validation... \n")
    N_Val_Batches <- floor(length(idxVal)/32L)            # about 237
    Val_Batch_idx <- rep(1:N_Val_Batches, each = 32L)     # batch ID
    if((partbatch <- length(idxVal) %% 32L) > 0) {        # if not an exact multiple of 32
        N_Val_Batches <- N_Val_Batches + 1           # then there is an extra batch
        Val_Batch_idx <- c(Val_Batch_idx, rep(N_Val_Batches, partbatch))
    }
    
    ## Create an idx--batch map
    val_batches <- data.frame(idx = idxVal,
                              batch = Val_Batch_idx)

    ## For each batch compute the NLL
    ValCosts <- lapply(unique(val_batches$batch), function(i) {
        idx_val <- filter(val_batches, batch == i)$idx
        fd <- dict(data_in = d[idx_val,,,],
                   data_current = dfinal[idx_val,,,drop=F],
                   data_future = dpred[idx_val,,,drop=F])
        run(Cost1, feed_dict = fd)
    })
    
    ## Compute summary statistics at this epoch
    Epoch_val[epoch, ] <- c(mean(unlist(ValCosts)),
                            median(unlist(ValCosts)),
                            sd(unlist(ValCosts)),
                            mad(unlist(ValCosts)))
    
    ## Initialise
    epoch_order <- idxTrain
    count <- 0
    Objective[[epoch]] <- Objective_val[[epoch]] <- rep(0, nsteps_per_epoch)
    
    while(length(epoch_order) >= N_Batch) {
        
        train_cov_pars <- (epoch >= (28L))       # Estimate cov. pars if epoch >= 28
        count <- count + 1                       # increment batch number in epoch
        idx <- epoch_order[1:N_Batch]            # Take first N_Batch
        epoch_order <- epoch_order[-(1:N_Batch)] # Remove first batch from list
        fd <- dict(data_in = d[idx,,,],          # Creat dictionary
                   data_current = dfinal[idx,,,drop=F],
                   data_future = dpred[idx,,,drop=F])
        
        if(train_cov_pars) {
            TFrun <- run(list(trainnetcov, Cost1), feed_dict = fd)
        } else {
            TFrun <- run(list(trainnet, Cost1), feed_dict = fd) # Train network on its own  
        }
        
        ## Get Cost for this batch
        Objective[[epoch]][count] <- TFrun[[2]]
        
        ## Every 10 samples do a random validation check
        if((count %% 10) == 0) {
            idx_val <- sample(idxVal, N_Batch, replace = FALSE)
            fd <- dict(data_in = d[idx_val,,,],
                       data_current = dfinal[idx_val,,,drop=F],
                       data_future = dpred[idx_val,,,drop=F])
            Objective_val[[epoch]][count/10] <- run(Cost1, feed_dict = fd)
            cat(paste0("Epoch ", epoch, " ...  Step ", count, 
                       " ... Cost: ", Objective_val[[epoch]][count/10], "\n"))
        }

    Epoch_train[epoch, ] <- c(mean(Objective[[epoch]]),
                              median(Objective[[epoch]]),
                              sd(Objective[[epoch]]),
                              mad(Objective[[epoch]]))

    }
}

## Save all variables
Trained_Vars <- run(tf$trainable_variables(scope = NULL))
save(Trained_Vars, file = "./intermediates/SSTIDE_weights_TF.rda")
save(nepochs, Objective, Objective_val, file = "./intermediates/Objective_series.rda")
