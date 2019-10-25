library("dplyr")
tau <- 3L                   # lags to consider
nT <- 4456L                 # total number of time points
nT_Train_Val <- 4000L       # time points for training + validatino
nT_Test <- 4456L - 4000L    # time points for testing
nZones <- 19L               # number of spatial zones
N_Data <- nT * nZones       # total number of data points (nT * nZones)
idxTrain_Val <- rep(1:nT_Train_Val, nZones) +      # indices of train/val data
    rep((0:(nZones-1)*nT), each = nT_Train_Val)
idxTest <- setdiff(1:N_Data, idxTrain_Val)         # indices of test data
N_Data_Train <- round(length(idxTrain_Val) * 9 / 10) # number of training data
N_Data_Val <- round(length(idxTrain_Val) * 1 / 10)   # number of val data
N_Batch <- 16L           # minibatch size in SGD

## Split the non-test data into training and validation
set.seed(1)
idxTrain <- sample(idxTrain_Val, N_Data_Train)
idxVal <- setdiff(idxTrain_Val, idxTrain)

Date_Zone_Map <- tibble(idx = 1:N_Data,
                        t = rep(1:nT, nZones),
                        zone = rep(1:nZones, each = nT)) %>%
  left_join(tibble(t = 1:nT,
                   startdate = as.Date("2006-12-26") + 1:nT,
                   currentdate = startdate + 2,
                   futuredate = startdate + 3))
