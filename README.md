# Deep Integro-Difference Equation Models for Spatio-Temporal Forecasting

<img align="right" src="https://github.com/andrewzm/deepIDE/blob/master/2_Fit_CNN_IDE/img/BallResults2.png" alt="drawing" width="400"/>


This GitHub page provides code for reproducing the results in the manuscript titled [*Deep Integro-Difference Equation Models for Spatio-Temporal Forecasting* by A. Zammit-Mangion and C.K. Wikle](https://arxiv.org/abs/1910.13524). The manuscript describes the use of Convolution Neural Networks (CNNs) to learn about the dynamic evolution of environmental processes. Once the CNN is trained using analyses data (i.e., full, complete, data) of a certain phenomenon, it can be used within statistical models for analysing or forecasting other environmental phenomena that exhibit similar physical behaviour.

The arrows in the figure on the right show the output of the CNN in response to a spatio-temporal input depicted by the red contours (where transparency is used to denote time). Note how the CNN outputs a flow direction which is reasonable based on the spatio-temporal evolution of the red contours. The CNN we use is based on that of de Bezenac (2018).

## Instructions

To reproduce the results please download this repository. Add a subfolder data/ to 1_Preproc_data/ and put in it the SST netCDF file, which can be downloaded from [here](https://hpc.niasra.uow.edu.au/azm/global-analysis-forecast-phy-001-024_1551608429013.nc). Instructions on how this netCDF file was generated is available in 1_Preproc_data/README.txt.

After putting in the data, cycle through the folders 1_, 2_, etc. running through the code in each folder in order. The code populates the img/ and intermediates/ directories, the contents of which are either used in the paper or by subsequent code.


## Abstract

Integro-difference equation (IDE) models describe the conditional dependence between the spatial process at a future time point and the process at the present time point through an integral operator. Nonlinearity or temporal dependence in the dynamics is often captured by allowing the operator parameters to vary temporally, or by re-fitting a model with a temporally-invariant linear operator in a sliding window. Both procedures tend to be excellent for prediction purposes over small time horizons, but are generally time-consuming and, crucially, do not provide a global prior model for the temporally-varying dynamics that is realistic.  Here, we tackle these two issues by using a deep convolution neural network (CNN) in a hierarchical statistical IDE framework, where the CNN is designed to extract process dynamics from the process' most recent behaviour. Once the CNN is fitted, probabilistic forecasting can be done extremely quickly online using an ensemble Kalman filter with no requirement for repeated parameter estimation. We conduct an experiment where we train the model using 13 years of daily sea-surface temperature data in the North Atlantic Ocean. Forecasts are seen to be accurate and calibrated. A key advantage of our approach is that the CNN provides a global prior model for the dynamics that is realistic, interpretable, and computationally efficient. We show the versatility of the approach by successfully producing 10-minute nowcasts of weather radar reflectivities in Sydney using the same model that was trained on daily sea-surface temperature data in the North Atlantic Ocean.

## Videos

All images can be viewed in the img/ subdirectories in this repository. There are also two animations which can be viewed:

1. An animation of the SST product in the North Atlantic ([link](http://htmlpreview.github.com/?https://github.com/andrewzm/deepIDE/blob/master/1_Preproc_data/anim/SST_anim.html))

2. An animation of predictions and forecasts in Zone 1 for out-of-sample data ([link](http://htmlpreview.github.com/?https://github.com/andrewzm/deepIDE/blob/master/5_Extract_Results/anim/SST_filter.html))

## Software versions and hardware used to generate the results

### R

R 3.6.1

verification_1.42 dtw_1.20-1        proxy_0.4-22      CircStats_0.2-6
MASS_7.3-51.1     boot_1.3-20       IDE_0.2.0         FRK_0.2.2
gridExtra_2.3     fields_9.6        maps_3.3.0        spam_2.2-0
dotCall64_1.0-0   STRbook_0.1.0     sp_1.3-1          spacetime_1.2-2
gstat_1.1-6       reticulate_1.10   ncdf4_1.16        animation_2.5
ggquiver_0.1.0    ggplot2_3.1.0     R.utils_2.6.0     R.oo_1.22.0
R.methodsS3_1.7.1 tidyr_0.8.1       Matrix_1.2-17     lubridate_1.7.4
dplyr_0.7.6       tensorflow_1.9
DEoptim_2.2-4       jsonlite_1.5        splines_3.6.1
Formula_1.2-3       assertthat_0.2.1    latticeExtra_0.6-28
pillar_1.3.1        backports_1.1.2     lattice_0.20-38
glue_1.3.0          digest_0.6.18       RColorBrewer_1.1-2
checkmate_1.8.5     colorspace_1.3-2    htmltools_0.3.6
plyr_1.8.4          pkgconfig_2.0.2     purrr_0.2.5
scales_1.0.0        intervals_0.15.1    whisker_0.3-2
tibble_2.1.1        htmlTable_1.12      withr_2.1.2
nnet_7.3-12         lazyeval_0.2.2      survival_2.43-3
magrittr_1.5        crayon_1.3.4        xts_0.11-0
foreign_0.8-70      FNN_1.1.2           data.table_1.12.0
tools_3.6.1         sparseinv_0.1.3     stringr_1.3.1
munsell_0.5.0       cluster_2.0.7-1     bindrcpp_0.2.2
compiler_3.6.1      rlang_0.3.2         rstudioapi_0.7
htmlwidgets_1.3     base64enc_0.1-3     gtable_0.3.0
R6_2.2.2            tfruns_1.3          zoo_1.8-3
knitr_1.20          bindr_0.1.1         Hmisc_4.1-1
stringi_1.2.4       parallel_3.6.1      Rcpp_0.12.18
rpart_4.1-13        acepack_1.4.1       tidyselect_0.2.4

### Python

Python 3.6.7

tensorflow 1.12.0

numpy 1.15.0

scipy 1.2.0

### GPU

CUDA V9.0.176

nVIDIA Driver Version: 390.67

### Hardware

Intel Core i9-7900X @ 3.30GHz

GeForce GTX 1080Ti

64GB RAM


## References

de Bezenac, E., Pajot, A., Gallinari, P., 2018. Deep learning for physical processes: Incorporating
prior scientic knowledge. In: Proceedings of ICLR 2018. Vancouver, Canada.
