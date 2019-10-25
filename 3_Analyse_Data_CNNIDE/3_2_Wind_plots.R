#######################################################
## Title: Visualise wind speeds in spherical histograms
## Date: 22 October 2019
## Author: Andrew Zammit-Mangion
#######################################################
library("tensorflow")
library("dplyr")
library("tidyr")
library("fields")
library("ggplot2")
library("gridExtra")
library("gstat")
library("Matrix")
library("lubridate")
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


## Load weights from Stage 1 and assign to graph
load(file = "../2_Fit_CNN_IDE/intermediates/SSTIDE_weights_TF.rda")
run <- tf$Session()$run
All_Vars_tf <- tf$trainable_variables(scope = NULL)
for(i in 1:length(All_Vars_tf)) {
  run(tf$assign(All_Vars_tf[[i]], Trained_Vars[[i]]))
}

## Create spatial grid
s1 <- seq(0, 1, length.out = W)  # grid points for s1
s2 <- seq(0, 1, length.out = H)  # grid points for s2
sgrid <- expand.grid(s1 = s1, s2 = s2)  # spatial grid in long format

## Replace updated with forecasts to see effect of update on
## dynamics
for(fname in c("intermediates/data_for_dir_plots_Zone11_i10.rda",
               "intermediates/data_for_dir_plots_Zone11_i11.rda",
               "intermediates/data_for_dir_plots_Zone11_i12.rda")) {

    load(fname)
    print(sum(c(ParticleForecasts) - c(Pframe[,,,tau])))
    print(sum(c(ParticlePreds) - c(Pframe[,,,tau])))
    Pframe2 <- Pframe
    Pframe2[,,,tau] <- array(c(ParticleForecasts),
                             dim = c(nParticles, 64L, 64L))

    fd1 <- dict(data_in = Pframe)
    fd2 <- dict(data_in = Pframe2)

    ## Extract forecast and updated ensemble
    U1 <- -run(u_long, feed_dict = fd1)
    V1 <- -run(v_long, feed_dict = fd1)
    angles_df1 <- as.data.frame(t((atan2(V1,U1)*360/(2*pi))[,,1])) %>%
        cbind(sgrid) %>%
        gather(Sim, Angle, -s1, -s2) %>%
        mutate(Angle = ifelse(Angle < 0, Angle + 360, Angle),
               Group = "Updated")

    U2 <- -run(u_long, feed_dict = fd2)
    V2 <- -run(v_long, feed_dict = fd2)
    angles_df2 <- as.data.frame(t((atan2(V2,U2)*360/(2*pi))[,,1])) %>%
        cbind(sgrid) %>%
        gather(Sim, Angle, -s1, -s2) %>%
        mutate(Angle = ifelse(Angle < 0, Angle + 360, Angle),
               Group = "Forecast")

    angles_df <- rbind(angles_df1, angles_df2)
    angles_df$Group <- as.factor(angles_df$Group)

    s1sub <- unique(sgrid$s1)[seq(8, 64, by = 16)]
    s2sub <- unique(sgrid$s2)[seq(8, 64, by = 16)]
    sgrid_sub <- expand.grid(s1sub, s2sub)
    gempty <- ggplot(data.frame(x = -40, y = 42)) +
        geom_point(aes(x,y), col = "white") +
        coord_fixed(xlim = c(-43.333, -38.0833),
                    ylim = c(40.333 ,45.5833)) + theme_bw() +
        theme(text = element_text(size=20)) +
            xlab("Longitude (deg)") + ylab("Latitude (deg)")

    gall <- gempty
    for(i in 1:nrow(sgrid_sub)) {

        dszone <- (43.333 - 38.083)
        angles_df_sub <- filter(angles_df, s1 == sgrid_sub[i,1] &
                                           s2 == sgrid_sub[i,2]) %>%
            mutate(s1 = s1*dszone - 43.333,
                   s2 = s2*dszone + 40.333)

        gi <- ggplot() +
            geom_histogram(data = filter(angles_df_sub, Group == "Updated"),
                           aes(x = Angle), colour = "black", breaks = seq(0,360, by = 30),
                           alpha = 0.5,
                           position = "identity") +
            coord_polar(start = 0) + theme_minimal() + ylab("") + 
            scale_x_continuous("", limits = c(0, 360),
                               breaks = seq(0, 360, by = 45)) +
            theme(text = element_blank(),
                  title = element_blank(),
                  legend.position = "none") 

        givp <- ggplotGrob(gi)
        gall <- gall + annotation_custom(grob = givp,
                                         xmin = angles_df_sub$s1[1] - dszone/9,
                                         xmax = angles_df_sub$s1[1] + dszone/9,
                                         ymin = angles_df_sub$s2[1] - dszone/9,
                                         ymax = angles_df_sub$s2[1] + dszone/9) 
    }
    ggsave(gall, file = paste0("./img/", tools::file_path_sans_ext(basename(fname)),".png"))
}
