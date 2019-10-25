library("animation")
library("dplyr")
library("ggplot2")
library("gridExtra")
library("tidyr")
library("verification")
library("R.utils")
sourceDirectory("../common")

FilterResults <- FcastResults <- NULL
for(zone in 1:nZones) {

    load(paste0("../3_Analyse_Data_CNNIDE/intermediates/Results_CNNIDE_Zone_", zone, ".rda"))
    load(paste0("../4_Analyse_Data_Other/intermediates/Results_kriging_Zone_", zone, ".rda"))
    load(paste0("../4_Analyse_Data_Other/intermediates/Results_STK_Zone_", zone, ".rda"))
    load(paste0("../4_Analyse_Data_Other/intermediates/Results_IDE_Zone_", zone, ".rda"))
    sgrid <- all_data$sgrid
    taxis <- taxis_df$idx
    maskidx <- which(sgrid$s1 > 0.1 & sgrid$s1 < 0.9 &  # we train on CNN only ..
                     sgrid$s2 > 0.1 & sgrid$s2 < 0.9) # inside this square box ..
    mask <- rep(1, 64^2)                                # to avoid boundary effects
    mask[-maskidx] <- 0
    maskidx_orig <- maskidx

    for(i in seq_along(taxis)) {
        if(i > tau) {

            ## intersection between interior and all unobserved locations
            maskidx <- intersect(maskidx_orig, 
                                 setdiff(1:4096, as(all_data$C[[i]], "dgTMatrix")@j+1))
            
            FilterResults <-  FilterResults %>%
                              rbind(summarystats(results[[i]]$truth[maskidx],
                                          results[[i]]$filter_mu[maskidx],
                                          results[[i]]$filter_sd[maskidx],
                                          name = "IDECNN",
                                          time = i,
                                          zone = zone)) %>%
                             rbind(summarystats(results[[i]]$truth[maskidx],
                                   results_IDE[[i]]$filter_mu_IDE[maskidx],
                                   pmax(1e-10,results_IDE[[i]]$filter_sd_IDE[maskidx]),
                                   name = "IDE",
                                   time = i,
                                   zone = zone)) %>%
                             rbind(summarystats(results[[i]]$truth[maskidx],
                                          results_kriging[[i]]$filter_mu_kriging[maskidx],
                                          pmax(1e-10,results_kriging[[i]]$filter_sd_kriging[maskidx]),
                                          name = "SKriging",
                                          time = i,
                                          zone = zone)) %>%
                             rbind(summarystats(results[[i]]$truth[maskidx],
                                          results_STK[[i]]$filter_mu_STK[maskidx],
                                          pmax(1e-10,results_STK[[i]]$filter_sd_STK[maskidx]),
                                          name = "STKriging",
                                          time = i,
                                          zone = zone)) 
            
            FcastResults <-  FcastResults %>%
                              rbind(summarystats(results[[i]]$truth[maskidx],
                                          results[[i]]$fcast_mu[maskidx],
                                          results[[i]]$fcast_sd[maskidx],
                                          name = "IDECNN",
                                          time = i,
                                          zone = zone)) %>%
                         rbind(summarystats(results[[i]]$truth[maskidx],
                                          results_IDE[[i]]$fcast_mu_IDE[maskidx],
                                          pmax(1e-10,results_IDE[[i]]$fcast_sd_IDE[maskidx]),
                                          name = "IDE",
                                          time = i,
                                          zone = zone)) %>%
                         rbind(summarystats(results[[i]]$truth[maskidx],
                                          results_kriging[[i-1]]$filter_mu_kriging[maskidx],
                                          pmax(1e-10,results_kriging[[i-1]]$filter_sd_kriging[maskidx]),
                                          name = "SKriging",
                                          time = i,
                                          zone = zone)) %>%
                             rbind(summarystats(results[[i]]$truth[maskidx],
                                          results_STK[[i]]$fcast_mu_STK[maskidx],
                                          pmax(1e-10,results_STK[[i]]$fcast_sd_STK[maskidx]),
                                          name = "STKriging",
                                          time = i,
                                          zone = zone))
            
        }
    }


    print(paste0("Zone ", zone))

    this_zone <- zone
    filter(FilterResults, zone == this_zone) %>%
        group_by(method) %>% summarise_all(mean, na.rm = TRUE)
    filter(FcastResults, zone == this_zone) %>%
        group_by(method) %>% summarise_all(mean, na.rm = TRUE)
    

    png(paste0("./img/ResultsTS_Zone", zone, ".png"), width = 1200, height = 400)
    g1 <- ggplot(filter(FilterResults, zone == this_zone)) +
          geom_line(aes(x = time, y = RMSPE, colour = method)) + theme_bw()
    g2 <- ggplot(filter(FcastResults, zone == this_zone)) +
        geom_line(aes(x = time, y = RMSPE, colour = method)) + theme_bw()
    print(grid.arrange(g1, g2, nrow = 1))
    dev.off()  
       
}

FilterResults_long <- gather(FilterResults, Diagnostic, Value, -method, -time, -zone)
FcastResults_long <- gather(FcastResults, Diagnostic, Value, -method, -time, -zone)

FilterResults_long$method  <- factor(FilterResults_long$method, levels = levels(FilterResults_long$method)[c(1,2,4,3)])
FcastResults_long$method  <- factor(FcastResults_long$method, levels = levels(FcastResults_long$method)[c(1,2,4,3)])

FilterResults_long <- group_by(FilterResults_long, time, zone, Diagnostic) %>%
    mutate(IDEdiag = Value[1]) %>%
    mutate(Value = Value/IDEdiag) %>%
    filter(!(method == "IDECNN"))
FcastResults_long <- group_by(FcastResults_long, time, zone, Diagnostic) %>%
    mutate(IDEdiag = Value[1]) %>%
    mutate(Value = Value/IDEdiag) %>%
    filter(!(method == "IDECNN"))

FilterResults_long2 <-
    FilterResults_long %>% group_by(method, zone, Diagnostic) %>%
     filter(Value > quantile(Value, 0.1, na.rm = TRUE) & Value < quantile(Value, 0.9, na.rm = TRUE))
FcastResults_long2 <-
    FcastResults_long %>% group_by(method, zone, Diagnostic) %>%
     filter(Value > quantile(Value, 0.1, na.rm = TRUE) & Value < quantile(Value, 0.9, na.rm = TRUE))

FilterResults_long2 <- mutate(FilterResults_long2,
                              Diagnostic2 =
                                  case_when(Diagnostic == "RMSPE" ~ "RMSPE ratio",
                                            Diagnostic == "CRPS" ~ "CRPS ratio",
                                            Diagnostic == "IS90" ~ "IS90 ratio",
                                            Diagnostic == "Cov90" ~ "Cov90 ratio"))
FcastResults_long2 <- mutate(FcastResults_long2,
                              Diagnostic2 =
                                  case_when(Diagnostic == "RMSPE" ~ "RMSPE ratio",
                                            Diagnostic == "CRPS" ~ "CRPS ratio",
                                            Diagnostic == "IS90" ~ "IS90 ratio",
                                            Diagnostic == "Cov90" ~ "Cov90 ratio"))


FilterResults_long2$Diagnostic2  <- factor(FilterResults_long2$Diagnostic2, 
                                           levels = c("RMSPE ratio", "CRPS ratio", 
                                                      "IS90 ratio", "Cov90 ratio"))
FcastResults_long2$Diagnostic2  <- factor(FcastResults_long2$Diagnostic2, 
                                          levels = c("RMSPE ratio", "CRPS ratio", 
                                                     "IS90 ratio", "Cov90 ratio"))


g1 <- ggplot(filter(FilterResults_long2, Diagnostic %in% c("RMSPE", "CRPS"))) +
    geom_boxplot(aes(y = Value, fill = method), position = 'dodge') + theme_bw(base_size = 22) + 
    facet_grid(Diagnostic2 ~ zone, scales = "free_y") +
    #geom_hline(data = data.frame(zone = c(0, 20), Value = 0.9, Diagnostic = "Cov90"), aes(yintercept = 0.9)) +
    scale_fill_brewer(palette = "Dark2") + theme(axis.text.x = element_blank()) +
    geom_hline(aes(yintercept = 1), linetype = "dashed")
ggsave(g1, file = "./img/FilterResults.png", width = 18, height = 6)

g2 <- ggplot(filter(FcastResults_long2, Diagnostic %in% c("RMSPE", "CRPS"))) +
      geom_boxplot(aes(y = Value, fill = method), position = 'dodge') + theme_bw(base_size=22) + 
      facet_grid(Diagnostic2 ~ zone, scales = "free_y") +
      scale_fill_brewer(palette = "Dark2") + theme(axis.text.x = element_blank()) +
      geom_hline(aes(yintercept = 1), linetype = "dashed")
ggsave(g2, file = "./img/ForecastResults.png", width = 18, height = 6)

## See how many times the IDECNN is better than STkriging
mean(filter(FilterResults,method == "IDECNN")$RMSPE < 
       filter(FilterResults,method == "STKriging")$RMSPE)
mean(filter(FcastResults,method == "IDECNN")$RMSPE < 
       filter(FcastResults,method == "STKriging")$RMSPE, na.rm = TRUE)
mean(filter(FilterResults,method == "IDECNN")$CRPS < 
       filter(FilterResults,method == "STKriging")$CRPS, na.rm = TRUE)
mean(filter(FcastResults,method == "IDECNN")$CRPS < 
       filter(FcastResults,method == "STKriging")$CRPS, na.rm = TRUE)
