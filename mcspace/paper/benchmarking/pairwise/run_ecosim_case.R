# script that inputs pre-binarized data matrix and outputs matrix of co-association p-values
# auc calc done in python script in post

# set working directory
#C:\Users\Gary2\Dropbox (Partners HealthCare)\research_bwh\MCSPACE\MCSPACE\mcspace\paper\pairwise_methods
#setwd("C:/Users/gary/Dropbox (Partners HealthCare)/research_bwh/MC_SPACE2/MCSPACE/mcspace/B_benchmarking_lower_negbin/pairwise")

setwd("C:/Users/Gary2/Dropbox (Partners HealthCare)/research_bwh/MCSPACE/MCSPACE/mcspace/paper_cluster/pairwise")

# load packages
library(data.table)
library(ggplot2)

source("functions.R")

run_case <- function(outpath, datapath, datafile){
  print(paste("RUNNING datafile ", datafile, sep=""))
  data <- read.csv(file.path(datapath,datafile))
  testmat <- as.matrix(data)
  npart <- dim(testmat)[1]
  notus <- dim(testmat)[2]
  colnames(testmat) <- paste("ASV",1:notus, sep="")
  row.names(testmat) <- paste("Particle",1:npart, sep="")
  # perform the randomization
  print("performing ecosim randomization")
  simres <- ParallelSim9(testmat,50,12)
  # do the comparison to the overall distribution
  print("Comparing to the Calculated Distribution")
  distsum <- DistributionDistanceComparison(testmat,simres,"binary","Example","SIM9")
  outfile <- paste("results_",datafile,sep="")
  write.csv(distsum, file.path(outpath, outfile))
  print("DONE")
}


get_filename <-function(dset, num_clust, num_particles, nreads, gweight, nsubj, base_sample){
  datafile <- paste("bindata_D",dset, "_K", num_clust, "_P",num_particles,"_R",nreads,"_G",gweight,"_B",base_sample,"_S",nsubj,".csv",sep="")
  return(datafile)
}


get_cases <- function(bsample){
  # cases
  npart_cases <- list(10000, 5000, 1000, 500, 100)
  rdepth_cases <- list(10000, 5000, 1000, 500, 100)
  nclust_cases <- list(5, 10, 15, 20, 25)
  pgarb_cases <- list("0.0", 0.025, 0.05, 0.075, 0.1)
  n_dsets <- 9 # + 1 (start index from 0)
  
  dfiles <- list()

  for(dset in 0:n_dsets){
    # vs num clusters
    for(nclust in nclust_cases){ 
      df <- get_filename(dset, nclust, "default", "default", "default", "default", bsample)
      dfiles[[length(dfiles)+1]] = df
    }
    
    # vs num particles
    for(npart in npart_cases){
      df <- get_filename(dset, "default", npart, "default", "default", "default", bsample)
      dfiles[[length(dfiles)+1]] = df
    }
    
    # vs read depth
    for(rdepth in rdepth_cases){
      df <- get_filename(dset, "default", "default", rdepth, "default", "default", bsample)
      dfiles[[length(dfiles)+1]] = df
    }
    
    # vs contamination noise
    for(gweight in pgarb_cases){
      df <- get_filename(dset, "default", "default", "default", gweight, "default", bsample)
      dfiles[[length(dfiles)+1]] = df
    }
    
    # for mouse, vs subjects
    if (bsample == 'Mouse'){
      nsubj_cases <- list(1,3,5,7,10)
      for(nsubj in nsubj_cases){
        df <- get_filename(dset, "default", "default", "default", "default", nsubj, bsample)
        dfiles[[length(dfiles)+1]] = df
      }
    }
  }
  
  return(dfiles)
}

# ===========================================================================
# LOAD DATA AND RUN SIM
# ===========================================================================
start <- Sys.time()

base_sample <- "Mouse"

# data directory
datapath <- file.path("ecosim_data", base_sample)
outpath <- file.path("ecosimR_results", base_sample)
dir.create(outpath, recursive=TRUE, showWarnings = FALSE)

datafiles <- get_cases(base_sample)

# **get idx from command line (from 1 to number ...)
idx <- 1

df <- datafiles[[idx]]
run_case(outpath, datapath, df)


# output run-time
print("***ALL DONE***")
print("RUN TIME:")
print( Sys.time() - start )


