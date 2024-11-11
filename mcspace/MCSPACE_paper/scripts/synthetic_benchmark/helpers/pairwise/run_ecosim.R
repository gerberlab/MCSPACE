# script that inputs pre-binarized data matrix and outputs matrix of co-association p-values
# auc calc done in python script in post

# set working directory
# C:\Users\Gary2\Dropbox (Partners HealthCare)\research_bwh\MCSPACE_FINAL\MCSPACE\mcspace\MCSPACE_paper\scripts\synthetic_benchmark\helpers\pairwise
setwd("C:/Users/Gary2/Dropbox (Partners HealthCare)/research_bwh/MCSPACE_FINAL/MCSPACE/mcspace/MCSPACE_paper/scripts/synthetic_benchmark/helpers/pairwise")

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


get_filename <-function(dset, num_particles, nreads, base_sample){
  datafile <- paste("bindata_D",dset, "_P",num_particles,"_R",nreads,"_B",base_sample,".csv",sep="")
  return(datafile)
}


get_cases <- function(bsample){
  # cases
  npart_cases <- list(5000, 2500, 1000, 500, 250)
  rdepth_cases <- list(5000, 2500, 1000, 500, 250)
  n_dsets <- 9 # + 1 (start index from 0)
  
  dfiles <- list()

  for(dset in 0:n_dsets){
    # vs num particles
    for(npart in npart_cases){
      df <- get_filename(dset, npart, "default", bsample)
      dfiles[[length(dfiles)+1]] = df
    }
    
    # vs read depth
    for(rdepth in rdepth_cases){
      df <- get_filename(dset, "default", rdepth, bsample)
      dfiles[[length(dfiles)+1]] = df
    }
  }
  return(dfiles)
}


# ===========================================================================
# LOAD DATA AND RUN SIM
# ===========================================================================
start <- Sys.time()

base_sample <- "Human"

# data directory
datapath <- file.path("..", "..","..","..","output","assemblage_recovery", "ecosim_data", base_sample)
outpath <- file.path("..", "..","..","..","output","assemblage_recovery","ecosimR_results", base_sample)
dir.create(outpath, recursive=TRUE, showWarnings = FALSE)

datafiles <- get_cases(base_sample)

# **get idx from command line (from 1 to number ...)
for(idx in 1:100){
  df <- datafiles[[idx]]
  run_case(outpath, datapath, df)
}


# output run-time
print("***ALL DONE***")
print("RUN TIME:")
print( Sys.time() - start )


