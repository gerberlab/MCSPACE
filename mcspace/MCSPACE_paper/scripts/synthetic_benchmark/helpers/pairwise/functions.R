# load packages
library(data.table)
library(ggplot2)
# library(EcoSimR)

# the function that performs the randomization
ParallelSim9 <- function(BINARYMAT,NUMBERRANDOMCOMM,CORES){
  
  # initialize the parallel infrastructure
  library(doParallel)
  registerDoParallel(CORES)
  numberiterations <- NUMBERRANDOMCOMM
  
  # do the simulation in parallel, with X cores each performing an independent simulation of Y shuffles
  # the .combine=rbind means the resulting data will be rbinded together
  simulatedresults <- foreach(i = icount(numberiterations), .combine=rbind,.packages=c('reshape2',"EcoSimR","data.table")) %dopar% {
    # get the initial otu matrix
    testmat2 <- BINARYMAT
    
    # perform the sim9 shuffling X times
    for(j in 1:25000){
      testmat2 <- sim9_single(testmat2)
    }
    
    # melt it and only save instances that contain an actual count, label the iteration
    testres <- as.data.table(reshape2::melt(testmat2))[value > 0,]
    testres$Iteration <- i
    print(paste("Finished Iteration",i))
    return(testres)
  }
  names(simulatedresults) <- c("Particle","OTU","Count","Iteration")
  
  return(simulatedresults)
}

# function that compared the observed to the simulated value
# this should output the Zscore
DistributionDistanceComparison <- function(ORIGINALMATRIX,SIMRESULTS,DISTANCEMETRIC,DONOR,RANDOMIZATION){
  
  # calculate the otu associations in each simulatied community in parallel
  distanceiteration <- foreach(i = unique(SIMRESULTS$Iteration), .combine=rbind,.packages=c('reshape2',"parallelDist","data.table")) %dopar% {
    
    # cast data and put into appropriate form
    casted <- dcast.data.table(SIMRESULTS[Iteration == i],OTU~Particle,value.var = "Count")
    setnafill(casted,fill=0, cols=2:ncol(casted))
    castmat <- as.matrix(casted[,2:ncol(casted)])
    row.names(castmat) <- casted$OTU
    
    # perform the distance calculation and remove duplicate values
    distres <- as.matrix(parallelDist::parallelDist(castmat,DISTANCEMETRIC))
    
    distres[upper.tri(distres)] <- NA
    diag(distres) <- NA
    
    # make it a datatable and give it the iteration number
    distmelt <- as.data.table(reshape2::melt(distres,na.rm=T))
    distmelt$Iteration <- i
    
    return(distmelt)
  }
  
  # calculate distances for the original community
  distres <- as.matrix(parallelDist::parallelDist(t(ORIGINALMATRIX),DISTANCEMETRIC))
  
  # melt the original distance and merge it with the simulated communities
  distmelt <- as.data.table(reshape2::melt(distres,na.rm=T))
  names(distmelt)[3] <- "Original"
  combinedmatrix <- merge(distanceiteration,distmelt,by=c("Var1","Var2"))
  
  # # look at how the simulated value compares to the original
  # compplot <- ggplot(combinedmatrix,aes(Original,value,color=Iteration)) +
  #   geom_point(alpha=.2) +
  #   facet_wrap(~Iteration)
  # 
  # ggsave(paste(Sys.Date(),RANDOMIZATION,DISTANCEMETRIC,DONOR,"vs. simulated.png"),compplot,dpi=400,type="cairo-png",width=20,height=15)
  # ggsave(paste(Sys.Date(),RANDOMIZATION,DISTANCEMETRIC,DONOR,"vs. simulated.pdf"),compplot,width=20,height=15)
  
  
  # calculated means/sds for each otu pair, then calculate zscores and p values
  combinedsummary <- combinedmatrix[,.(meandist=mean(value),
                                       distsd=sd(value),
                                       median=median(value),
                                       Original=unique(Original)),
                                    by=.(Var1,Var2)]
  
  combinedsummary[,Zscore := ((meandist-Original)/(distsd + 1e-20))]
  combinedsummary[,zpvalue := pnorm(Zscore, lower.tail=FALSE)] 
  combinedsummary$zpadjusted <- p.adjust(combinedsummary$zpvalue,"fdr")

  # make sure there is no weirdness with the parallel stuff
  closeAllConnections() 
  
  return(combinedsummary)
}


# use this function if you just want the distance outputs from the comparison
DistributionDistanceComparisonOnlyDist <- function(ORIGINALMATRIX,SIMRESULTS,DISTANCEMETRIC,DONOR,RANDOMIZATION){
  
  # calculate the otu associations in each simulatied community in parallel
  distanceiteration <- foreach(i = unique(SIMRESULTS$Iteration), .combine=rbind,.packages=c('reshape2',"parallelDist","data.table")) %dopar% {
    
    # cast data and put into appropriate form
    casted <- dcast.data.table(SIMRESULTS[Iteration == i],OTU~Particle,value.var = "Count")
    setnafill(casted,fill=0, cols=2:ncol(casted))
    castmat <- as.matrix(casted[,2:ncol(casted)])
    row.names(castmat) <- casted$OTU
    
    # perform the distance calculation and remove duplicate values
    distres <- as.matrix(parallelDist::parallelDist(castmat,DISTANCEMETRIC))
    
    distres[upper.tri(distres)] <- NA
    diag(distres) <- NA
    
    # make it a datatable and give it the iteration number
    distmelt <- as.data.table(reshape2::melt(distres,na.rm=T))
    distmelt$Iteration <- i
    
    return(distmelt)
  }
  
  # calculate distances for the original community
  distres <- as.matrix(parallelDist::parallelDist(t(ORIGINALMATRIX),DISTANCEMETRIC))
  
  # melt the original distance and merge it with the simulated communities
  distmelt <- as.data.table(reshape2::melt(distres,na.rm=T))
  names(distmelt)[3] <- "Original"
  combinedmatrix <- merge(distanceiteration,distmelt,by=c("Var1","Var2"))
  
  # look at how the simulated value compares to the original
  compplot <- ggplot(combinedmatrix,aes(Original,value,color=Iteration)) +
    geom_point(alpha=.2) +
    facet_wrap(~Iteration)
  
  ggsave(paste(Sys.Date(),RANDOMIZATION,DISTANCEMETRIC,DONOR,"vs. simulated.png"),compplot,dpi=400,type="cairo-png",width=20,height=15)
  ggsave(paste(Sys.Date(),RANDOMIZATION,DISTANCEMETRIC,DONOR,"vs. simulated.pdf"),compplot,width=20,height=15)
  
  
  # calculated means/sds for each otu pair, then calculate zscores and p values
  combinedsummary <- combinedmatrix[,.(meandist=mean(value),
                                       distsd=sd(value),
                                       median=median(value),
                                       Original=unique(Original)),
                                    by=.(Var1,Var2)]
  
  combinedsummary[,Zscore := ((meandist-Original)/distsd)]
  combinedsummary[,zpvalue := 2*pnorm(abs(Zscore), lower.tail=FALSE)]
  
  combinedsummary$zpadjusted <- p.adjust(combinedsummary$zpvalue,"fdr")
  
  # get the p distributions
  pdist <- ggplot(combinedsummary,aes(zpvalue)) +
    geom_histogram() +
    scale_x_log10(limit=c(10^-6,1)) +
    geom_vline(xintercept = .05)
  
  ggsave(paste(Sys.Date(),RANDOMIZATION,DISTANCEMETRIC,DONOR,"otu unadjusted z p distribution.pdf"),pdist,width=6,height=5)
  ggsave(paste(Sys.Date(),RANDOMIZATION,DISTANCEMETRIC,DONOR,"otu unadjusted z p distribution.png"),pdist, dpi=1000,type="cairo-png",width=6,height=5)
  
  
  # get the p distribution
  adjustedpdist <- ggplot(combinedsummary,aes(zpadjusted)) +
    geom_histogram() +
    scale_x_log10(limit=c(10^-6,1)) +
    geom_vline(xintercept = .05)
  
  ggsave(paste(Sys.Date(),RANDOMIZATION,DISTANCEMETRIC,DONOR,"otu adjusted z p distribution.pdf"),adjustedpdist,width=6,height=5)
  ggsave(paste(Sys.Date(),RANDOMIZATION,DISTANCEMETRIC,DONOR,"otu adjusted z p distribution.png"),adjustedpdist,dpi=1000,type="cairo-png",width=6,height=5)
  
  # order the OTUs on the plot
  otuorder <- names(colSums(ORIGINALMATRIX))[order(colSums(ORIGINALMATRIX),decreasing = T)]
  combinedsummary$Var1 <- factor(combinedsummary$Var1,levels=otuorder)
  combinedsummary$Var2 <- factor(combinedsummary$Var2,levels=otuorder)
  
  combinedsummary2 <- combinedsummary
  names(combinedsummary2)[1:2] <- c("Var2","Var1") 
  
  combinedplot <- rbind(combinedsummary,combinedsummary2)
  
  
  # get the significant pairs
  combinedplot$Significant <- combinedplot$zpadjusted < .05
  
  # plot the jaccard overlap per otu pair
  distanceplot <- ggplot(combinedplot,aes(Var1,Var2,fill=Original)) +
    geom_tile() +
    scale_fill_gradient2(high="blue", low="darkred",midpoint = mean(combinedplot$Original)) +
    theme(axis.text.x = element_text(angle = 270)) +
    labs(fill = paste(DISTANCEMETRIC,"Distance"))  +
    xlab("OTU 1") +
    ylab("OTU 2") +
    geom_point(data=function(x){x[combinedplot$Significant == TRUE, ]},size=1,alpha=.7,color="black",shape=8) 
  
  ggsave(paste(Sys.Date(),RANDOMIZATION,DISTANCEMETRIC,DONOR,"pairwise plot.pdf"),distanceplot,width=30,height=30)
  ggsave(paste(Sys.Date(),RANDOMIZATION,DISTANCEMETRIC,DONOR,"pairwise plot.png"),distanceplot,dpi=1000,type="cairo-png",width=30,height=30)
  
  # plot and save the zscore plot
  print("Plotting Z Score Pairwise Plot")
  zplot <- ggplot(combinedplot,aes(Var1,Var2,fill=Zscore)) +
    geom_tile() +
    scale_fill_gradient2(low="blue", high="darkred",midpoint = 0) +
    theme(axis.text.x = element_text(angle = 270)) +
    labs(fill = "Z Score")  +
    xlab("OTU 1") +
    ylab("OTU 2") +
    geom_point(data=function(x){x[combinedplot$Significant == TRUE, ]},size=1,alpha=.7,color="black",shape=8) 
  
  ggsave(paste(Sys.Date(),RANDOMIZATION,DISTANCEMETRIC,DONOR,"pairwise otu zscore plot.pdf"),zplot,width=30,height=30)
  ggsave(paste(Sys.Date(),RANDOMIZATION,DISTANCEMETRIC,DONOR,"pairwise otu zscore plot.png"),zplot,dpi=1000,type="cairo-png",width=30,height=30)
  
  
  # make sure there is no weirdness with the parallel stuff
  closeAllConnections() 
  
  return(combinedmatrix)
}

