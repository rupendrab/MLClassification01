setwd("/data1/RPred/")
load("pml_base.RDA")

.libPaths( c( .libPaths(), "/data1/R-repo/") )

library(caret)

set.seed(12123)
preProc_80 <- preProcess(training[,predictionCols], method="pca", thresh = 0.8)
preProc_80$numComp
preProc_90 <- preProcess(training[,predictionCols], method="pca", thresh = 0.9)
preProc_90$numComp
preProc_95 <- preProcess(training[,predictionCols], method="pca", thresh = 0.95)
preProc_95$numComp

save(preProc_80, preProc_90, preProc_95, file = "pca_data.RDA")
