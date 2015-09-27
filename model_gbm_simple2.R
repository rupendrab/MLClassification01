setwd("/data1/RPred/")
repoDir = "/data1/R-repo"
if ( ! any(.libPaths() == repoDir) ) {
    .libPaths( c( .libPaths(), "/data1/R-repo/") )
}

library(caret)
library(doMC)
registerDoMC(cores = 8)

load("pml_base.RDA")
load("pca_data.RDA")

fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 1,
                           ## Estimate class probabilities
                           classProbs = TRUE)

system.time({
        modelFit_gbm_simple2 <- train(x = training[,predictionCols], 
                                      y = training$classe,
                                      method = "gbm",
                                      verbose = FALSE,
                                      trControl = fitControl)
})

print(confusionMatrix(training$classe, predict(modelFit_gbm_simple2, newdata=training[,predictionCols])))

print(confusionMatrix(testing$classe, predict(modelFit_gbm_simple2, newdata=testing[,predictionCols])))

save(modelFit_gbm_simple2, file = "modelFit_gbm_simple2.RDA", compress="gzip")

