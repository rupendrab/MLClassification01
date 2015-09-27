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
                           number = 10,
                           repeats = 5,
                           ## Estimate class probabilities
                           classProbs = TRUE)

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

system.time({
        modelFit_gbm_simple <- train(x = training[,predictionCols], 
                                     y = training$classe,
                                     method = "gbm",
                                     verbose = FALSE,
                                     trControl = fitControl)
})

print(confusionMatrix(training$classe, predict(modelFit_gbm_simple, newdata=training[,predictionCols])))

print(confusionMatrix(testing$classe, predict(modelFit_gbm_simple, newdata=testing[,predictionCols])))

save(modelFit_gbm_simple, file = "modelFit_gbm_simple.RDA", compress="gzip")

