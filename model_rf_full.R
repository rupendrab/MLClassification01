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

system.time({
        modelFit_rf_full <- train(x = training[,predictionCols], 
                                  y = training$classe,
                                  method="rf", 
                                  trControl=trainControl(method="cv",number=5, allowParallel = TRUE), 
                                  prox=TRUE)
})

print(confusionMatrix(training$classe, predict(modelFit_rf_full, newdata=training[,predictionCols])))

print(confusionMatrix(testing$classe, predict(modelFit_rf_full, newdata=testing[,predictionCols])))

save(modelFit_rf_full, file = "modelFit_rf_full.rda", compress="gzip")

