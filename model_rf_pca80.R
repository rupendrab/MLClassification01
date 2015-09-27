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

pca_train_80 <- predict(preProc_80, training[,predictionCols])
pca_test_80 <- predict(preProc_80, testing[,predictionCols])

system.time({
        modelFit_rf_pca80 <- train(x = pca_train_80, 
                                  y = training$classe,
                                  method="rf", 
                                  trControl=trainControl(method="cv",number=5, allowParallel = TRUE), 
                                  prox=TRUE)
})

print(confusionMatrix(training$classe, predict(modelFit_rf_pca80, newdata=pca_train_80)))

print(confusionMatrix(testing$classe, predict(modelFit_rf_pca80, newdata=pca_test_80)))

save(modelFit_rf_pca80, file = "modelFit_rf_pca80.RDA", compress="gzip")
