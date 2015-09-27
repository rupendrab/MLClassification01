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

pca_train_95 <- predict(preProc_95, training[,predictionCols])
pca_test_95 <- predict(preProc_95, testing[,predictionCols])

system.time({
        modelFit_rf_pca95 <- train(x = pca_train_95, 
                                  y = training$classe,
                                  method="rf", 
                                  trControl=trainControl(method="cv",number=5, allowParallel = TRUE), 
                                  prox=TRUE)
})

print(confusionMatrix(training$classe, predict(modelFit_rf_pca95, newdata=pca_train_95)))

print(confusionMatrix(testing$classe, predict(modelFit_rf_pca95, newdata=pca_test_95)))

save(modelFit_rf_pca95, file = "modelFit_rf_pca95.RDA", compress="gzip")

