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

pca_train_90 <- predict(preProc_90, training[,predictionCols])
pca_test_90 <- predict(preProc_90, testing[,predictionCols])

system.time({
        modelFit_rf_pca90 <- train(x = pca_train_90, 
                                  y = training$classe,
                                  method="rf", 
                                  trControl=trainControl(method="cv",number=5, allowParallel = TRUE), 
                                  prox=TRUE)
})

print(confusionMatrix(training$classe, predict(modelFit_rf_pca90, newdata=pca_train_90)))

print(confusionMatrix(testing$classe, predict(modelFit_rf_pca90, newdata=pca_test_90)))

save(modelFit_rf_pca90, file = "modelFit_rf_pca90.RDA", compress="gzip")

