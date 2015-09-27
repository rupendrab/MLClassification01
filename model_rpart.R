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
        modelFit_rpart <- train(x = training[,predictionCols], 
                              y = training$classe,
                              method="rpart")
})

print(confusionMatrix(training$classe, predict(modelFit_rpart, newdata=training[,predictionCols])))

print(confusionMatrix(testing$classe, predict(modelFit_rpart, newdata=testing[,predictionCols])))

save(modelFit_rpart, file = "modelFit_rpart.RDA", compress="gzip")

