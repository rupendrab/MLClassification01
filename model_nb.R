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
        modelFit_nb <- train(x = training[,predictionCols], 
                              y = training$classe,
                              method="nb")
})

print(confusionMatrix(training$classe, predict(modelFit_nb, newdata=training[,predictionCols])))

print(confusionMatrix(testing$classe, predict(modelFit_nb, newdata=testing[,predictionCols])))

save(modelFit_nb, file = "modelFit_nb.RDA", compress="gzip")

