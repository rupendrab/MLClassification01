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
        modelFit_lda <- train(x = training[,predictionCols], 
                              y = training$classe,
                              method="lda")
})

print(confusionMatrix(training$classe, predict(modelFit_lda, newdata=training[,predictionCols])))

print(confusionMatrix(testing$classe, predict(modelFit_lda, newdata=testing[,predictionCols])))

save(modelFit_lda, file = "modelFit_lda.RDA", compress="gzip")

