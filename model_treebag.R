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
        modelFit_treebag <- bag(training[,predictionCols], 
                                training$classe,
                                B = 10,
                                bagControl = bagControl(
                                    fit = ctreeBag$fit,
                                    predict = ctreeBag$pred,
                                    aggregate = ctreeBag$aggregate
                                ))
})

print(confusionMatrix(training$classe, predict(modelFit_treebag, newdata=training[,predictionCols])))

print(confusionMatrix(testing$classe, predict(modelFit_treebag, newdata=testing[,predictionCols])))

save(modelFit_treebag, file = "modelFit_treebag.RDA", compress="gzip")

## Need to make sure all columns are the same class before applying prediction
finalData <- pml_test
for (i in 1:length(predictionCols)) {
    c1 <- class(training[,predictionCols[i]])
    c2 <- class(finalData[,predictionCols[i]])
    if (c1 != c2) {
        print(c(predictionCols[i], c1, c2))
        fn1 <- get(paste("as.", c1, sep=""), mode="function")
        finalData[,predictionCols[i]] <- fn1(finalData[,predictionCols[i]])
    }
}

predict(modelFit_treebag, newdata=finalData[,predictionCols])
load("modelFit_rf_full.RDA")
x1 <- predict(modelFit_treebag, newdata=finalData[,predictionCols])
x2 <- predict(modelFit_rf_full, newdata=finalData[,predictionCols])
any(x1 != x2)

cftest_treebag <- confusionMatrix(testing$classe, predict(modelFit_treebag, newdata=testing[,predictionCols]))
cftest_treebag$overall[1]

