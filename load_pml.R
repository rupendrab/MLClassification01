library(caret)

pml_train <- read.csv("./data/pml-training.csv", header=TRUE)
names(pml_train)
table(pml_train$classe)
pml_test <- read.csv("data/pml-testing.csv", header=TRUE)

findNA <- function(inp) {
        for (i in 1:dim(inp)[2]) {
                colname <- colnames(inp)[i]
                colclass <- class(inp[,i])
                pctNA <- mean(is.na(inp[,i]) * 1)
                if (colclass == "factor") {
                        # print(colname)
                        pctNA <- mean(grepl("^\\s*$", as.character(inp[,i])) * 1)
                }
                df = data.frame(colname = colname, colclass = colclass, pctNA = pctNA)
                if (i == 1) {
                        dfret = df
                }
                else {
                        dfret <- do.call(rbind, list(df, dfret))
                }
        }
        dfret
}

set.seed(12121)
inTrain <- createDataPartition(y=pml_train$classe, p=0.7, list=FALSE)
training <- pml_train[inTrain,]
testing <- pml_train[-inTrain,]
dim(training); dim(testing)

naSummary <- findNA(training)
dim(naSummary[naSummary$pctNA <= 0.9,])

predictionCols <- naSummary[naSummary$pctNA <= 0.9 & ! naSummary$colname %in% c("classe","X"),]$colname
predictionCols <- as.character(predictionCols)
predictionCols

## Find variables with little variance - Near Zero
nsv <- nearZeroVar(training[,as.character(predictionCols)], saveMetrics = TRUE)
nsv[nsv$nzv,]
predictionCols <- predictionCols[! predictionCols %in% as.character(row.names(nsv[nsv$nzv,]))]

predictionCols <- predictionCols[! predictionCols %in% c("cvtd_timestamp", 
                                                         "raw_timestamp_part_1",
                                                         "raw_timestamp_part_2",
                                                         "user_name")]

save(pml_train, pml_test, training, testing, inTrain, predictionCols, file = "pml_base.RDA")

