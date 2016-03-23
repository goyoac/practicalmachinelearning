# Nedded libraries

install.packages("ggplot2", repos = "https://mran.revolutionanalytics.com/snapshot/2016-03-19")
install.packages("caret", repos = "https://mran.revolutionanalytics.com/snapshot/2016-03-19")
install.packages("randomForest", repos = "https://mran.revolutionanalytics.com/snapshot/2016-03-19")
install.packages("e1071", repos = "https://mran.revolutionanalytics.com/snapshot/2016-03-19")

library(ggplot2)
library(caret)
library(randomForest)


# Loading data

pmlTrainingFile <- "pml-training.csv"
pmlTestFile     <- "pml-testing.csv"

pmlTrainingUrl  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
pmlTestUrl      <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if (!file.exists(pmlTrainingFile)) {
  download.file(pmlTrainingUrl, pmlTrainingFile, method="curl")
}

if (!file.exists(pmlTestFile)) {
  download.file(pmlTestUrl, pmlTestFile, method="curl")
}

pmlTraining <- read.csv(pmlTrainingFile, na.strings = c("", "NA", "#DIV/0!"))
pmlTesting  <- read.csv(pmlTestFile,     na.strings = c("", "NA", "#DIV/0!"))

# Preparing data

## Removing variables with NAs or empty
#pmlTraining.cleaned <- pmlTraining[, -which(sapply(pmlTraining, function(x) any(is.na(x)) || any(x=="")))]
pmlTraining.cleaned <- pmlTraining [, colSums(is.na(pmlTraining)) == 0]
pmlTesting.cleaned  <- pmlTesting  [, colSums(is.na(pmlTesting))  == 0]


## Removing unnecessary variables
colsNotNeeded <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")
pmlTraining.cleaned <- pmlTraining.cleaned [, -which(names(pmlTraining.cleaned) %in% colsNotNeeded)]
pmlTesting.cleaned  <- pmlTesting.cleaned  [, -which(names(pmlTesting.cleaned)  %in% colsNotNeeded)]

## Spliting training data
set.seed(1111)
toTrain <- createDataPartition(pmlTraining.cleaned$classe, p=0.7, list=FALSE)
pmlTraining.cleaned.trainSet      <- pmlTraining.cleaned[ toTrain, ]
pmlTraining.cleaned.validationSet <- pmlTraining.cleaned[-toTrain, ]

# Modeling

pmlModelRandomForest <- train(classe ~ .,
                              data          = pmlTraining.cleaned.trainSet,
                              method        = "rf",
                              trControl     = trainControl(method="cv", number=5),
                              prox          = TRUE,
                              ntree         = 250
                             )

pmlModelRandomForest <- train(classe ~ .,
                              data          = pmlTraining.cleaned.trainSet,
                              method        = "rf",
                              trControl     = trainControl(method="cv", number=5)
)

validationPredictRandomForest <- predict(pmlModelRandomForest, pmlTraining.cleaned.validationSet)
confMatrixRndomForest <- confusionMatrix(pmlTraining.cleaned.validationSet$classe, validationPredictRandomForest)

testPredictRandomForest <- predict(pmlModelRandomForest, pmlTesting.cleaned)