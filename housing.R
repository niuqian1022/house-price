library(caret)
library(parallel)
library(doParallel)
library(dplyr)
train<-read.csv("train.csv", header = TRUE)
test<-read.csv("test.csv", header = TRUE)
train<-tbl_df(train)
test<-tbl_df(test)
str(train)
dim(train)
#colnames(train)
#remove variables with many NAs
na_counts <- apply(train, 2, function(x) sum(is.na(x)))
col_rm<-colnames(train[,na_counts>100])
train<-train%>%mutate(LotFrontage = NULL, Alley = NULL,FireplaceQu = NULL,
               PoolQC = NULL, Fence = NULL, MiscFeature = NULL)
test<-test%>%mutate(LotFrontage = NULL, Alley = NULL,FireplaceQu = NULL,
                    PoolQC = NULL, Fence = NULL, MiscFeature = NULL)    
test<-test%>%mutate(SalePrice = 0)


#transform factors to integers, and Near-Zero variables of a cut point (99/1, 0.05) 
#removed
class.index<-unlist(lapply(train, class))
train_factor<-train[,(class.index=="factor")]
train_numeric<-train[,(class.index!="factor")]
test_factor<-test[,(class.index=="factor")]
test_numeric<-test[,(class.index!="factor")]
nzv<-nearZeroVar(train_numeric, freqCut = 99/1, uniqueCut = 5)
nzv_new<-integer(37)
nzv_new[nzv]<-1
train_noNZ<-train_numeric[,(nzv_new == 0)]
test_noNZ<-test_numeric[,(nzv_new == 0)]

train_factor<-mutate_all(train_factor, funs(as.integer))
test_factor<-mutate_all(test_factor, funs(as.integer))
train<-data.frame(cbind(train_factor, train_noNZ))
test<-data.frame(cbind(test_factor, test_noNZ))
imput<-preProcess(train[,-71], method = "medianImpute")
train<-predict(imput, train)
test<-predict(imput, test)

set.seed(0)
inTrain <- createDataPartition(train$SalePrice, p = 0.8, list = FALSE) #change it to 80% 
training <- train[ inTrain,]
testing <- train[-inTrain,]


#allow parallel
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
#========================================================================================
#random foretest model, with selected mtry = 23 and ntree = 500;
fitControl <- trainControl(
    ## 10-fold CV
    method = "repeatedcv",
    number = 10,
    ## repeated 5 times
    repeats = 5)
rfGrid<-expand.grid(mtry = 70)
bestmtry <- tuneRF(training[, -71], training[, 71], 
                   stepFactor=2, improve=1e-5, ntree=500, plot = TRUE)
fit<-caret::train(SalePrice~., data = training, 
                  method = "rf", 
                  trControl = fitControl,
                  tuneGrid = rfGrid,
                  ntree = 500
                  )
#========================================================================================
#log(x+1) all the predictors
library(glmnet)
train_log<-mutate_all(train, funs(log(.+1)))
inTrain <- createDataPartition(train_log$SalePrice, p = 0.9, list = FALSE) 
#change it to 90% 
training <- train_log[ inTrain,]
testing <- train_log[-inTrain,]

fit1_crl <- trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 10,
    verboseIter=FALSE)  

glmGrid = expand.grid(alpha = seq(.05, 1, length = 15),
                       lambda = c((1:5)/10))

glm_fit1<-caret::train(SalePrice~., data = training, 
                  method = "glmnet", 
                  trControl = fit1_crl,
                  preProc = c("center", "scale"),
                  tuneGrid = glmGrid
)
glm_fit1<-caret::train(SalePrice~., data = training, 
                       method = "glmnet", 
                       trControl = fit1_crl,
                       preProc = c("center", "scale"),
                       tuneGrid = expand.grid(alpha = 0.05, lambda = 0.5)
)
test_log<-mutate_all(test, funs(log(.+1)))
prediction<-predict(glm_fit1, newdata = test_log)
result<-cbind(Id = test$Id, SalePrice = prediction)
write.csv(result, "submit.csv", row.names = F)
View(result)
#stop parallel processing 
stopCluster(cluster)
registerDoSEQ()

train_pred<-predict(glm_fit1, newdata = training)
RMSE(train_pred, training$SalePrice)
test_pred<-predict(glm_fit1, newdata = testing)
RMSE(test_pred, testing$SalePrice)

prediction<-predict(fit, newdata = test)
result<-cbind(Id = test$Id, SalePrice = exp(prediction)-1)
write.csv(result, "submit.csv", row.names = F)
View(result)

