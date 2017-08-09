library(caret)
library(parallel)
library(doParallel)
library(plyr)
library(dplyr)
library(randomForest)
library(RRF)
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

#split dataset into factor (to generate dummys) and numeric variables 
#(may needs transform)
class.index<-unlist(lapply(train, class))
train_factor<-train[,(class.index=="factor")]
train_numeric<-train[,(class.index!="factor")]
test_factor<-test[,(class.index=="factor")]
test_numeric<-test[,(class.index!="factor")]
#remove near_zero_variance variables
nzv<-nearZeroVar(train_numeric, freqCut = 95/5, uniqueCut = 10)
nzv_new<-integer(37)
nzv_new[nzv]<-1
train_noNZ<-train_numeric[,(nzv_new == 0)]
test_noNZ<-test_numeric[,(nzv_new == 0)]
#remove ID column
train_noNZ<-train_noNZ[,-1]
test_noNZ<-test_noNZ[,-1]

#generate dummy variables
dummys<-dummyVars(~., data = train_factor)
train_dummy<-predict(dummys, train_factor)
test_dummy<-predict(dummys, test_factor)
train<-data.frame(cbind(train_dummy, train_noNZ))
test<-data.frame(cbind(test$Id, test_dummy, test_noNZ))

#imput
imput<-preProcess(train[,-262], method = "medianImpute")
train<-predict(imput, train)
test<-predict(imput, test)


#change it to 90% 
inTrain <- createDataPartition(train$SalePrice, p = 0.9, list = FALSE)
training <- train[ inTrain,]
testing <- train[-inTrain,]

#allow parallel
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
#========================================================================================
#Random foretest model
fitControl <- trainControl(
    method = "CV",
    number = 5
    )

bestmtry <- tuneRF(training[, -262], training[, 262], 
                   stepFactor=2, improve=1e-6, ntree=500, plot = TRUE, metric = "RMSE")
rrfGrid<-expand.grid(mtry = 87, coefReg = 0.5, coefImp = 0.8)
rrfFit<-caret::train(SalePrice~., data = training, 
                  method = "RRF", 
                  trControl = fitControl,
                  tuneGrid = rrfGrid,
                  ntree = 500,
                  nodesize = 10,
                  metric = "RMSE"
                 )
rrfFit

rrf_pred<-predict(rrfFit, newdata = training)
RMSE(log(rrf_pred), log(training$SalePrice))
test_rrf_pred<-predict(rrfFit, newdata = testing)
RMSE(log(test_rrf_pred), log(testing$SalePrice))



#gbm models
#============================================================================
gbmGrid <-  expand.grid(interaction.depth = c(3, 5, 10, 15), 
                        n.trees = 500, 
                        shrinkage = 0.1,
                        n.minobsinnode = 10)
fitControl <- trainControl(
    method = "CV",
    number = 5
    )
gbmFit<-caret::train(SalePrice~., data = training, 
      method = "gbm", 
      trControl = fitControl,
      tuneGrid = gbmGrid,
      verbose = FALSE, 
      metric = "RMSE")
gbmFit<-caret::train(SalePrice~., data = training, 
                     method = "gbm", 
                     trControl = fitControl,
                     tuneGrid = gbmGrid,
                     verbose = FALSE, 
                     metric = "RMSE")
gbmFit
gbm_pred<-predict(gbmFit, newdata = training)
RMSE(log(gbm_pred), log(training$SalePrice))
test_gbm_pred<-predict(gbmFit, newdata = testing)
RMSE(log(test_gbm_pred), log(testing$SalePrice))

#==============================================
xgbGrid <- expand.grid(nrounds = 90,
                       max_depth = 6,
                       eta = 0.1,
                       gamma = 0,
                       colsample_bytree = 0.6,
                       min_child_weight = 1,
                       subsample = 1
                       #lambda = 7,
                       #alpha = 0.05
                       )
fitControl <- trainControl(
    method = "CV",
    number = 5
)
xgbFit<-caret::train(SalePrice~.,data = training,  
                     method = "xgbTree",
                     trControl = fitControl,
                     tuneGrid = xgbGrid,
                     metric = "RMSE"
                     )
#fit to all the data
xgbFit1<-caret::train(SalePrice~.,data = training,
                      method = "xgbTree",
                      trControl = trainControl(method = "none"),
                      tuneGrid = xgbGrid,
                      metric = "RMSE"
)
                     
xgbFit1
xgb_pred<-predict(xgbFit, newdata = training)
RMSE(log(xgb_pred), log(training$SalePrice))
test_xgb_pred<-predict(xgbFit, newdata = testing)
RMSE(log(test_xgb_pred), log(testing$SalePrice))
#=====================================================

level1Features.training<-data.frame(rrf = rrf_pred, 
                           gbm = gbm_pred, 
                           xgb = xgb_pred, 
                           training)
level1Features.testing<-data.frame(rrf = test_rrf_pred, 
                                   gbm = test_gbm_pred, 
                                   xgb = test_xgb_pred, 
                                   testing)
fitControl <- trainControl(
    method = "repeatedcv",
    number = 5,
    repeats = 1
)
combineFit<-caret::train(SalePrice~., data = level1Features.training,  
                     method = "nnt",
                     trControl = trainControl(method = "none"),
                     metric = "RMSE"
)
combineFit
comb_pred<-predict(combineFit, newdata = level1Features.training)
RMSE(log(comb_pred), log(level1Features.training$SalePrice))
test_comb_pred<-predict(combineFit, newdata = level1Features.testing)
RMSE(log(test_comb_pred), log(level1Features.testing$SalePrice))
#========================================================================================

#generate results and stop parallel processing 
test_rrf<-predict(rrfFit, newdata = test)
test_gbm<-predict(gbmFit, newdata = test)
test_xgb<-predict(xgbFit, newdata = test)
level1Features.test<-data.frame(rrf = test_rrf, 
                                   gbm = test_gbm, 
                                   xgb = test_xgb, 
                                   test[,2:263])
prediction<-predict(combineFit, newdata = level1Features.test)
result<-cbind(Id = test$test.Id, SalePrice = prediction)
write.csv(result, "submit.csv", row.names = F)
View(result)

stopCluster(cluster)
registerDoSEQ()