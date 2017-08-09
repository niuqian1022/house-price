library(caret)
library(dplyr)
library(glmnet)
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

#split dataset into factor (to generate dummys) and numeric variables (needs transform)
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
#log(x+1) numeric variables
train_noNZ<-train_noNZ[,-1]
test_noNZ<-test_noNZ[,-1]
train_noNZ<-mutate_all(train_noNZ, funs(log(.+1)))
test_noNZ<-mutate_all(test_noNZ, funs(log(.+1)))

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

fit1_crl <- trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 10,
    verboseIter=FALSE)  

glmGrid = expand.grid(alpha = seq(.05, 1, length = 15),
                      lambda = c((1:5)/10))
glmGrid = expand.grid(alpha = 0.05,
                      lambda = 0.1)
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
                       tuneGrid = glm_fit1$bestTune
)
coef<- data.frame(coef = row.names(coef(glm_fit1$finalModel, s=0.1)),
                  coef.value =coef(glm_fit1$finalModel, s=0.1)[,1])
coef <- coef[-1,]
coef<-coef[(coef$coef.value!=0),]


train_pred<-predict(glm_fit1, newdata = training)
RMSE(train_pred, training$SalePrice)
test_pred<-predict(glm_fit1, newdata = testing)
RMSE(test_pred, testing$SalePrice)

prediction<-predict(glm_fit1, newdata = test)
result<-cbind(Id = test$test.Id, SalePrice = exp(prediction)-1)
write.csv(result, "submit.csv", row.names = F)
View(result)