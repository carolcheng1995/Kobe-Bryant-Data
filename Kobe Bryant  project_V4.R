#final project

bryan<-read.csv(file="C:\\Users\\oweni\\Downloads\\bryant data set(1).csv")


#bryan <- read.csv(file="~/R/bryant data set.csv")
#final project
library(Metrics)
library(lattice)
library(caret)
library(ggplot2)
library(usdm)
library(glmnet)
library(caret)
library(mlbench)

apply(bryan, 2, function(x) sum(is.na(x)))


str(bryan)
#summary(training)

#deal with data type
bryan <- subset( bryan,  select = -c(3, 4, 12, 20, 21, 25 ))
bryan$playoffs<- as.factor(bryan$playoffs)
# back to integer 
bryan$shot_made_flag<-as.factor(bryan$shot_made_flag)
bryan$shot_made<-as.integer(bryan$shot_made_flag)
bryan$gameyear <- as.numeric(format(as.Date(bryan$game_date, format = "%m/%d/%Y"), "%Y"))
bryan$gamemonth <- as.factor(format(as.Date(bryan$game_date, format = "%m/%d/%Y"), "%m"))
bryan$gamedate <- as.numeric(format(as.Date(bryan$game_date, format = "%m/%d/%Y"), "%d"))
bryan <- subset( bryan,  select = -c(game_date))
bryan$shot_zone_area<-ifelse(grepl("Right", bryan$shot_zone_area), "R", ifelse(grepl("Left", bryan$shot_zone_area), "L", ifelse(grepl("Back", bryan$shot_zone_area), "B", "C")))
bryan$shot_zone_area<-as.factor(bryan$shot_zone_area)
bryan$matchup<-ifelse(grepl("@", bryan$matchup), 1, 0)
bryan$matchup<- as.factor(bryan$matchup)
bryan$minutes_remaining<-bryan$minutes_remaining*60
bryan$seconds_remaining<-bryan$minutes_remaining+bryan$seconds_remaining
bryan$minutes_remaining<-NULL
## change game year to Kobe's age
bryan$age = bryan$gameyear - 1978
bryan$age = as.numeric(bryan$age)

set.seed(1234)

data.split<- createDataPartition(bryan$shot_made, p=.7, list=F)
training <- bryan[data.split,]
testing = bryan[-data.split,]

str(training)
summary(training)

pairs(training[,c(3:8, 10, 11, 12, 19 )], lower.panel = panel.smooth)


#plot categorical variables
pairs(training[,c(3:7, 9, 10, 18, 20, 22)], lower.panel = panel.smooth)
#plot categorical variables
ggplot(data = training, aes(x = shot_distance, fill = shot_made_flag)) +
  geom_bar(position="fill", stat = 'count', alpha=.3)
ggplot(data = training, aes(x = combined_shot_type, fill=shot_made_flag)) +
  geom_bar(position="fill")
ggplot(data = training, aes(x = period, fill= shot_made_flag)) +
  geom_bar(position="fill")
ggplot(data = training, aes(x = loc_x, y = loc_y, col= shot_made_flag)) +
  facet_wrap(~ playoffs) +
  geom_point() +
  labs(title = "playoffs")
ggplot(data = training, aes(x = shot_type, fill= shot_made_flag)) +
  geom_bar(position="fill")
ggplot(data = training, aes(x = shot_zone_area, fill=shot_made_flag)) +
  geom_bar(position="fill")
ggplot(data = training, aes(x = loc_x, y = loc_y, col= shot_zone_area)) +
  facet_wrap(~ shot_made_flag) +
  geom_point(alpha = 2 / 5) +
  labs(title = "shot_zone_area")
ggplot(data = training, aes(x =matchup, fill= shot_made_flag )) +
  geom_bar(position="fill")
ggplot(data = training, aes(x = loc_x, y = loc_y, col= shot_made_flag)) +
  facet_wrap(~ matchup) +
  geom_point(alpha = 2 / 5) +
  labs(title = "Home or Away")
ggplot(data = training, aes(x =opponent, fill= shot_made_flag )) +
  geom_bar(position="fill")
ggplot(data = training, aes(x =gamemonth, fill= shot_made_flag )) +
  geom_bar(position="fill")
ggplot(data = training, aes(x = loc_x, y = loc_y, col= shot_made_flag)) +
  facet_wrap(~ gamemonth) +
  geom_point(alpha = 2 / 5) +
  labs(title = "Month")


trainmd<- subset( training,  select = -c(1, 3:6, 14, 15, 19, 21))
testmd<- subset( testing,  select = -c(1, 3:6, 14, 15, 19, 21))

#feature selection

trainmd.lm1<-glm(shot_made_flag ~ .-shot_made, data=trainmd, family = binomial())
summary(trainmd.lm1)

#plot to check assumptions
par(mfrow=c(2,2))
plot(trainmd.lm1)
par(mfrow=c(1,1))
vif(trainmd)

## multiple linear regression model result 

#predict(trainmd.lm1,type="response")
#mse(trainmd$shot_made,predict_lm1)
#predict(trainmd.lm1,type="response",newdata = trainmd)

predict_lm1<- round(predict(trainmd.lm1,type="response"), digits = 0)
mean(predict_lm1==trainmd$shot_made_flag)
xtabs(~ trainmd$shot_made_flag + predict_lm1)

#stepwise
trainmd.step.b<- step(trainmd.lm1, direction = "backward",
                      scope=formula(glm(shot_made_flag ~ . - shot_made , data=trainmd,family = binomial())))
summary(trainmd.step.b)
### ression using stepwise selected variables
trainmd_step = glm(shot_made_flag ~ combined_shot_type+period +
                     seconds_remaining+shot_distance+
                     shot_type, 
                   data=trainmd, family = binomial())

predict(trainmd_step,type="response")
predict_step<- round(predict(trainmd_step,type="response"), digits = 0)
mean(predict_step==trainmd$shot_made_flag)
xtabs(~ trainmd$shot_made_flag + predict_step)

# testing 

testmd_step=glm(shot_made_flag ~ combined_shot_type + period + seconds_remaining + shot_distance + shot_type,
                data=testmd,family = binomial())
predict(testmd_step,type='response')
predict_test_step<-round(predict(testmd_step,type='response'),digits=0)
mean(predict_test_step==testmd$shot_made_flag)
xtabs(~ testmd$shot_made_flag+predict_test_step)



#Lasso

training.matrix<- model.matrix(shot_made_flag ~ .- shot_made, data=trainmd)

#we can now imput the data into glmnet
#we use cv.glmnet for cross validation
#set alpha=1 for lasso regularization; alpha=0 for ridge regularization
training.lasso<- cv.glmnet(x = training.matrix, y = trainmd$shot_made, family = "binomial", alpha=1)

#we can plot the values of lambda to identify which values lead to smallest mse
plot(training.lasso)

#we can identify coefficients of the lambda that produces the smallest mse
coef(training.lasso, s="lambda.min")
#or the one that is the most parsimonious while being within the mse confidence interval
coef(training.lasso, s="lambda.1se")

#calculate training accuracy
predict(training.lasso, newx=training.matrix, type="response", s="lambda.1se")
predict_lasso = round(predict(training.lasso, newx=training.matrix, type="response", s="lambda.1se"), digit = 0)
mean(predict_lasso==trainmd$shot_made_flag)
xtabs(~ trainmd$shot_made_flag + predict_lasso)
#calculate testing accuracy
test.matrix<- model.matrix(shot_made_flag ~ .- shot_made, data=testmd)
predict_testlasso = round(predict(training.lasso, newx=test.matrix, type="response", s="lambda.1se"), digit = 0)
mean(predict_testlasso==testmd$shot_made_flag)
xtabs(~ testmd$shot_made_flag + predict_testlasso)

#evaluation

class(trainmd$shot_made_flag)

#stepwise

trainmd_step.glm.cv <- train(shot_made_flag ~ combined_shot_type+period + seconds_remaining+shot_distance+shot_type, data=trainmd, family = binomial(),trControl=trainControl(method="repeatedCV",repeats = 5, number = 5,savePredictions = T))
trainmd_step.glm.cv$resuls
trainmd_step.glm.cv

#lasso1

trainmd.lm2.glm.cv <- train(shot_made_flag ~ combined_shot_type+period +seconds_remaining+shot_distance+shot_type+matchup+opponent+gamemonth, data=trainmd, family = binomial(),trControl=trainControl(method="repeatedCV",repeats = 5, number = 5,savePredictions = T))
trainmd.lm2.glm.cv$results
trainmd.lm2.glm.cv

#lasso2

trainmd.lm3.glm.cv <- train(shot_made_flag ~ combined_shot_type+shot_distance, data=trainmd, family = binomial(),trControl=trainControl(method="repeatedCV",repeats = 5, number = 5,savePredictions = T))
trainmd.lm3.glm.cv$results
trainmd.lm3.glm.cv

sum(trainmd$shot_made == 0)
summary(trainmd)

### Classification

# Run machine learning algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# a) linear algorithms
set.seed(7)
fit.lda <- train(shot_made_flag ~ .- shot_made, data=trainmd, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(shot_made_flag ~ .- shot_made, data=trainmd, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(shot_made_flag ~ .- shot_made, data=trainmd,method="knn", metric=metric, trControl=control)

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn))
# , svm=fit.svm, rf=fit.rf
summary(results)

# compare accuracy of models
dotplot(results)


# summarize Best Model
print(fit.lda)

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, testmd)
confusionMatrix(predictions, testmd$shot_made_flag)
Accuracy(predictions, testmd$shot_made_flag)

# rf
library('randomForest')
library(MLmetrics)
library(ModelMetrics)

rf1 <- randomForest(shot_made_flag ~ .- shot_made, data=trainmd,
                    ntree=4000, 
                    importance=TRUE,keep.forest=TRUE, 
                    sampsize=c(2000,8000),
                    strata=trainmd$shot_made_flag)
rf1

# Imp
par(mfrow = c(1,1))
varImpPlot(rf1, n.var=10)

pred_rf_train = predict(rf1,newdata = trainmd)
xtabs(~ trainmd$shot_made_flag + pred_rf_train)

pred_rf_test = predict(rf1,newdata = testmd)
xtabs(~ testmd$shot_made_flag + pred_rf_test)

#compare the accuracy between testing and training
Accuracy(trainmd$shot_made_flag,pred_rf_train)
Accuracy(testmd$shot_made_flag,pred_rf_test)


