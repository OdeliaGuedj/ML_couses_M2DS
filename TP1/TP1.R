library(mvtnorm)
library(MASS)
library(class)
library(rpart)

##Données
N = 100
mean_blue = as.matrix(cbind(2,2))
mean_red = as.matrix(cbind(0,0))
sigma_blue = matrix(  c(1,0,0,1), nrow=2, ncol=2,byrow = TRUE)
sigma_red = matrix(  c(1,0.75,0.75,1), nrow=2, ncol=2,byrow = TRUE)
x_blue <- rmvnorm(n = N, mean=mean_blue, sigma=sigma_blue)
x_red <- rmvnorm(n = N, mean=mean_red, sigma=sigma_red)
 

##Graphe
plot(x_blue, col = "blue", ylim = c(min(x_red[,2],x_blue[,2]),max(x_red[,2],x_blue[,2])),
     xlim = c(min(x_red[,1],x_blue[,1]),max(x_red[,1],x_blue[,1])),
     xlab= "",ylab  ="")
points(x_red, col = "red")

##Naive Bayes Classifier
mu_hat_blue = c(mean(x_blue[,1]), mean(x_blue[,2]))
mu_hat_red = c(mean(x_red[,1]), mean(x_red[,2]))
sigma_hat_blue = matrix(c(sd(x_blue[,1]), cov(x_blue[,1], x_blue[,2]), cov(x_blue[,2], x_blue[,1]), sd(x_blue[,2])),
                        nrow = 2, ncol = 2, byrow = T)
sigma_hat_red = matrix(c(sd(x_red[,1]), cov(x_red[,1], x_red[,2]), cov(x_red[,2], x_red[,1]), sd(x_red[,2])),
                        nrow = 2, ncol = 2, byrow = T)

y_new = function(x_new){
ifelse((dmvnorm(x_new, mu_hat_blue, sigma_hat_blue) > dmvnorm(x_new, mu_hat_red, sigma_hat_red)),1,2)
}

##Points of the grid generation
K=40
seqx1 = seq(min(x_red[,1],x_blue[,1]),max(x_red[,1],x_blue[,1]), length = K)
seqx2 = seq(min(x_red[,2],x_blue[,2]),max(x_red[,2],x_blue[,2]), length = K)
mygrid = expand.grid(z1 = seqx1, z2 = seqx2)
names(mygrid) = c("X1","X2")
y_pred_grid = y_new(mygrid)

#prediction viz
red2 = rgb(red = 254/255, green = 231/255, blue = 240/255, alpha = .8)
blue2 = rgb(red = 51/255, green = 161/255, blue = 201/255, alpha = .2)
plot.new()
image(seqx1, seqx2, matrix(y_pred_grid,K), col=c(blue2, red2), xlab="", ylab="", xaxt="n", yaxt="n")
points(x_blue[,1], x_blue[,2], col="blue", pch=16, lwd=2, cex=0.8)
points(x_red[,1], x_red[,2], col="red", pch=16, lwd=2, cex=0.8)
contour(seqx1, seqx2, matrix(y_pred_grid,K), col="black", lty=1, lwd=1, add=TRUE, drawlabels=FALSE)

## Linear Discriminant Analysis
z = as.data.frame(rbind(cbind(x_blue,1), cbind(x_red,2)))
colnames(z) = c("X1","X2","Y")
model.lda = lda(z$Y~., data = z)
y_pred_lda = predict(model.lda, newdata = mygrid) 
y_pred_grid_lda = as.numeric(y_pred_lda$class)
plot.new()
image(seqx1, seqx2, matrix(y_pred_grid_lda,K), col=c(blue2, red2), xlab="", ylab="", xaxt="n", yaxt="n")
points(x_blue[,1], x_blue[,2], col="blue", pch=16, lwd=2, cex=0.8)
points(x_red[,1], x_red[,2], col="red", pch=16, lwd=2, cex=0.8)
contour(seqx1, seqx2, matrix(y_pred_grid_lda,K), col="black", lty=1, lwd=1, add=TRUE, drawlabels=FALSE)

## Quadratic Discriminant Analysis
model.qda = qda(z$Y~., data = z)
y_pred_qda = predict(model.qda, newdata = mygrid) 
y_pred_grid_qda = as.numeric(y_pred_qda$class)
plot.new()
image(seqx1, seqx2, matrix(y_pred_grid_qda,K), col=c(blue2, red2), xlab="", ylab="", xaxt="n", yaxt="n")
points(x_blue[,1], x_blue[,2], col="blue", pch=16, lwd=2, cex=0.8)
points(x_red[,1], x_red[,2], col="red", pch=16, lwd=2, cex=0.8)
contour(seqx1, seqx2, matrix(y_pred_grid_qda,K), col="black", lty=1, lwd=1, add=TRUE, drawlabels=FALSE)

## Logistic regression
z0 = z
z0$Y = as.numeric(z0$Y)-1
model.glm = glm(z0$Y~., data= z0, family = "binomial")
y_pred_glm = predict(model.glm, newdata = mygrid)
y_pred_grid_glm = 1*(y_pred_glm>0)+1
plot.new()
image(seqx1, seqx2, matrix(y_pred_grid_glm,K), col=c(blue2, red2), xlab="", ylab="", xaxt="n", yaxt="n")
points(x_blue[,1], x_blue[,2], col="blue", pch=16, lwd=2, cex=0.8)
points(x_red[,1], x_red[,2], col="red", pch=16, lwd=2, cex=0.8)
contour(seqx1, seqx2, matrix(y_pred_grid_glm,K), col="black", lty=1, lwd=1, add=TRUE, drawlabels=FALSE)

## K Nearest Neighbors
# k = 3 
y_pred_grid_knn3 = as.numeric(knn(train = z[,1:2], test = mygrid,cl = z[,3] ,k=3))

plot.new()
image(seqx1, seqx2, matrix(y_pred_grid_knn3,K), col=c(blue2, red2), xlab="", ylab="", xaxt="n", yaxt="n")
points(x_blue[,1], x_blue[,2], col="blue", pch=16, lwd=2, cex=0.8)
points(x_red[,1], x_red[,2], col="red", pch=16, lwd=2, cex=0.8)
contour(seqx1, seqx2, matrix(y_pred_grid_knn3,K), col="black", lty=1, lwd=1, add=TRUE, drawlabels=FALSE)

# k = 41 
y_pred_grid_knn41 = as.numeric(knn(train = z[,1:2], test = mygrid,cl = z[,3] ,k=41))

plot.new()
image(seqx1, seqx2, matrix(y_pred_grid_knn41, K), col=c(blue2, red2), xlab="", ylab="", xaxt="n", yaxt="n")
points(x_blue[,1], x_blue[,2], col="blue", pch=16, lwd=2, cex=0.8)
points(x_red[,1], x_red[,2], col="red", pch=16, lwd=2, cex=0.8)
contour(seqx1, seqx2, matrix(y_pred_grid_knn41, K), col="black", lty=1, lwd=1, add=TRUE, drawlabels=FALSE)

# Evaluation of the predictive power of the classifier
kf = 5
length_kf = kf/N
for (i in 1:kf){
  test = z[(length_kf*(i-1)+1):length_kf*i,]
  train = z[-c((length_kf*(i-1)+1):length_kf*i),]
  model = lda(train[,3]~., data = train[1:2])
  y_pred = predict(model, newdata = test[1,2])
  y_pred_grid = as.numeric(y_pred_lda$class)
  
  
  }



















