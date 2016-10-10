library(ggplot2)

sumwv.accdf<-read.csv('sumword2vec_accuracies.csv', stringsAsFactors = FALSE)
auto.accdf<-read.csv('autoencoder_accuracies.csv', stringsAsFactors = FALSE)
cnn.accdf<-read.csv('cnn_accuracies.csv', stringsAsFactors = FALSE)
concat.accdf<-rbind(data.frame(algorithm='SumWV', accuracy=sumwv.accdf$accuracy),
                    data.frame(algorithm='Auto', accuracy=auto.accdf$accuracy),
                    data.frame(algorithm='CNN', accuracy=cnn.accdf$accuracy))

g<-ggplot(data=concat.accdf)+geom_boxplot(aes(factor(algorithm), accuracy))+ggtitle('Performance of Algorithms (Cross Validation)')
