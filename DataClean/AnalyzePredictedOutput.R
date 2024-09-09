library("tidyverse")
library("ggplot2")
data <- read.csv("CNN_RGB_50.csv") %>%
  mutate(Correct=as.logical(Correct)) %>%
  arrange(Targets)
test <- read.csv("TestingCorrect_combinedULcases.csv") %>% 
  arrange(Class.Label)
data <- data %>%
  mutate(Type = str_sub(as.character(test$File.Name),1,-13))

num <- data %>% group_by(Targets) %>% tally()
#remove class labels for lowercase c o s u v y z
Letter <- rep(c(LETTERS,letters)[-c(29,41,45,47,48,51,52)], num$n)
data <- data  %>% arrange(Targets) %>% mutate(ClassLetter=Letter)

num <- data %>% group_by(Predictions) %>% tally()
#remove class labels for lowercase c o s u v y z
Letter <- rep(c(LETTERS,letters)[-c(29,41,45,47,48,51,52)], num$n)
data <- data %>% arrange(Predictions) %>% mutate(PredictLetter=Letter)

# Accuracy by image type
type <- data %>%
  group_by(Type) %>%
  summarize(Total=n(), Accuracy=sum(Correct)/Total)

# Accuracy by class
class <- data %>%
  group_by(Targets) %>%
  summarize(Total=n(), Accuracy=sum(Correct)/Total, Letter=first(ClassLetter))
