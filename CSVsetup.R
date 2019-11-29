library(tidyverse)
# ClassLabel, FileName, Category
GImg <- rep("GImg", 7112)
Fnt <- rep("Fnt",52832)
Hnd <- rep("Hnd",2860)

################################################################################
# GImg
num2 <- c(558,115,215,191,446,79,143,193,302,77,92,215,149,363,382,159,35,389,342,312,
          92,84,67,80,67,55,158,38,63,46,227,37,36,52,127,33,34,58,43,135,
          148,37,54,126,124,111,41,33,36,35,43,35)

int <- formatC(1:52, width=2, flag='0')

string <- rep(int, num2)

n <- c()
index <- 1
for (i in 1:length(num2)) {
  size = num2[i]
  n <- c(n,1:size)
  index = index + size
}
n <- formatC(n, width=4, flag='0') 

name <- paste(GImg,string,n,sep="_")
name <- paste(name, ".png",sep="")

df <- data.frame(ClassLabel=as.numeric(string)-1, FileName=name, Group=GImg)

################################################################################
# Fnt
num <- rep(1016, 52)
string <- rep(int, num)

n <- rep(1:1016,52)
n <- formatC(n, width=4, flag='0')

name <- paste(Fnt,string,n,sep="_")
name <- paste(name, ".png",sep="")

df <- rbind(df, data.frame(ClassLabel=as.numeric(string)-1, FileName=name, Group=Fnt))

################################################################################
# Hnd
num <- rep(55, 52)
string <- rep(int, num)

n <- rep(1:55,52)
n <- formatC(n, width=4, flag='0')

name <- paste(Hnd,string,n,sep="_")
name <- paste(name, ".png",sep="")

df <- rbind(df, data.frame(ClassLabel=as.numeric(string)-1, FileName=name, Group=Hnd))

################################################################################
# Wrong class labels for certain fonts
# "Fnt_01_0385:0388" -> a
# a,b,d,e,f,j.h,i,j,l,m,n
#  0009:0012, 0189:0192, 0237:0240, 0301:0304, 0337:0340,
#  0433:0436, 0813:0820, 0897:0900, 0933:0936"

# Dataframe before correction as df2
df2 <- df

# Lower to Upper
num <- c(9:12,189:192,237:240,301:304,337:340,433:436,813:820,897:900,933:936)
num <- formatC(num, width=4, flag='0')

tmp <- df %>% 
  mutate(index = row_number()) %>%
  filter(Group=="Fnt",
         str_sub(FileName,8,-5) %in% num,
         ClassLabel %in% 26:51)

# New dataframe as df
df[tmp$index,] <- df[tmp$index,] %>%
  mutate(ClassLabel = ClassLabel-26)

# delete fnt 0941-0944 for all classes
num <- formatC(941:944, width=4, flag='0')
tmp <- df %>%
  mutate(index = row_number()) %>%
  filter(Group=="Fnt",
         str_sub(FileName,8,-5) %in% num)
df <- df[-tmp$index,]

# c to C, o to O, s to S, u to U, v to V, w to W, z to Z
label_kept = c(3,15,19,22,23,24,26)-1
label_rm = label_kept+26
tmp <- df %>%
  mutate(index = row_number()) %>%
  filter(ClassLabel %in% label_rm)
df[tmp$index,] <- df[tmp$index,] %>%
  mutate(ClassLabel = ClassLabel-26)

# make label to 0-44
original = c(29:39,41:43,45,46,50)
new = c(28:44)
for(i in 1:length(original)){
  tmp <- df %>%
    mutate(index = row_number()) %>%
    filter(ClassLabel %in% original[i])
  df[tmp$index,] <- df[tmp$index,] %>%
    mutate(ClassLabel = new[i])
}

################################################################################

set <- df %>%
  group_by(Group, ClassLabel) %>%
  tally()

set.seed(479)
category <- c()
for (i in 1:nrow(set)) {
  n <- set$n[i]
  train <- round(n*0.6,0)
  test <- round(n*0.2,0)
  valid <- round(n*0.2,0)
  
  diff <- n - (train+test+valid)
  train <- train + diff
  
  v <- rep(1:3, c(train,test,valid))
  v <- sample(v)
  
  category <- c(category,v)
}

df <- cbind(df, category) %>%
  rename('Class Label'=ClassLabel,
         'File Name'=FileName)

train_df <- df %>% filter(category==1) %>% select('Class Label', 'File Name')
test_df <- df %>% filter(category==2) %>% select('Class Label', 'File Name')
valid_df <- df %>% filter(category==3) %>% select('Class Label', 'File Name')

#Without correction on class label
#write.csv(train_df, "TrainingGood.csv", row.names=FALSE)
#write.csv(test_df, "TestingGood.csv", row.names=FALSE)
#write.csv(valid_df, "ValidationGood.csv", row.names=FALSE)

#With correction on class label
write.csv(train_df, "TrainingCorrect_combinedULcases.csv", row.names=FALSE)
write.csv(test_df, "TestingCorrect_combinedULcases.csv", row.names=FALSE)
write.csv(valid_df, "ValidationCorrect_combinedULcases.csv", row.names=FALSE)
