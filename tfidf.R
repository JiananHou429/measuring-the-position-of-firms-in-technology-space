## ------------------------------------------------------------------------- ##
## Script to calculate pairwise technology similarity between firms by TFIDF ##
## We take pairwise tech similarity in 1980 as an example, similarities in   ##
## other years could be calculated in the same way.                          ##
##                                                                           ##
## "Measuring the Position and Differentiation of Firms in                   ##
##  Technology Space" -- Sam Arts, Bruno Cassiman, and Jianan Hou            ##
##                                                                           ##
## cleaned patent texts are available from Arts et al. (2021)                ##
## https://zenodo.org/record/3515985                                         ##
## patent-firm linkages are available from Arora et al. (2021)               ##
## https://zenodo.org/record/3594743                                         ##
##                                                                           ##
## ------------------------------------------------------------------------- ##

library(readr)
library(Matrix)
library(bigmemory)
library(janitor)
library(plyr)
library(tidyr)
library(quanteda)
library(lsa)
library(Rcpp)
library(matlab)
library(tilting)
library(tidyverse)

## generate functions to calculate TFIDF weights
#  term frequency (TF)
term.frequency <- function(row) {
  row / sum(row)
}

# inverse document frequency (IDF)
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  log10(corpus.size / doc.count)
}

# TFIDF
tf.idf <- function(tf, idf) {
  tf * idf
}

## data loading
setwd("./data/")
## the following patent texts are collected from Arts et al. (2021) https://zenodo.org/record/3515985   
all_patent_text <- read_csv("all_patent_text.csv")

# "patent_portfolio" is compiled based on the "patent-gvkey" linkages developed by Arora et al. (2021) (see 
# https://zenodo.org/record/3594743). For each firm i in year t, it documents patents contained in the portfolio (with filing years between t-5
# and t-1). The file contains one row and three columns for each "gvkey-patent" linkage. The first column contains gvkey, the second column contains year, and third column contains patent number. 
patent_portfolio <- read_csv("patent_portfolio.csv")
                   
p1980<-subset(patent_portfolio,year==1980)
text_1980<-merge(x=p1980,y=all_patent_text,by="patent",all=FALSE)
text_1980<-text_1980[,c("gvkey","text")]
text_1980 <- text_1980[order(text_1980$gvkey),]
text_1980$gvkey <- as.factor(text_1980$gvkey)

## concatenate patent texts at firm level.  
text.sep<- ddply(text_1980, .(gvkey), summarize, text=paste(text, collapse  =""))

## generate all possible firm combinations in order to store the pairwise similarities between them.
text.sep$order<-seq(1, 876, by=1)
## 876 corresponds to the length of text.sep
simp_text<-text.sep[,c("gvkey","order")]
simp_text$order1<-simp_text$order
simi_1980 <-simp_text%>% expand(order1,order)
simi_1980<-subset(simi_1980,order1<order)
simi_1980 <- simi_1980[order(simi_1980$order1,simi_1980$order),]

## vectorize aggregated patent texts at firm level and store all vectos in a matrix
text.tokens <- tokens(text.sep$text, what = "word")
text.tokens.dfm <- dfm(text.tokens, tolower = FALSE)
text.tokens.matrix <- as.matrix(text.tokens.dfm)

## TFIDF adjustment
text.tokens.tf <- apply(text.tokens.matrix, 1, term.frequency)
whole.idf <- apply(text.tokens.dfm, 2, inverse.doc.freq)
text.tokens.tfidf <-  apply(text.tokens.tf, 2, tf.idf, idf = whole.idf)

## cosine similarity calculation
ptm <- proc.time()
A<-text.tokens.tfidf
normA<-col.norm(A)
A<-sweep(A,2,normA,FUN='/')

B<-A
n<-876
cos.out<-matrix(0,n,n-1)
for (j in 2:n-1)
{
  B <- B[,c(2:n,1)]
  cos.out[,j]<-colSums(A*B)
  if(j %% 200==0) {
    print(j)
  }
}

cos.out<-t(cos.out)
cosine_similarity_vector<-cos.out[!fliplr(lower.tri(cos.out,diag=TRUE))]
proc.time() - ptm

## export similarity matrix to a csv file
simi_1980$similarity_tfidf<-cosine_similarity_vector
simp_text$gvkey1<-NULL
simi_1980<-merge(x=simi_1980,y=simp_text,by="order1",all=FALSE)
simi_1980<-simi_1980[,c(1:4)]
colnames(simi_1980)[c(2)]<-"order"
colnames(simi_1980)[c(4)]<-"gvkey1"
simi_1980<-merge(x=simi_1980,y=simp_text,by="order",all=FALSE)
simi_1980<-simi_1980[,c(1:5)]
colnames(simi_1980)[c(2)]<-"order1"
colnames(simi_1980)[c(5)]<-"gvkey2"
simi_1980$year <-1980
write.csv(simi_1980, file = "simi_1980_tfidf.csv")
