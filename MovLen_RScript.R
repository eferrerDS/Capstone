##----//Provided code//----
start_time <- Sys.time()
##########################################################
# Create edx and final_holdout_test sets ----
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(lubridade)) install.packages("lubridade", repos = "http://cran.us.r-project.org")
if(!require(latexpdf)) install.packages("latexpdf", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(scales)
library(lubridate)
library(latexpdf)
library(recosystem)
library(knitr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")   

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##----//My Code//---------------------------------------------------------------

#Exploratory Analysis----
#Describe the data and show a summary of it.

#row count on edx----
edx_row_cnt <- edx %>% summarise(n())
#count of unique users id----
unique_userids <- edx %>% summarise(users_count = n_distinct(userId))
#count of unique movies id----
unique_movieids <- edx %>% summarise(movies_count = n_distinct(movieId))
#count of unique genders combinations on edx----
edx %>% summarise(genres_comb_count = n_distinct(genres))
#count of unique genders----
list_genres <- unique(unlist(strsplit(edx$genres, split = "\\|")))

unique_genres_cnt<-length(list_genres)

#Statistical summary of rating----
fivenum(edx$rating)

#Ratings Summary plot
boxplot(edx$rating, main = 'Rates')

#Number of ratings per movie
edx%>%group_by(movieId)%>%
  summarise(rates_cnt = n(), ave_rating = sum(rating)/n())%>%
  ggplot(aes(ave_rating,rates_cnt))+
  geom_point(binwidth = .5, color = "black")+
  ggtitle("Average Rating Vs Rate Count")+
  xlab("Average Rating")+
  ylab("Rate Count")

#Zooming for movies with less than 100 rates
edx%>%group_by(movieId)%>%
  summarise(rates_cnt = n(), ave_rating = sum(rating)/n())%>%
  ggplot(aes(ave_rating,rates_cnt))+
  geom_point(binwidth = .5, color = "black")+
  ylim(0,100)+
  ylab("Rate Count") +
  xlab("Average Rating") +
  ggtitle("Movies with less than 100 rates")


#Ratings Histogram
options(scipen = 999)
edx %>% ggplot(aes(rating)) +
  geom_histogram(binwidth = .5, color = "black") +
  ylab("Count (millions)") +
  xlab("Rating") +
  ggtitle("Rating Count") +
  scale_y_continuous(breaks = c(1000000,2000000), labels = c(1,2))

#Plot average rating by movie
edx %>% group_by(movieId) %>%
  summarise(ave_rating = sum(rating)/n()) %>%
  ggplot(aes(ave_rating)) +
  geom_histogram(bins = 20, color = "black") +
  ylab("Count") +
  xlab("Average Rating") +
  ggtitle("Average Rating by Movie")

#Plot average rating by genre combination
edx %>% group_by(genres) %>%
  summarise(ave_rating = sum(rating)/n()) %>%
  ggplot(aes(ave_rating)) +
  geom_histogram(bins = 20, color = "black") +
  ylab("Count") +
  xlab("Average rating") +
  ggtitle("Average rating by genre combination")


##Creating training and testing datasets----------------------------------------

# Test dataset will be 10% of edx
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
training <- edx[-test_index,]
test <- edx[test_index,]


temp <- test %>% 
  semi_join(training, by = "movieId") %>%
  semi_join(training, by = "userId")

# Add rows removed from test set back into training set
removed <- anti_join(test, temp)
training <- rbind(training, removed)


##Linear Model------------------------------------------------------------------


#1-Prediction based on Rating's mean (mu_hat)----

mu_hat <- mean(training$rating)

naive_rmse <- RMSE(test$rating, mu_hat)

rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)

#2-Movie Effect----

movie_ave <- training %>%
  group_by(movieId) %>%
  summarize(bi_hat = mean(rating - mu_hat))

predicted_ratings <- mu_hat + test %>%
  left_join(movie_ave, by = 'movieId') %>%
  pull(bi_hat)

movieEffect_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- bind_rows(rmse_results, tibble(method = "Movie effect", RMSE = movieEffect_rmse))

#3-User Effect----

user_ave <- training %>%
  left_join(movie_ave, by = 'movieId') %>%
  group_by(userId) %>%
  summarize(bu_hat = mean(rating - mu_hat - bi_hat))

predicted_ratings <- test %>%
  left_join(movie_ave, by = 'movieId') %>%
  left_join(user_ave, by = 'userId') %>%
  mutate(pred = mu_hat + bi_hat + bu_hat) %>%
  pull(pred)

userEffect_rmse <- RMSE(predicted_ratings, test$rating)
rmse_results <- bind_rows(rmse_results, tibble(method = "User Effect", RMSE = userEffect_rmse))



#4-Regularization----

lambdas <- seq(0,10,.25)

rmses <- sapply(lambdas, function(x){
  mu <- mean(training$rating)
  
  bi <- training %>%
    group_by(movieId) %>%
    summarize(bi = sum(rating - mu)/(n()+x))
  
  bu <-training %>%
    left_join(bi, by="movieId") %>%
    group_by(userId) %>%
    summarize(bu = sum(rating - bi - mu)/(n()+x))
  
  predicted_ratings <- test %>%
    left_join(bi, by = "movieId") %>%
    left_join(bu, by = "userId") %>%
    mutate(pred = mu + bi + bu) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test$rating))
})
qplot(lambdas, rmses)
rmse_results <- bind_rows(rmse_results, tibble(method = paste(c("Regularization with lambda =", as.character(lambdas[which.min(rmses)])), collapse =  " "), RMSE = min(rmses)))

#5-Matrix Factorization----
if(!require(recosystem))
  install.packages("recosystem", repos = "http://cran.us.r-project.org")
train_matrix <- with(training, data_memory(user_index = userId, item_index = movieId, rating = rating))
test_matrix <- with(test, data_memory(user_index = userId, item_index = movieId, rating = rating))
model_object <- recosystem::Reco()


model_object$train(train_matrix)
y_hat <- model_object$predict(test_matrix, out_memory())
rmse_results <- bind_rows(rmse_results, tibble(method = "Matrix Factorization(recosystem)", RMSE = RMSE(test$rating, y_hat)))
rmse_results

end_time <- Sys.time()

elapsed_time <- end_time - start_time
elapsed_time

test_matrix <- with(final_holdout_test, data_memory(user_index = userId, item_index = movieId, rating = rating))
model_object <- recosystem::Reco()


model_object$train(train_matrix)
y_hat <- model_object$predict(test_matrix, out_memory())
rmse_results <- bind_rows(rmse_results, tibble(method = "Final", RMSE = RMSE(final_holdout_test$rating, y_hat)))
rmse_results
#Enviroment----

#```{r, echo = FALSE}
#version
#```


#Capstone
