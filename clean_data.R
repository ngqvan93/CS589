# Load library ---------------------------------
require(lubridate)
require(dplyr)


# Read data in ---------------------------------
hour <- read.csv('./Data/UCI/hour.csv')
q1.2011 <- read.csv('./Data/Q1_2011.csv')
q2.2011 <- read.csv('./Data/Q2_2011.csv')
q3.2011 <- read.csv('./Data/Q3_2011.csv')
q4.2011 <- read.csv('./Data/Q4_2011.csv')
q1.2012 <- read.csv('./Data/Q1_2012.csv')
q2.2012 <- read.csv('./Data/Q2_2012.csv')
q3.2012 <- read.csv('./Data/Q3_2012.csv')
q4.2012 <- read.csv('./Data/Q4_2012.csv')
  
  
# Clean hour.csv ---------------------------------
hour <- hour %>%
  mutate(dteday = ymd(dteday)) %>%
  select(-yr, -mnth, -weekday, -instant, -casual, -registered, -cnt, -temp)


# Clean quarterly data ---------------------------------
colnames(q1.2012) <- names(q1.2011)
colnames(q2.2012) <- names(q1.2011)
colnames(q3.2012) <- names(q1.2011)
colnames(q4.2012) <- names(q1.2011)


# Merge and clean all quarterly data ---------------------------------
quarter <- bind_rows(q1.2011, q2.2011, q3.2011, q4.2011,
                 q1.2012, q2.2012, q3.2012, q4.2012)

quarter <- quarter %>%
  mutate(Member.Type = ifelse(Member.Type != 'Casual', 'Registered', 'Casual'),
         Start.date = mdy_hm(Start.date),
         dteday = date(Start.date), 
         hr = hour(Start.date),
         Duration = as.numeric(as.duration(hms(as.character(Duration))))) %>%
  select(-Start.date, -End.date, -End.station, -Bike.)


# Clean data, join with hour.csv ---------------------------------
full.df <- right_join(hour, quarter, by = c('dteday', 'hr')) %>%
  arrange(dteday, hr)

names(full.df) <- c('date', 'season', 'hour', 'holiday', 'workingday', 'weathersit', 
                    'feeling_temp', 'humidity', 'windspeed', 'duration', 'station', 'type')



# Split train-test ---------------------------------


temp <- full.df %>%
  group_by(station) %>%
  summarise(count = n())

ix <- which(temp$count == 1)
single.station <- temp$station[ix]

single <- full.df %>%
  filter(station %in% single.station)

multiple <- full.df %>% 
  filter(!(station %in% single.station))


scalar <- (0.7*nrow(full.df) - nrow(single))/(nrow(full.df) - nrow(single))


set.seed(0)
multiple$random <- runif(nrow(multiple), 0, 1)

multiple <- multiple %>%
  group_by(station) %>%
  mutate(n = floor(n()*scalar)) %>%
  arrange(station, random)

train <- multiple %>%
  group_by(station) %>%
  filter(row_number() <= n) %>%
  select(-n, -random)

train <- bind_rows(single, train)

test <- multiple %>%
  group_by(station) %>%
  filter(row_number() > n) %>%
  select(-n, -random)


write.csv(train, 'train.csv', row.names = F)
write.csv(test, 'test.csv', row.names = F)

