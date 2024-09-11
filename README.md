# ATP Tennis Prediction Model & Dashboard
A Personal Project by AJ Beiza Showcasing Data Collection, Data Prepatation, Data Cleansing, Feature Engineering, Model Building, &amp; Model Evaluation.

# Link to Dashboard Demo

### Hosting a dashboard that trains 3 machine learning models on the cloud is too resource intenstive to host for free. As an alternative, you can see how the dashboard works in this youtube link: https://youtu.be/YbRoVn88bIY

# Purpose:

#### I started playing tennis about two years ago and have loved it ever since. I try to play as much as possible, but between trying to study and trying to obtain internships, I do not have as much time to play. While looking for project ideas, I realized that I could continue my love for tennis while also creating a project that showcases my data and programming skills. While it is obvious that through doing this project, I learned a good bit more about data processing, data cleansing, feature engineering, model building, and model evaluation. However, what may be less obvious is that I also learned a great deal about tennis as a game. I learned what statistics are most important for a player to win matches (specifically about how important a player's return game is) and how surfaces can affect match outcomes. I am excited to be able to apply the knowledge from this project both into future projects/careers and on to the tennis court!

![image](https://github.com/user-attachments/assets/a584d9d8-e23b-4d95-b424-083fcb29ece1)

##### A picture of my favorite tennis players: Rafael Nadal & Daniil Medvedev


# The Data:

#### All of the data used in this project was obtained from Jeff Sackman’s “tennis-atp” github repository (https://github.com/JeffSackmann). The dataset contains every ATP match that has been played since the 70s, along with a dataset of player information and a dataset of player rankings. The dataset does lack some matches and is not updated very frequently, but it is the BEST source of information one can get without paying for APIs or webscraping the entire ATP website. The data I chose to train my models on was from the year 2005 and onward.

# Data Cleansing and Feature Engineering:

## Differentiating Olympic Matches

#### In the dataset, Olympic matches are listed as being at the same level as ATP 250 & ATP 500 matches, which I do not feel is accurate. The meaning of winning at an Olympics is far different from winning a mere ATP 250. Thus, the two tournament levels need to be differentiated.

```
def code_olympics(row):

    if str(row["tourney_name"])[-8:]=="Olympics":

        row["tourney_level"] = "O"

    return row
```

## Calculating Aggregate Player Statistics

#### In the dataset provided, player stats are given for the winner and the loser and the statistics pertain only to the match. The statistics are also **not in percentages** (e.g. # First Serves in instead of % First serve in). To make an accurate model, I had to make aggregate percentage statistics (e.g. % break points converted) for each player and save them in a different dataframe to later merge it back with the match dataframe to train the model on those aggregate statistics.

#### The only statistics provided were for serve statistics, however, with some basic math and logic, you can calculate a player's return statistics by analyzing their opponent's serve statistics

##### For example, a players amount of 2nd return points won is equal to the amount of service points the opponent had minus the amount of valid 1st serves the opponent made minus the amount of second serve points the opponent won:

$$
\text{Second Return Points Won} = \text{Opponent Service Points} - \text{Opponent 1st Serve Points} - \text{Opponent 2nd Serve Points Won}
$$

####

##### I also calculated player win rates across various surfaces and tournament levels (example with Daniil Medvedev stats)

Surfaces           |  Tournament Levels
:-------------------------:|:-------------------------:
![image](https://github.com/user-attachments/assets/2379d8a2-e214-42e9-aa2b-565336380ca4)  |  ![image](https://github.com/user-attachments/assets/2b1c4b2a-100e-4da0-bc0a-ef47120f2bce)

## Omitting Retired Matches

#### Retired matches (ie walkovers) happen when one player concedes the game, normally due to injury. Wins from walkovers do not say much about the winner's actual skill and thus, may confuse a model so I decided to get omit them.

![image](https://github.com/user-attachments/assets/3d740c43-6025-4206-9839-092698c7b0d1)

## Home Advantage

#### I wanted to see if home advantage is a thing in tennis (after my analysis, it did not seem to be the case). To do this, I used the information I had about where a player was born and what tournament they were playing in. I had ChatGPT help me type out a map that maps every ATP tournaments to the country it is played in. I then compared the country of the tournament to the country of each player. If there was a match, a player is given a 1, else 0 (e.g. I one hot encoded if a player was playing at home) 
```
def player_in_home(row):
    winner_at_home = 1 if row["winner_ioc"] == row["tourney_country"] else 0
    loser_at_home = 1 if row["loser_ioc"] == row["tourney_country"] else 0
    
    return pd.Series([winner_at_home, loser_at_home], index=["winner_at_home", "loser_at_home"])

match_df[["winner_at_home", "loser_at_home"]] = match_df.apply(player_in_home, axis=1)
```

## Recent Performance

#### An aggregate win rate percentage does a fair job at describing how good a player is. But as with all sports, players go through ups and downs where they are playing better than average or worse than average. How can we capture this without solely focusing on recent matches? After initially training the models, I came across a blog of someone who made a very similar model to me (https://nycdatascience.com/blog/student-works/utilizing-data-to-predict-winners-of-tennis-matches/). They suggested using a "recent performance" metric which is calculated as follows:

$$
\text{Overall Win \%} + \log_{10}\left(1 - \text{Overall Win \%} + \text{Last 6 Months Win \%}\right)
$$

$$
\text{Where the log expression is a "penalty"}
$$

#### This was a very resource intensive metric to calculate, but it was worth it as it increased the accuracy of my models

## Reducing the Bias of Winner and Loser Columns

#### The statistics in the given match dataset were split into winner stats and loser stats. The existence of columns for the winner stats will perpetuate ***bias*** within the classifier, as it will quickly learn to weight the winner statistics more than the loser statistics. To counter this and prep the data for training, we need to get rid of the ***winner/loser*** schema and instead implement a ***player1, player2*** schema and randomly decide if player1 or player2 will inherit the winner stats for each row (the other player will inherit the loser stats). In the winner column, 1 designates that player1 won and a 0 designates that player2 won.

## One Hot Encoding Playing Hand

#### It is commonly heard in the tennis community that left handed players have inherent advantages against their right handed counterparts because of their ability to hit cross-court forehands to their opponents backhand (the backhand is normally weaker than a forehand). I wanted to test this theory out, and I had the data to do so. However, I first needed to encode the 'playing hand' values of my dataset to be either 1 or 0

```
match_df['p1_hand_encoded'] = match_df['p1_hand'].apply(lambda x: 1 if x == 'R' else 0)
match_df['p2_hand_encoded'] = match_df['p2_hand'].apply(lambda x: 1 if x == 'R' else 0)

match_df = match_df.drop(columns=["p1_hand", "p2_hand"])
```

## Removing/Imputing Null Values

#### The match dataset had ~1,263 missing data points dispersed throughout the age, rank, and height columns. While not entirely necesarry for some tree models, having no null values is a requirement for logistic regression, so the null values had to be either removed or imputed. The null values were as follows:

![image](https://github.com/user-attachments/assets/635db5c5-c90b-4cb7-bc2b-c0a63a68a9c2)

#### Since the amount of null values for age and rank was so insignificant (the dataset had >50,000 rows) I decided it would not hurt my model to drop entries with missing age or rank values. However, there was a lot more missing height values. Since height tends to be normally distributed, I decided it would be sufficient to impute null height values with the mean height.

#### This is the distribution of player heights in the dataset:

![image](https://github.com/user-attachments/assets/24d16135-9b43-4160-b657-50c9f086c44b)


# Model Testing and Model Training

## Baseline Prediction

#### For our classifier models to be worth anything, they must be more accurate than simply picking the higher seeded player to win each time. That is, the models must be able to sometimes predict upsets.

![image](https://github.com/user-attachments/assets/ac222321-8de4-41a9-bbe0-8b951563bb2a)

#### So we should aim for anything >66%, although the higher the better of course!

## Random Forest Classifier

#### The Random Forest Classifier is a fairly obvious first choice for a sports predicting model. It is resistent to overfitting, is fairly quick to train, does not require scaling of data, and is also very accurate. After fitting the the model and using GridSearchCV to optimize the hyperparameters, I was able to achieve a k-fold cross validation score of 71% (5 folds)!

#### These were the most important features for the Random Forest model to make a prediction:

![image](https://github.com/user-attachments/assets/66cc6e53-91e9-4167-ae43-bafa3cf13e51)

## XGBoost

#### The newest statistical model that I have used for this project, XGBoost is renowned for its accuracy and the other pros that make tree based algorithms so great. After training the XGBoost model and grid searching for the best parameters, I was able to achieve a out-of-sample test accuracy score of %70.61. 

#### These were the most important features for the XGBoost model to make a prediction:

![image](https://github.com/user-attachments/assets/f0dbdce4-4dd6-4dad-b228-335e573c3c35)

## Logistic Regression

#### Logistic Regression is a fairly old statistical model, but still performs great when it comes to binary classification problems. Before fitting the model to my date, I scaled all of the features, which is good practice for logistic regression models. Then, to reduce overfitting and make computations easier, I performed Recursive Feature Elimination on my dataset to find the set of features that optimizes model accuracy. Recursive Feature Elimination yielded 28 features, which can be seen in the importance graph below. After using the optimal features and performing a grid search to optimize hyperparameters, I got the model to have an out-of-sample test accuracy of 71.09%!

#### These were the most important features for the Linear Regression model to make a prediction:

![image](https://github.com/user-attachments/assets/56ffb891-288d-42c3-b725-09014a53f057)

## Model Analysis

#### I know how these models perform overall, but how do they perform across different playing surfaces or across different tournament levels. Is one model more likely to predict upsets on certain surfaces or in certain tournament levels? To test this, I diveded a testing dataframe into different surfaces and different tournament levels and plotted all of the models' accuracy across different surfaces and tournament levels.

![image](https://github.com/user-attachments/assets/ea1da441-71f9-42f1-8fdf-e54e9458b6e5)

### As we can see by the graphs, it appears that **grass** is the best surface for our 3 models to predict upsets on followed by clay and then hard court. Additionaly, the logistic regression model appears to be great at predicting Olympic matches, with a high upset accuracy as well. It appears that **Grand Slam matches and Olympic matches** seem to be the easiest to accurately predict across the 3 models.

# Future Room for Improvement (TODO list):

- Changing each aggregate performance metric (ie: %first serves made or %break points converted) to incorporate a “penalty” value that rewards recent (last 4-6 months) improvements to performance metrics. Currently the only statistic to implement this is the recent performance metric. This would mean statistics would be more representative of the player right now, as players are always working with coaches to get certain statistics up.

- Adding a head to head feature for both players prior to every match in dataset. I avoided doing this because my computer lacks computational resources and I know that calculating this is a very resource intensive calculation.

- Adding a feature that describes if a player was recently injured (would require some other datasource or API)

- Adding categorical feature about a player’s play style (ie: all court or baseliner) (requires other datasets)

- Adding average serve speed feature for all players (requires other datasets)

- Adding feature for how often a player makes unforced errors (requires other datasets)

- Adding feature for how often a player makes/receives a winner (requires other datasets)

- Further hyperparameter tuning of models

- Adding features so that if one player has not played any matches at a certain level, then the tournament level win percentage for both players is not considered (i.e. both are set to 0).
Same thing with surface win percentage

- Get feedback and decide if users would like the ability to test matchups of players in the past. I.e. allow user to select their own date for the match instead of assuming the match is today.

