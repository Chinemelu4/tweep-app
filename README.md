# tweet-stalker-app

## Problem
Social media is a great place to meet new people and get access to many sources of information speedily. It is also a place to be weary off, and such you will be wise to restrict the kind of people that have access to you as well as the kind of information that can get to you. But how do you do that?

## Solution
Focusing on the social media provider twitter, this prototype application helps you quickly analyze any twitter user to know whether the person is someone you would like to follow or engage with. How does affect the kind of info on your feed? The kind of people you follow affects the kind of information om your feed because twitter works on some recommendation systems bringing things that your followers and those you follow to your timeline. So by providing a summary analysis of any twitter user you get to quickly see whether the person is worth your followership. The application carries out analysis on the tweets of the user and provides information like *number of tweets*, *average number of likes* and *retweets per tweet*, *frequently used words*, *sentiment analysis* and also a machine learning model I trained on twitter's topic segementation data which I extracted to classify the tweets into topics the individual is interested in.

## Improvements
Build an ingestion pipeline to periodically extract tweets labelled by topics, check performance against present model and retrain model when their is model decay or data drift.
