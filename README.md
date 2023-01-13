# Machine Learning Fundamentals - Data Science Retreat 2023 course
Repository for the 2023 edition of my Machine Learning Fundamentals course, held quarterly at the Berlin's bootcamp Data Science Retreat.

## Contents
This course introduce to the basic ML algorithms that are foundational for more advanced applications. The course is developed over the course of three days. Despite having a flexible agenda depending on the specific class needs, I usually try to cover the following topics with this timing:

* Day 1: the basic problems of Machine Learning
  * Introduction to Supervised Learning
  * Evaluating supervised learning models results
  * Bias and Variance Tradeoff
  * Linear Regression
  * Regularization
  * Optimization Methods
  * Metrics for Classification

* Day 2: a focus on the most important ML Algorithims for classification
  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Gradient Boosting

* Day 3: Unsupervised Learning and additional topics
  * Support Vector Machines
  * Elements of Unsupervised Learning
  * Principal Component Analysis
  * Cluster Analysis: K-Means
  * Additional topics depending on the class.

Day 3 can vary quite a lot depending on the class. Often here I complete some topics from the previous days, and/or answer to specific questions from students that we didn't have the time to delve into earlier. 

The course is meant to provide clear intuitions about the most important concepts in ML. For this reason, we won't focus on applications but more on ideas. Despite that, I prepared a few notebooks that will be used to experiment with the ML models especially to get a better idea of how hyperparameters affect the results. Please note that these notebooks are currently (January 2023) still a working in progress and I will work on them and expand them during 2023. 

## Teaching Style
I will mainly use a whiteboard (either a physical one, or with an iPad) discussing concepts and sketching notes on the fly. There aren't already prepared slides for this course. The idea is to have a discussion on the course topics and to involve every student as much as possible. 

I will present theoretical concepts in an accessible, light-math way, trying to put everything in a broader context. Practical sessions will happen during the course and students will be invited to complete some code snippets and to experiment with the already provided code. 

For the practical sessions, we'll make use MLFlow (https://mlflow.org/), a very important and popular tool used for MLOps and that will come in handy to check the experiment results (*experiments tracking*).

## Set-up for practical sessions
Use the `requirement.txt` file to install the needed dependencies. 

For the applications, the NY taxi trip data are used. Go to https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page and manually download "Green Taxi Trip Records (PARQUET)" under `data/input/data_raw`. Do it for the month of January 2022. These are quite big datasets so do the following to reduce the data size:

- `cd data`
- `python get_final_data.py` 

To access the mlflow UI, just run from the repository folder the following command `mlflow ui --backend-store-uri sqlite:///mlflow.db` and open on a web browser the generated ip (e.g. http://127.0.0.1:5000/)

## External Resources
A very good resource that I will refer to during the class is the following repository: https://github.com/dvgodoy/ML_Fundamentals. It contains interective widgets to explore the ML models properties. The author is a DSR alumnus and former DSR teacher. 