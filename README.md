# [Project 1 - Classification (Diabetes)](https://github.com/lcwhite29/Project-Classification)
- This project is designed to try and predict who has diabetes and who does not given some biological data about patients.
- The data used in the project comes from [Kaggle](https://www.kaggle.com/datasets/ashishkumarjayswal/diabetes-dataset?resource=download).
- This project includes data exploration using several Python libraries such as seaborn, matplotlib, pandas and numpy.
- Then using scikit-learn a logistic regression model is developed and tested against the dataset. This model has an accuracy of around 0.8 which is reasonably good given only 8 columns of biological data.
- After this, I try to find an improved model using deep learning. This model ends up having the same accuracy of 0.8. However, it might be a better model to use in come medical contexts as it is more likely to predict that people have diabetes. Therefore, it could be used as an initial warning for diabetes.
- If I could spend more time on this project I would try to optimise the model some more. Additionally, I would hope that some more data could collected which has a higher correlation with diabetes and some more patients to further refine the model.

# [Project 2 - Regression (Medical insurance)](https://github.com/lcwhite29/Project-Regression)
- This project attempts to find a model which can accurately predict the price of medical insurance.
- The data used in the project comes from a [raw](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv) data source.
- This project includes some data exploration using seaborn, matplotlib, pandas and numpy to find possible correlations between prices and six different characteristics of a person.
- Then using scikit-learn a linear regression model is used to try and predict the prices.

# Project 3 - Clustering (IPL Cricketers)
- In this project, we use data from IPL 2023 and a clustering algorithm to split players into different categories depending on stats like the number of wickets they got and the number of runs they scored.
- The data used in this project comes from [Kaggle](https://www.kaggle.com/datasets/purnend26/ipl-2023-dataset).
- In particular, I was interested in working out which players had a good IPL with the ball and which players had a good IPL with the bat. This required concatenation of the bowling and batting datasets using an inner join.
- Then I did some cleaning of the dataset before using skikit-learn's k-means clustering algorithm to cluster the players in to two different categories.
- Then I used seaborn's hue feature to visualise the results.
