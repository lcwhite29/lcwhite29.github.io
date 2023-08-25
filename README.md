# [Project 1 - Classification (Diabetes)](https://github.com/lcwhite29/Project-Classification)
- This project is designed to try and predict who has diabetes and who does not given some biological data about patients.
- The dataset used in the project comes from [Kaggle](https://www.kaggle.com/datasets/ashishkumarjayswal/diabetes-dataset?resource=download).
- This project included data exploration and visualisation using several Python libraries such as seaborn, matplotlib, pandas and numpy.
- Using scikit-learn I created a logistic regression model which I tested using a train test split. The logistic regression model has an accuracy of around 0.79 which is reasonably good given only 8 columns of biological data within the dataset.
- After looking at the logistic regression model, I then tried to find an improved model using deep learning. This new model ends up having the same accuracy of 0.79. However, it might be a better model to use in some medical contexts as it is more likely to predict that people have diabetes. Therefore, it could be used as an initial warning for diabetes in patients.
- If I could spend more time on this project I would try to optimise the model some more. Additionally, I would hope that some different data could be collected that has a correlation with diabetes. As this along with more patient data would help to refine the model.

# [Project 2 - Regression (Medical insurance)](https://github.com/lcwhite29/Project-Regression)
- This project attempts to find a model that can accurately predict the price of medical insurance.
- The data used in this project comes from a [raw](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv) data source.
- This project included data exploration using standard liberties such as seaborn, matplotlib, pandas and numpy. Using these to find possible correlations between medical costs and the six different characteristics of a particular person.
- I used scikit-learn to develop a linear regression model to predict the prices. This had an absolute error of around 4000 and the square root of the mean squared error was around 6000.
- To try and improve on this I decided to drop some of the medical costs which were at the excessive end as the model found these hard to predict. Instead, I hoped that I could better predict the majority of the medical costs by dropping these outliers.
- Doing this improved the results as the absolute error dropped to around 2500 and the square root of the mean squared error dropped to around 4250.
- This meant the model could predict results better if the price was not an outlier.
- To try and improve on this I made a deep learning model using neural networks.
- The deep learning model has an absolute error of around 1500 and the square root of the mean squared error is around 4300.
- The model is particularly good at predicting prices up to around 15000 but like the linear regression model struggles beyond that point.
- If I spent more time on this project I look more closely at what factors affect the price of someone's medical insurance costs. I would also seek more data to explain the outliers and some of the higher medical costs.

![](Images/Picture_1.png)

# Project 3 - Clustering (IPL Cricketers)
- In this project, we use data from IPL 2023 and a clustering algorithm to split players into different categories depending on stats like the number of wickets they got and the number of runs they scored.
- The data used in this project comes from [Kaggle](https://www.kaggle.com/datasets/purnend26/ipl-2023-dataset).
- In particular, I was interested in working out which players had a good IPL with the ball and which players had a good IPL with the bat. This required concatenation of the bowling and batting datasets using an inner join.
- Then I did some cleaning of the dataset before using skikit-learn's k-means clustering algorithm to cluster the players into two different categories.
- Then I used seaborn's hue feature to visualise the results.
