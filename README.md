# [Medical insurance cost model - Regression with deep learning](https://github.com/lcwhite29/Project-Regression)
- This project attempts to find a model that can accurately predict the price of medical insurance.
- The data used in this project comes from a [raw](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv) data source.
- This project includes data exploration using libraries such as Seaborn, Matplotlib, Pandas and NumPy.  Using these libraries, I explored the possible correlations between insurance prices and the six different characteristics of a particular person.
- I initially used sklearn to develop a linear regression model to predict insurance prices. This linear regression model had an absolute error of around **$4000** and the square root of the mean squared error was around **$6000**.
- Below is an image of the correct values against the predictions.

![](Images/Picture_1.png)

- To improve this model, I decided to drop some of the insurance prices in the excessive price range (outliers) as the model found these hard to predict. I hoped by dropping the outliers that the predictions for most of the insurance prices would improve.
- Doing this improved the results as the absolute error dropped to around **$2500** and the square root of the mean squared error dropped to around **$4250**.
- Below is an image of the correct values against the predictions for the new model.

![](Images/Picture_2.png)

- The linear regression model had a lower error if the price was not in the outlier range.
- To improve on this model,  I made a deep learning model using neural networks. Using TensorFlow and Keras I was able to make a regression model.
- I tested this model on the same dataset again without the outliers.
- This deep learning model has an absolute error of around **$1500** and the square root of the mean squared error is around **$4300**.

![](Images/Picture_3.png)

- The model is particularly good at predicting prices up to around $15000, but like the linear regression model, this model struggles to predict beyond that price.
- If I were to spend more time on this project, I would look more closely at what factors affect the price of someone's medical insurance costs. I would also seek more data to explain the outliers and some of the higher insurance prices people have been charged.

# [Grouping IPL Cricketers -  Clustering](https://github.com/lcwhite29/Project-Clustering)
- In this project, I use data from the IPL 2023 along with a k-means clustering algorithm to split players into different categories depending on stats like the number of wickets they got and the number of runs they scored.
- The data used in this project comes from [Kaggle](https://www.kaggle.com/datasets/purnend26/ipl-2023-dataset).
- In particular, I was interested in determining which players had a good IPL with the ball and which had a good IPL with the bat. This required concatenation of the bowling and batting datasets using an inner join.
- Using sklearn's KMeans clustering algorithm I was able to cluster the players into three groups according to runs scored and wickets taken. Roughly the clusters were those who scored more than 200 runs, those with more than 8 wickets and those with less than 8 wickets and less than 200 runs. From here we can work out using the data frame which players did well with the bat and which players did well with the ball in particular. Using Seaborn's hue feature I could then visualise the results.

![](Images/Picture_4.png)

- Next, I did the same but for the outer joint.
- Here the clusters are roughly those who scored more than 200 runs, those with more than 6 wickets and those with less than 6 wickets and less than 200 runs. 

![](Images/Picture_5.png)

- This is an interesting visualisation of how players played in the IPL 2023.
- There are plenty more things that I could do with this dataset. Including working out which players had the most impact in the tournament.
- If there are similar datasets from previous years then these could be used to see how players have performed across multiple IPLs.
- These sorts of projects could be used by sports trading companies to help inform their trades for other tournaments.

# [Diabetes Classifier - Classification with deep learning](https://github.com/lcwhite29/Project-Classification)
- This project is designed to try and classify who has diabetes and who does not given some biological data about patients.
- The dataset used in the project comes from [Kaggle](https://www.kaggle.com/datasets/ashishkumarjayswal/diabetes-dataset?resource=download).
- This project included data exploration and visualisation using several Python libraries such as Seaborn, Matplotlib, Pandas and NumPy.
- Image of the headmap of correlations below.

![](Images/Picture_6.png)

- Using sklearn I created a logistic regression model which I tested using a train test split. The logistic regression model has an accuracy of **0.79** which is reasonably good given only 8 columns of biological data within the dataset.
- Image of the confusion matrix.

![](Images/Picture_7.png)

- After looking at the logistic regression model, I then tried to find an improved model using deep learning.
- Using TensorFlow and Keras I was able to make a classification model.
- Image of the loss and val_loss plot.

![](Images/Picture_8.png)

- This new model ends up having the same accuracy of **0.79**. However, it might be a better model to use in some medical contexts as it is more likely to predict that people have diabetes. Therefore, it could be used as an initial warning for diabetes in patients.
- Image of the confusion matrix.

![](Images/Picture_9.png)

- If I could spend more time on this project I would try to optimise the model some more. Additionally, I would hope that some different data could be collected that has a correlation with diabetes. As this new data along with more patient data would help to refine the model.
