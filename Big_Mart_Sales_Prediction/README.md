# Big Mart Sales Prediction

The dataset is taken from the hackaton on Analytics Vidhya. The data set 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and predict the sales of each product at a particular outlet.

In the data set 8523 data points for training and 5681 data points for prediction. Almost for every data point in train data set there are 11 features with some missing values and outliers.

BigMartSales.ipynb file contains the **exploratory data analysis, feature engineering and transformation, outlier detection and treatment and statical analysis with correlation between features of the data set** and after data preprocessing the final data set contains **19 features which includes the additional 3 features after feature engineering with correlation of 0.49**.

The final data set is used for training models like linear regression, knearest neighbors, decision tree, random forest etc.

After training, XGBoost, CatBoost and LightGBM performed better than others. Especially LightGBM model given high weights for the features having high correlation with target variable.

Finally I ended up having a **test MSE of 1148.033** and gaining a rank of **295 out of 39655**(on 5th December 2020) participants on Analytics Vidhya Hackathons.
