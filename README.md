# Campaign Response Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tanatiem/campaign-response/blob/main/notebooks/campaign_response_prediction.ipynb)

**Task:** Binary Classification  
Predict whether or not a customer responds to the marketing campaign.

## Datasets
- `Retail_Data_Transactions.csv` contains customer transactions.
- `Retail_Data_Response.csv` contanis customer response. This is our label.
- Using `pandas` to aggregate and transform transactional data to the customer grain or Customer Single View to be used as training samples for fitting the model.

![Data](https://user-images.githubusercontent.com/11977931/178426327-7a235d1f-594c-4c56-a1cb-9bcde402eb64.png)

## Cohort Analysis
As a part of exploratory data analysis, we can make the Cohort Chart from our transaction data like this below picture.  
We group customer together based on their first transaction month as in the same cohort. Starting from the cohort month (Y axis), for each passing month (X axis), you can see the numbers. These numbers represent the `Retention Rate` for each cohort and month.  

For example, for all the customers who start visiting our business in `2011-05` (Cohort 2011-05), at the `3rd` month after their first visits, there are `37%` of customers from this cohort come back and purchase again.
![Cohort Chart](https://user-images.githubusercontent.com/11977931/178428530-2cbcd93e-2f11-4333-889d-1072da73bc75.png)

You might notice that starting from `2012-10` cohort. The transaction data that we have doesn't look natural. It may be provided not as a whole but just some parts of it.

---

## Feature Engineering
Even if we only have `transaction amount`, we can aggregate it in many ways.
### RFM Features
- `Recency` : Duration (number of days) from the last transaction to the campaign date.
- `Frequency` : Number of visits or count of transactions.
- `Monetary` : Total spend or sum of transaction amounts.
### Additional to RFM
- `Tenure` : Duration (number of days) from the first transaction to the campaign date.
- `Length of Stay` : Duration (number of days) from the first to last transactions.
- `Ticket Size` : Average spend per visit.
- `SD of Ticket Size` : Standard deviation of spend. This captures the behavior of spending amounts being consistent or fluctuated.
- `CV of Ticket Size` : Coefficient of variation of spend. Basically $\frac{\sigma}{\mu}$ of spending.
- `Average Spend per Month` : Total spend / number of months visited.
- `Average Visit per Month` : Total visit / number of months visited.
### Time to Event
- `Avg.TTE` : Average of the duration between each transaction.
- `SD.TTE` : Standard deviation of the duration between each tranaction.
- `CV.TTE` : SD.TTE / Avg.TTE
### Past-X-Year Features
- `{feature}_1y` : All of the features above but using only past one year transactions to aggregate.
- `{feature}_2y` : All of the features above but using only past two year transactions to aggregate.

## Mutual Information
With `sklearn.feature_selection.mutual_info_classif`, we can get the mutual information score for each feature in order to estimate the prediction power for our classification problem. We then try to select features based on these values.  

![image](https://user-images.githubusercontent.com/11977931/178439149-49d4bec4-dec3-43bf-8231-46740221a1c2.png)

---

## Model Experiments
Experimenting with feature sets and models with cross validation to see what works.
### Feature sets
- `RFM` only basic Recency, Frequency, Monetary. Because RFM is the most basic form of features, let's treat this as a baseline.
- `LTD` refers to the Feature Engineer section, this covers RFM, Additional-to-RFM, Time-to-Event features, not including Past-X-Year features. These features are generated using the whole transaction data for each customer, regardless of how old it is. They are Life-to-Date information.
- `SET1` includes almost all features but only some features with small Mutual Information score are dropped.
- `ALL` using all generated features.
### Models
`RandomForest`, `ExtraTrees`, `XGBoost`, and `LightGBM`
### Experiment: 
A combination of a feature set and model.
### Cross Validation
Using `RepeatedStratifiedKFold` with `5` folds and `6` repeats (each repeat splits folds with different randomization)  
We then get `30` of training and validation scores for each experiment.

---

## Cross Validation Result
- `XGBoost` and `LightGBM` seems to work well.
- `RFM` feature set yield poor results. (Of course, it's only 3 features)
- `LTD` feature set has more additional features from RFM. But they all are generated using all transactions. This improves the performance a bit from the RFM set.
- Seeing that `SET1` and `ALL` feature sets have much better performance, this means that the features generated using data from past 1 year, and 2 years are a big help.
- This demonstrates the power of `Feature Engineering`. Even though, we only have `transaction amount` data, we can raise our model performance with different aggregation techniques.

![image](https://user-images.githubusercontent.com/11977931/178443423-789f2c74-d754-42d6-80f4-7ebd7b69731a.png)

![result](https://user-images.githubusercontent.com/11977931/178422443-2f78c03b-188e-4424-b56a-fb963b529e6d.png)

## Future improvements
- We now have a shortlist of our model selection so that we can work on Hyperparameter tuning later.
- Experimenting with resampling techniques to address imbalance problem, such as, under-sampling, over-sampling, SMOTE, etc.
- More feature engineering.

Thanks!


