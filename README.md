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
- `Avg. TTE` : Average of the duration between each transaction.
- `SD. TTE` : Standard deviation of the duration between each tranaction.
- `CV. TTE` : Avg.TTE / SD.TTE
### Past-X-Year Featrues
- `{feature}_1y` : All of the features above but using only past one year transactions to aggregate.
- `{feature}_2y` : All of the features above but using only past two year transactions to aggregate.



## Cross Validation Result
![result](https://user-images.githubusercontent.com/11977931/178422443-2f78c03b-188e-4424-b56a-fb963b529e6d.png)
