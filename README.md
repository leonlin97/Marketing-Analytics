# Marketing-Analytics Acedemic Project

This project used several datasets from LobsterLand, a fictional theme park created by Professor Greg Page from AD654 Marketing Analytics at Boston University.

Tools used: `Jupyter Notebook`, `Tableau`

Technical skills: `Pyhton` (Data Analysis and Visualization)

A summary of the topics includes:

[**1. Data Exploratory and Manipulation**](#project-i-Explorary-Analysis-And-Visualization)
- Identified Null and innormal value/text, replacing them with apporate value based on business goals.
- Grouped, conditionally aggregated and visualized data to discover insights.

[**2. Visitor Segmentation for Custimzed Marketing Strategies**](#Visitor-Segmentation)
- Used `k-means clustering` method to build a behavior segmentation model in order to better understand the visiot traits.
- Created visualization to understand the trait of each cluster, generating business action for `off-season Winter` activities/promotions planning.

[**3. Conjoint Analysis from Survey data to Realize User Preferences on Arcade**](#Conjoint-Analysis-on-Survey-Data)
- Survey data includes areas like favorate category of sport game, music, prize... etc. (each category has 3 to 5 options to choose)
- Dummified variables and used `linear regression` to compare the rating of each survey to realize what are favored by most of the visitors, then recommended to the management level on how to furnish the theme park.

[**4. Prediction of Preference for Indoor or Outdoor Activities**](#Project-III-Prediction-of-Preference-for-Indoor-or-Outdoor-Activities)
- Built `Logistic Regression` and `Random Forest` model to predict if a visitor would prefer 'Indoor' or "Outdoor' activities.
- Compared the result and the accuracy of two models, generating actions for the business strategies.

[**5. Hypothesis Testing**](#Project-IIII-Hypothesis-Testing)
- Used `t-test` to find out if people returned more items during holiday, weekday and weekends.
- Used `Chi-Square Goodness of Fit` to test whether day type impacts the likelihood that a consumer will return something after making a purchase.

[**6. Marketing Mix Modeling with an Interaction Term **](#Project-V-Marketing-Mix-Model)
- Created interaction term and used `F-Statistic` method to inspect the marketing performance on YouTube Ad, Spotify Ad, and Banner Ad.

## Project I: Explorary Analysis And Visualization
A couple of insighs generated from this projects:
- Comparing the revenue gain from source A and source B during `rainy` day and `not rainy` day. --> Revenue of source B would be greatly affected by the weather factor.
- Low correlation between `temparature` and  `lost items`, so management's assumption: *People were more forgetable on hot days* was rejected.
- Using conecpt of `coefficient of variance` to explain to menegement level about why we cannot directly compare the revenue's relative dispersion from two different sources.
- Revenue from source A and unique visitors are higher during weekends, indicating that weekends have more business oppoetunities to attract visitor and boost revenue.

<table>
  <tr>
    <td>
      <img width="489" alt="image" src="https://github.com/leonlin97/Marketing-Analytics/assets/142073522/9f5d5be8-7b67-43a3-8d85-5a91c7254c41">
    </td>
    <td>
      <img width="531" alt="image" src="https://github.com/leonlin97/Marketing-Analytics/assets/142073522/7c30207b-4a01-42f7-a6ae-25f5c278bd70">
    </td>
  </tr>
</table>

- For revenue gain in Gold Zone (Indoor), visitors spent more on rainy day --> An indicator for us to `focus more on indoor shopping / activities in rainy seasons and days` to compensate the loss of visitor avoiding outdoor ativities.
<img width="530" alt="image" src="https://github.com/leonlin97/Marketing-Analytics/assets/142073522/b662cba3-c59a-4959-a235-613ec0af59ee">

- Number of 2024 pass signups increased rapidly in September. This might be due to the approaching deadline for pass registration, so many people likely decided to purchase the pass to avoid missing the deadline.
<img width="557" alt="image" src="https://github.com/leonlin97/Marketing-Analytics/assets/142073522/2fe93184-33d1-46ae-a288-27fc2a08bc45">

## Project II: Visitor Segmentation and Conjoint Analysis of Survey
This project used `k-means clustering` method to build a behavioral segmentation model in order to better understand Lobsterland's passholders.

Library used: `sklearn`,`matplotlib`, `seaborn`

### Visitor Segmentation
Process to build a segmentation model:
#### Step 1: Data Cleaning

#### Step 2: Data standardization
Scaled the data to `z-scores` to prepare for modeling.
```
passholder_std = pd.DataFrame(StandardScaler().fit_transform(passholder), columns=passholder.columns)
```

#### Step 3: Variable selection
I choose 5 out of 10 variables that are more impactful toward revenue -- `Visited Number`, `Spent on Game`, `Spent on Merchandise`, `Spend on Food`, and `Visited Duration`.

#### Step 4: Find optimal value for "k"
Used `The Elbow Method` to find optimal k value, which would be **6** in this case.
```
sse = {}
for k in range(1, 30):
# Initialize KMeans with k clusters
    kmeans = KMeans(n_clusters=k, random_state=654,n_init=10)
# Fit KMeans on the normalized dataset
    kmeans.fit(cluster_df)
    sse[k] = kmeans.inertia_
# Add the plot title "The Elbow Method"
plt.title('The Elbow Method')
# Add X-axis label "k"
plt.xlabel('k')
# Add Y-axis label "SSE"
plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()));
```
<img width="681" alt="image" src="https://github.com/leonlin97/Marketing-Analytics/assets/142073522/bd887117-7600-4126-a752-8b633f85d0c2">

#### Step 5: Apply k=6 into k-means clustering method
```
kmeans = KMeans(n_clusters=6, random_state=654,n_init=10)
kmeans.fit(cluster_df)
cluster_labels = kmeans.labels_
```

#### Step 6: Visualizing result and labeling each cluster
By visualizing, we can see a clear trait for each group, then lebeled these clusters for easier undestanding.

- `Cluster 0: Time-Rich, Budget-Savvy`

People in this group are staying longer than others, but they spend the least on food, with a bit less on merchandise as well. They are economical and choosy with their food purchases, avoiding additional on-site purchases.

- `Cluster 2: One-and-Done Nibblers`

People in this group stay for short periods, spends the least on games, merchandise and revisits, but spends slightly above average on food. They come for a quick, one-time visit, spends little on non-food items, but indulges slightly on the food offered during their brief stay.

- `Cluster 3 -- Big Spender Foodies`

People in this group spends the most on food, has above average stay times and merchandise spending, but below average game spending and revisits. They love food and indulging in it when they visit, staying longer to enjoy the food offerings and shopping more than playing games or coming back frequently.

- `Cluster 4 -- Game-Focused Regulars`

People in this group spends the most on games, has above average food spending and revisits, but below average stay times and merchandise spending. They focused on the games during their visits, coming back regularly despite not staying overly long per visit or spending much on extras like food and sou venirs. Their priority is the games and returning to play again.

- `Cluster 5: Souvenir-Loving Loyalists`

People in this group spends the most on merchandise, has above average revisits and game spending, but below average stay times and food spending. They focused on purchasing
sou venirs, returning often to shop the merchandise again, but don't prioritize lengthy stays or indulging in food offerings. Their loyalty is shown through repeat visits and merchandise purchases.

![image](https://github.com/leonlin97/Marketing-Analytics/assets/142073522/0c5146a9-e7da-4444-a0d0-0e671bbc0c9e)

#### Business Recommemdations for the coming winter Off-Season period
We can use this segmentation model to understand what traits in each group, then build a customized marketing strategy for each group to maximize profit. For the off-season winter months, I think we should target and engage more with the `Souvenir-Loving Loyalists` group. This is the second largest group in terms of number of people and frequency of revisits, representing a huge business opportunity. We can do the following strategies to attract them more during the winter:



*   Winter-Themed Souvenirs: Create unique, winter-themed souvenirs that are exclusive to your store. This could include items like custom snow globes, holiday ornaments, or cozy winter apparel like scarves and gloves.

*   Winter Events: Host special winter events and activities in and around your store. This might include live music, hot cocoa stations, or holiday crafting workshops. Use these events to draw people in and encourage repeat visits.

*   Frequent Shopper Discounts: Reward frequent shoppers with discounts or exclusive access to limited-edition winter souvenirs. Consider creating a tiered membership program for your best customers.

*   Winter Discounts and Bundles: Offer discounts and bundled deals for purchasing multiple souvenirs. This encourages customers to buy more items during their visit.

On the other hand, the "One-and-Done Nibblers" group should be the last priority for engagement, even though they are the largest group. This is because all of their KPIs are below average, apart from relatively higher food spending compared to others. It is clear that they are just stopping in for a meal rather than being very loyal to Lobster Land.

![image](https://github.com/leonlin97/Marketing-Analytics/assets/142073522/bbb87d34-2c14-4e5d-a048-d0712234e0df)


### Conjoint Analysis on Survey Data

#### Step 1: Dummfied variables
```
new_gold_dummy = pd.get_dummies(new_gold,drop_first = True,columns = ['music_physical','sports_physical','retro_arcade','pinball','tough_skill','coaster_sim','jukebox_run','crane_prizes'])
```

#### Step 2: Built a linear model
```
X = new_gold_dummy.iloc[:,1:]
y = new_gold_dummy['avg_rating']
regressor = LinearRegression()
regressor.fit(X,y)
```

#### Step 3: Realize visitor preferences based on coefficient
```
coef_new_gold = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
```

Based on the result, the mamagement level can make the following changes to the arcade to meet visitor preferences:

* **music_physical:** Dance Dance Revolution
* **sports_physical:** Connect Four Hoops
* **retro_arcade:** Simpsons
* **pinball:** Monopoly
* **tough_skill:** Big Wheel
* **crane_prize:** Ticket bundles
* **coaster_sim:** 540 seconds
* **jukebox_run:** 4 times

### Business Recommendation
Based on the results above, here are some recommendations on marketing strategies to further enhance visitor satisfaction:
* Emphasize and promote the Dance Dance Revolution and Connect Four Hoops machines more.
* Propose a collaboration with Simpsons - for example, obtaining copyrights on machine designs, decorations, and branding across Lobster Land. We could even host a "Simpsons" festival to boost overall profits.
* Re-design the ticket bundles to include prizes from customers' preferred games.

<img width="382" alt="image" src="https://github.com/leonlin97/Marketing-Analytics/assets/142073522/ea3cab37-7d7e-45e3-9981-3682aa109bf4">

## Project III: Prediction of Preference for Indoor or Outdoor Activities
Business Goal: How do we attract more visitors to indoor activities to boost revenue and build customer satisfaction?

From historical data recording each visitor's gender, age, distance, occupation... with their preference on indoor or outdoor activities, my goal is to build a model that find out what factors greatly impact their dicision, then build a strategies for promoting our indoor activities selection and activities.


### Step 1: Data cleaning and transformation

One important thing here, in addition to manipulating data for model preparation, is to identify variables that do not have enough power in showing people's preferences. For example, if there are 50 males and 50 females prefer indoor, then gender might not be a good indicator.

### Step 2: Built a Logistic Regression Model

```
# Dummy variable
preprocessor_lr = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_variable),
        ('cat', OneHotEncoder(drop='first'), categorical_variable)
    ])

# Split data to train and test
X = carnival.drop('preference', axis=1)
y = carnival['preference']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=654)


pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor_lr),
                              ('classifier', LogisticRegression(max_iter=1000, random_state=654))])
pipeline_lr.fit(X_train, y_train)

# Prediction
y_pred_lr = pipeline_lr.predict(X_test)

# Calculating accuracy and AUC for Logistic Regression model
accuracy_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, pipeline_lr.predict_proba(X_test)[:, 1])

# Extracting feature names after one-hot encoding
feature_names_lr = (pipeline_lr.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .get_feature_names_out(input_features=categorical_variable))

# Combining feature names for both numeric and categorical variables
feature_names_lr = np.concatenate([numeric_variable, feature_names_lr])
# Getting coefficients for Logistic Regression model
coefficients_lr = pipeline_lr.named_steps['classifier'].coef_[0]

# Creating a DataFrame for Logistic Regression feature importance
lr_feature_importance = pd.DataFrame({'Feature': feature_names_lr, 'Coefficient': coefficients_lr})
lr_top_features = lr_feature_importance.reindex(lr_feature_importance.Coefficient.abs().sort_values(ascending=False).index).head(4)

print(f'The accuracy of Logistic Regression Model is: {round(accuracy_lr,2)} with AUC = {round(auc_lr,2)}')
lr_feature_importance
lr_top_features
```

### Step 3: Built a Random Forest Model

```
preprocessor_rf = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_variable),
        ('cat', OneHotEncoder(), categorical_variable)
    ])
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor_rf),
                              ('classifier', RandomForestClassifier(random_state=654))])

pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)

# Calculating accuracy and AUC for Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, pipeline_rf.predict_proba(X_test)[:, 1])

# Getting feature importances for Random Forest model
feature_importances_rf = pipeline_rf.named_steps['classifier'].feature_importances_

# Extracting feature names for Random Forest model (no level dropped)
feature_names_rf = (pipeline_rf.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .get_feature_names_out(input_features=categorical_variable))

# Combining feature names for both numeric and categorical variables
feature_names_rf = np.concatenate([numeric_variable, feature_names_rf])

# Creating a DataFrame for Random Forest feature importance
rf_feature_importance = pd.DataFrame({'Feature': feature_names_rf, 'Importance': feature_importances_rf})
rf_top_features = rf_feature_importance.reindex(rf_feature_importance.Importance.abs().sort_values(ascending=False).index).head(4)


print(f'The accuracy of Random Forest Model is: {round(accuracy_rf,2)} with AUC = {round(auc_rf,2)}')
rf_feature_importance
rf_top_features
```

### Step 4: Compared the Accuracy of Two Model

Below Table shows the model accuracy comparing Logistic Regression and Random Forest. Since the Logistic Regression Model has a better performance in both accuracy and AUC, we decided to use the result from logistic regression to generate some insights.

<img width="725" alt="image" src="https://github.com/leonlin97/Marketing-Analytics/assets/142073522/f09dc563-77ce-484b-8cee-db34b820a5c1">

### Step 5: Discovered Insights from Logistic Regression Model

Three type of people would be more preferable to infoor activities:
- Physically sedentary
- Less companion (solo visitor or with 2 to 3 friends/family member)
- Foreigners (People living outside the US)

If the management wants to promote Indoor Activities, they can emphasize on these strategies that greatly impact a person’s decision to choose indoor activities. The strategy can focus on increasing customer satisfaction and attracting more visitors.

#### Increaseing customer satisfaction

- Enhanced the accessibility infrastructure within the theme park
- Provided diverse indoor events, such as food tasting, entertainments shows
- Guide tours and social networking events for solo visitors
- Multi-culture acrade and food vendors within the park

#### Attracting visitors who prefer outdoor activities to indoors
- Group visitor: family-friendly shows and exhibition center; educational program for winter camping
- Indoor sport playground for less sedentary person to get exercise

<img width="704" alt="image" src="https://github.com/leonlin97/Marketing-Analytics/assets/142073522/2a079047-d444-4e77-849a-dd324cbac0b1">

## Project IIII: Hypothesis Testing

This project discovered the pattern of customer return of purchased items during holidays, weekdays and weekends to improve the sales performance.

### Is the return rate is higher for product purchased on holidays, weekdays, and weekends?

After getting the average return proportion on each three day type, I performed t-tests of three combination to see if there are significant differences.

```
a = 0.0167 -- Bonferroni correction from a = 0.05
t_WR,p_WR = stats.ttest_ind(df3.loc[df3['day_code']=='W','return_proportion'].values,
                     df3.loc[df3['day_code']=='R','return_proportion'].values,equal_var=False)
t_WH,p_WH = stats.ttest_ind(df3.loc[df3['day_code']=='W','return_proportion'].values,
                     df3.loc[df3['day_code']=='H','return_proportion'].values,equal_var=False)
t_RH,p_RH = stats.ttest_ind(df3.loc[df3['day_code']=='R','return_proportion'].values,
                     df3.loc[df3['day_code']=='H','return_proportion'].values,equal_var=False)
p_result = [p_WR,p_WH,p_RH]
a_result = ["Reject null hypothesis" if i < a else "Accept null hypothesis" for i in p_result]
ttest_result = pd.DataFrame({
    'Comparison' : ['Weekend x Regular Day','Weenend x Holiday','Regular Day x Holiday'],
    'p value' : p_result,
    'Test Result' : a_result
})
ttest_result
```

The t-test results indicate that there is no significant difference in the return proportion between weekends and regular days (p = 0.512313) or weekends and holidays (p = 0.040996), meaning customers are returning goods at similar rates across these day types. However, the return proportion is significantly different between regular days and holidays (p = 0.008406), meaning customers are actually return more items purchased on holiday comparing to regular day.

I think the management can analyze the types of products purchased and the promotions run on holidays that result in lower return rates, with the aim of replicating successful aspects throughout the year.

<img width="402" alt="image" src="https://github.com/leonlin97/Marketing-Analytics/assets/142073522/5c6a8e8f-221c-4068-a39b-637233c60d6f">

```
a = 0.0167
t_WR_return,p_WR_return = stats.ttest_ind(df4.loc[df4['day_code']=='W','return_proportion'].values,
                     df4.loc[df4['day_code']=='R','return_proportion'].values,equal_var=False)
t_WH_return,p_WH_return = stats.ttest_ind(df4.loc[df4['day_code']=='W','return_proportion'].values,
                     df4.loc[df4['day_code']=='H','return_proportion'].values,equal_var=False)
t_RH_return,p_RH_return = stats.ttest_ind(df4.loc[df4['day_code']=='R','return_proportion'].values,
                     df4.loc[df4['day_code']=='H','return_proportion'].values,equal_var=False)
p_result_return = [p_WR_return,p_WH_return,p_RH_return]
a_result_return = ["Reject null hypothesis" if i < a else "Accept null hypothesis" for i in p_result]
ttest_result_return = pd.DataFrame({
    'Comparison' : ['Weekend x Regular Day','Weenend x Holiday','Regular Day x Holiday'],
    'p value' : p_result_return,
    'Test Result' : a_result_return
})
ttest_result_return
```

When focusing on those who returned something, there is even a significant differences between weekend and holiday day, indicating that the management should put extra effor on reviewing holiday promotions and products. Some issues might be misleading promotion on holidays or lower-quality products on holiday.

### Testing whether day type impacts the likelihood that a consumer will return something after making a purchase.

The null hypothesis here would be "there is no impact of returning goods based on different day type", so our `expected returns on each day type`would be: `( Total purchases / Total returns) × Total purchases for day type `.


The return rate is 37%, calculating based on the `total return number / total purchase number`.
From the calculation, the expected returns of each day type is approximately:
- Holiday: 206
- Regular Day: 485
- Weekend: 359
```
actual = [len(df4[df4['day_code'] == 'H']),
           len(df4[df4['day_code'] == 'R']),
            len(df4[df4['day_code'] == 'W'])]
chi_df = pd.DataFrame({
    'day type':day_type,
    'Expect': adjusted_expected,
    'Actual': actual
})

chi_df
```

Then, using the actual and run the chi-square test.

```
chi,p_chi = chisquare(f_obs =chi_df.iloc[:,1] ,f_exp = chi_df.iloc[:,2])
print('The chisquare value is: ',chi)
print('p-value is: ',p_chi)
```

The p-value is 0.97 -- far larger than my ahpla threshold (0.05), meaning based on this statistical testing I do not have evidence to reject null hyphothesis.

The conclusion from the chi-square testing is that different day type does not impact the likelihood that a consumer will return something after making a purchase. 

## Project V: Marketing Mix Model

This project examed the marketing performance by extracting cost and revenue from YouTube Ad, Spotify Ad, and Banner Ad to see if investing in all these three channel would create a positive effect for sales.

### 1st Model: taking YouTube, Spotify, and Banner channel into testing

```
X = ad2[['YouTube','Spotify','Banners']]
y = ad2['Sales']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
```

The p-value of the F-statistic is 1.58e-96, which is really low, indicating that there is strong evidence that at least one of the independent variables (YouTube ad spending, Spotify ad spending, or banner ad spending) has a statistically significant impact on the dependent variable (Sales). Besides, the overall regression model is likely a good fit for the data because it explains a significant amount of the variation in the dependent variable.

After looking deeper in this model, the p-value of YouTube and Spotify is 0 -- meaning they have significant impact on Sales. However, Banners get a p-value of 0.86 -- meaning Banners does not have strong impact toward Sales.

<img width="868" alt="image" src="https://github.com/leonlin97/Marketing-Analytics/assets/142073522/2bd18628-ef06-4ed1-85ef-0ad8ceb6eb0e">

### 2nd Model: adding interaction term of `Spotiy` x `YouTube`
```
ad2['Interaction'] = ad2['YouTube'] * ad2['Spotify']
X = ad2[['YouTube','Spotify','Interaction']]
y = ad2['Sales']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
```

The p-value of all variables are 0, meaning YouTube ad spending and Spotify ad spending both have significant and independent effects on Sales. Also, the interaction between YouTube and Spotify ad spending is also highly significant and represents a combined effect beyond their individual contributions.

The r-square with interaction is 0.968 -- higher than that without interaction (0.897). Since r-square indicates how well the model fits the data, it is clear to see the model performs better when considering the interaction between YouTube and Spotify ad spending.

<img width="880" alt="image" src="https://github.com/leonlin97/Marketing-Analytics/assets/142073522/67464f04-8496-4300-b23e-84929d6d6380">

### Using the model to predict sales

With spending 150 on YouTube and 30 on Spotify, the Sales prediction (by using the model with interaction) is: 1,142.98

```
YouTube = 150
Spotify = 30

input_data = pd.DataFrame({'YouTube': [YouTube],
                           'Spotify': [Spotify],
                           'Interaction': [YouTube * Spotify],
                           'const': [1]})

# Use the model to make predictions
predicted_sales = model.predict(input_data)

# Print the predicted sales outcome
print("Predicted Sales:", round(predicted_sales.iloc[0],2))
```




