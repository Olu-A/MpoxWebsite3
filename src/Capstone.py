import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor as RFC
from prophet import Prophet
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df = pd.read_csv('https://raw.githubusercontent.com/owid/monkeypox/main/owid-monkeypox-data.csv')

df.to_csv('Cdata.csv')

# Edouard Mathieu, Fiona Spooner, Saloni Dattani, Hannah Ritchie and Max Roser (2022) - "Mpox (monkeypox)". Published online at OurWorldInData.org. Retrieved from: 'https://ourworldindata.org/monkeypox' [Online Resource]

def import_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    return df

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[["COUNTRY", "DATE", "TOTAL_CONF_CASES", "TOTAL_CONF_DEATHS"]].rename(
        columns={
            "COUNTRY": "location",
            "DATE": "date",
            "TOTAL_CONF_CASES": "total_cases",
            "TOTAL_CONF_DEATHS": "total_deaths",
        }
    )

def harmonize_countries(df: pd.DataFrame) -> pd.DataFrame:
    return geo.harmonize_countries(
        df,
        countries_file=SOURCE_COUNTRY_MAPPING,
        country_col="location",
        warn_on_missing_countries=False,
    )

def clean_date(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df.date).dt.date.astype(str)
    return df

def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date", ascending=False)
    df["total_cases"] = df[["location", "total_cases"]].groupby("location").cummin()
    df["total_deaths"] = df[["location", "total_deaths"]].groupby("location").cummin()
    return df.sort_values(["location", "date"])

def explode_dates(df: pd.DataFrame) -> pd.DataFrame:
    df_range = pd.concat(
        [
            pd.DataFrame(
                {
                    "location": location,
                    "date": pd.date_range(
                        start=df.date.min(), end=df.date.max(), freq="D"
                    ).astype(str),
                }
            )
            for location in df.location.unique()
        ]
    )
    df = pd.merge(
        df, df_range, on=["location", "date"], validate="one_to_one", how="right"
    )
    df["report"] = df.total_cases.notnull() | df.total_deaths.notnull()
    return df

def add_population_and_countries(df: pd.DataFrame) -> pd.DataFrame:
    pop = pd.read_csv(
        SOURCE_POPULATION, usecols=["entity", "population", "iso_code"]
    ).rename(columns={"entity": "location"})
    missing_locs = set(df.location) - set(pop.location)
    if len(missing_locs) > 0:
        raise Exception(f"Missing location(s) in population file: {missing_locs}")
    df = pd.merge(pop, df, how="right", validate="one_to_many", on="location")
    return df

def derive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    def derive_country_metrics(df: pd.DataFrame) -> pd.DataFrame:

        # Add daily values
        df["new_cases"] = df.total_cases.diff()
        df["new_deaths"] = df.total_deaths.diff()

        # Add 7-day averages
        df["new_cases_smoothed"] = (
            df.new_cases.rolling(window=7, min_periods=7, center=False).mean().round(2)
        )
        df["new_deaths_smoothed"] = (
            df.new_deaths.rolling(window=7, min_periods=7, center=False).mean().round(2)
        )

        # Add per-capita metrics
        df = df.assign(
            new_cases_per_million=round(df.new_cases * 1000000 / df.population, 3),
            total_cases_per_million=round(df.total_cases * 1000000 / df.population, 3),
            new_cases_smoothed_per_million=round(
                df.new_cases_smoothed * 1000000 / df.population, 3
            ),
            new_deaths_per_million=round(df.new_deaths * 1000000 / df.population, 3),
            total_deaths_per_million=round(
                df.total_deaths * 1000000 / df.population, 3
            ),
            new_deaths_smoothed_per_million=round(
                df.new_deaths_smoothed * 1000000 / df.population, 3
            ),
        ).drop(columns="population")

        min_reporting_date = df[df.report].date.min()
        max_reporting_date = df[df.report].date.max()
        df = df[(df.date >= min_reporting_date) & (df.date <= max_reporting_date)].drop(
            columns="report"
        )
        return df
    return df.groupby("iso_code").apply(derive_country_metrics)
def filter_dates(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.date >= "2022-05-01"]
#do groupings and feature addtion by country especially lag and difference
revised=df[['location','date','new_cases','iso_code']]
#Rdata = pd.DataFrame.from_dict(revised)
#Rdata.to_csv('Rdata.csv')
#do groupings and feature addtion by country especially lag and difference

import numpy as np
df = pd.pivot(df, index='date', columns='location', values='new_cases')
df=df.fillna(0)
df=df.drop(['World','North America','South America','Europe','Asia','Africa','Oceania'], axis=1)
df

c_df=df['United States'].copy()
c_df= c_df.reset_index(level=0)
c_df

#from lets_plot import *
#ggplot(c_df    ) + geom_line(aes(color="United States", x="date", y="United States"))

df.index = pd.to_datetime(df.index)
df
dff=df.copy()
#Using Pearson Correlation
plt.figure(figsize=(8,8))
cor = df.corr()

#Correlation with output variable
cor_target = abs(cor["United States"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
relevant_features= relevant_features.reset_index(level=0)
relevant_features

#from lets_plot import *
#ggplot(relevant_features) + geom_point(aes(color="location", x="location", y="United States"))

irrelevant_features =cor_target[cor_target<0.1]
irrelevant_features= irrelevant_features.reset_index(level=0)
irrelevant_features

#from lets_plot import *
#ggplot(irrelevant_features) + geom_point(aes(color="location", x="location", y="United States"))

lags = range(0, 15)
lagr = range(14)

df = pd.concat([df.shift(t).add_suffix(f" (t-{t})") for t in lags], axis=1)  # add lags
df

target_lags = []
for t in lags:
    target_lags.append(('United States' f" (t-{t})"))

#include isolation forest for outlier removal
from sklearn.model_selection import train_test_split
#include isolation forest for outlier removal
df = df.loc[df.index >= '2022-05-19']
df = df.dropna(axis=1)

train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)

train
test

from sklearn.metrics import mean_squared_error, mean_absolute_error

fold = 0
preds = []
scores = []

FEATURES = df.drop(target_lags, axis =1)
TARGET = 'United States (t-0)'
#include isolation forest for the removal of outliers
X_train = train.drop(target_lags, axis =1)
y_train = train[TARGET]
X_test = test.drop(target_lags, axis =1)
y_test = test[TARGET]
len(X_train.columns)

reg = RFC(random_state=42).fit(X_train, y_train)
rfe=RFE(reg)
predictions = reg.predict(X_test)

# model evaluation
print('Train accuracy score:',reg.score(X_train,y_train))
print('Test accuracy score:', reg.score(X_test,y_test))

print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))
print('mean_absolute_percentage_error : ', mape(y_test, predictions))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(reg, X_train, y_train, cv=10)
scores
print("%0.2f train accuracy with a standard deviation of %0.2f over folds" % (scores.mean(), scores.std()))

reg.get_params().keys()

param_grid = {"max_features":['sqrt',None,'log2'], "max_depth": [None],
              "min_samples_split": [10],'min_samples_leaf':[1,2,3], 'max_leaf_nodes':[None], 'max_samples':[None]}

model_cv = HalvingGridSearchCV(reg, param_grid, resource='n_estimators',max_resources=400,
                    random_state=42, cv=10).fit(X_train,y_train)

model_cv.best_params_

print (f'CV Train Accuracy - : {model_cv.score(X_train,y_train):.3f}')
print (f'CV Test Accuracy - : {model_cv.score(X_test,y_test):.3f}')
print('mean_squared_error : ', mean_squared_error(y_test, model_cv.predict(X_test)))
print('mean_absolute_error : ', mean_absolute_error(y_test, model_cv.predict(X_test)))
print('mean_absolute_percentage_error : ', mape(y_test, model_cv.predict(X_test)))

# Retrain on all data
#df = create_features(df)

FEATURES = df.drop(target_lags, axis =1)
TARGET = 'United States (t-0)'
#include isolation forest for the removal of outliers
X_all = FEATURES
y_all = df[TARGET]

reg = RFC(random_state=42).fit(X_all, y_all)
y_pred=reg.predict(X_all)
print('Accuracy of the model :',reg.score(X_all,y_all))
print('mean_squared_error : ', mean_squared_error(y_all, y_pred))
print('mean_absolute_error : ', mean_absolute_error(y_all, y_pred))
print('mean_absolute_percentage_error : ', mape(y_all, y_pred))
scores = cross_val_score(reg, X_all, y_all, cv=10)
scores
print("%0.2f Model accuracy with a standard deviation of %0.2f over folds" % (scores.mean(), scores.std()))

param_grid = {"max_features":['sqrt',None,'log2'], "max_depth": [None],
              "min_samples_split": [10],'min_samples_leaf':[1,2,3], 'max_leaf_nodes':[None], 'max_samples':[None]}

model_cv = HalvingGridSearchCV(reg, param_grid, resource='n_estimators',max_resources=400,
                    random_state=42, cv=10).fit(X_all,y_all)

model_cv.best_params_

print (f'Model Accuracy - : {model_cv.score(X_all,y_all):.3f}')
print('mean_squared_error : ', mean_squared_error(y_all,model_cv.predict(X_all)))
print('mean_absolute_error : ', mean_absolute_error(y_all, model_cv.predict(X_all)))
print('mean_absolute_percentage_error : ', mape(y_all, model_cv.predict(X_all)))

importances = pd.DataFrame(data={
    'Attribute': X_all.columns,
    'Importance': reg.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)
importances

#from lets_plot import *
#ggplot(importances) + geom_bar(aes(fill="Attribute", x="Attribute", y="Importance"), stat='identity')

df.index.max()

# #comparing predictions with real values
# future_data = pd.DataFrame(columns=X_test.columns)
# for i in range(14):
#     future_data = future_data.append(df.iloc[-1])
#     future_data.index = [df.index[-1] + i + 1]
#
# future_pred = model.predict(future_data)
#
# # Plot the forecasted data
# plt.plot(future_pred)
#
# plt.show()

df2 = df.copy()
pred = model_cv.predict(X_all)
df2['Pred']= pred
og =df2['United States (t-0)']
df2=df2.join(y_all, how = 'left', lsuffix='left')
df2=df2.reset_index(level=0)
df2

#from lets_plot import *
#ggplot(df2) + geom_line(aes(x="United States (t-0)", y="Pred"))

df2.plot(x='date', y=['Pred', "United States (t-0)"],
        kind="line", figsize=(10, 10))

# #FORCASTING

df2.rename(columns={'date':'ds','Pred':'y'}, inplace=True)

df2['cap'] = 1000
df2['floor'] = 0
model_p = Prophet(changepoint_prior_scale=0.5, growth='logistic', weekly_seasonality=False,)
model_p.add_country_holidays(country_name='US')
model_p.fit(df2)

future_dates = model_p.make_future_dataframe(periods=14, freq='D')
future_dates['cap'] = 1000
future_dates['floor'] = 0

future_dates.head()

forecast = model_p.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()

model_p.plot(forecast, uncertainty=True)

model_p.plot_components(forecast)

forecast

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(14)

forecast
Fdata = pd.DataFrame.from_dict(forecast)
Fdata.to_csv('forecast.csv')
