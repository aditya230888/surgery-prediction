import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
df = pd.read_excel('F:\project\Vanderbilt-University-Medical-Center-Elective-Surgery-Schedule-main\surgery_shedule_prediction\Vanderbilt Univ Case Dataset.xlsx')
df.groupby('DOW')['Actual'].mean(), df.groupby('DOW')['Actual'].std()
# Validating for T - 1
df.groupby('DOW')['T - 1'].mean(), df.groupby('DOW')['T - 1'].std()
df['DOW'].value_counts()
# Trend of Number of Surgeries performed by Day of Week
df2 = df[['DOW', 'Actual']]
ax = sns.barplot(x='DOW',y='Actual', data=df, palette='coolwarm')
sns.boxplot(x='DOW',y='Actual', data=df, palette='coolwarm')
test=df.loc[1:242, 'T - 28':'T - 1']
mean = test.mean()
maximum = test.max()
std = test.std()
a = pd.DataFrame(mean)
b = pd.DataFrame(maximum)
c = pd.DataFrame(std)
new = pd.concat([a,b,c], axis = 1)
new.columns = ['Mean', 'Maximum', 'Standard Deviation']

import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('Actual ~ DOW', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table

from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(endog=df['Actual'], groups=df['DOW'], alpha=0.05)
print(tukey)


df1 = df.drop(columns = ['SurgDate', 'DOW'])
from scipy import stats
df1 = df1[(np.abs(stats.zscore(df1)) < 3).all(axis=1)]

x = df[['T - 3', 'T - 2', 'T - 1']]
y = df[['Actual']]

from sklearn import linear_model
import statsmodels.api as sm
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print(print_model)

from sklearn.linear_model import LinearRegression
regressorObject=LinearRegression()
regressorObject.fit(x,y)

predict = df1.loc[2:242, 'T - 3':'T - 1']
y_pred_test_data=regressorObject.predict(predict)

Predicted = pd.DataFrame(y_pred_test_data, columns = ['Predicted'])

result = pd.concat([df, Predicted], ignore_index=True, axis=1)
result.columns = ['SurgDate', 'DOW', 'T - 28', 'T - 21', 'T - 14', 'T - 13', 'T - 12', 'T - 11',' T - 10', 'T - 9',
                  'T - 8', 'T - 7', 'T - 6', 'T - 5', 'T - 4', 'T - 3', 'T - 2', 'T - 1', 'Actual', 'Predicted']
result['Predicted'] = round(result['Predicted'])

final = result.dropna()
final.tail(10)

from sklearn.metrics import mean_squared_error
from math import sqrt
y_actual = final['Actual']
y_predicted = final['Predicted']
rms = sqrt(mean_squared_error(y_actual, y_predicted))
rms

y = final['Actual']
yhat = final['Predicted']
d = y - yhat
mse_f = np.mean(d**2)
mae_f = np.mean(abs(d))
rmse_f = np.sqrt(mse_f)
r2_f = 1-(sum(d**2)/sum((y-np.mean(y))**2))

print("Results by manual calculation:")
print("MAE:",mae_f)
print("MSE:", mse_f)
print("RMSE:", rmse_f)
print("R-Squared:", r2_f)

x = df1[['T - 28', 'T - 21', 'T - 14', 'T - 13', 'T - 12', 'T - 11', 'T - 10', 'T - 9', 'T - 8', 'T - 7', 
         'T - 6', 'T - 5', 'T - 4', 'T - 3', 'T - 2', 'T - 1']]
y = df1[['Actual']]

from sklearn import linear_model
import statsmodels.api as sm
regr = linear_model.LinearRegression()
regr.fit(x, y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


model = sm.OLS(y, x).fit()
predictions = model.predict(x) 
 
print_model = model.summary()
print(print_model)

from sklearn.linear_model import LinearRegression
regressorObject=LinearRegression()
regressorObject.fit(x,y)


predict = df1.loc[2:242, 'T - 28':'T - 4']
y_pred_test_data=regressorObject.predict(predict)

Predicted = pd.DataFrame(y_pred_test_data, columns = ['Predicted'])

result2 = pd.concat([df, Predicted], ignore_index=True, axis=1)
result2.columns = ['SurgDate', 'DOW', 'T - 28', 'T - 21', 'T - 14', 'T - 13', 'T - 12', 'T - 11',' T - 10', 'T - 9',
                  'T - 8', 'T - 7', 'T - 6', 'T - 5', 'T - 4', 'T - 3', 'T - 2', 'T - 1', 'Actual', 'Predicted']
result2['Predicted'] = round(result2['Predicted'])

final2 = result2.dropna()

round(result['Actual'].mean()), round(result2['Predicted'].mean())

from sklearn.metrics import mean_squared_error
from math import sqrt
y_actual = final2['Actual']
y_predicted = final2['Predicted']
rms = sqrt(mean_squared_error(y_actual, y_predicted))
rms

y = final2['Actual']
yhat = final2['Predicted']
d = y - yhat
mse_f = np.mean(d**2)
mae_f = np.mean(abs(d))
rmse_f = np.sqrt(mse_f)
r2_f = 1-(sum(d**2)/sum((y-np.mean(y))**2))

print("Results by manual calculation:")
print("MAE:",mae_f)
print("MSE:", mse_f)
print("RMSE:", rmse_f)
print("R-Squared:", r2_f)

x = df1[['T - 28', 'T - 21', 'T - 14', 'T - 13', 'T - 12', 'T - 11', 'T - 10', 'T - 9', 'T - 8']]
y = df1[['Actual']]

#Fitting Simple Linear regression data model
from sklearn.linear_model import LinearRegression
regressorObject=LinearRegression()
regressorObject.fit(x,y)

#predict number of surgeries for the data set
predict = df1.loc[2:242, 'T - 28':'T - 8']
y_pred_test_data=regressorObject.predict(predict)

Predicted = pd.DataFrame(y_pred_test_data, columns = ['Predicted'])

result3 = pd.concat([df, Predicted], ignore_index=True, axis=1)
result3.columns = ['SurgDate', 'DOW', 'T - 28', 'T - 21', 'T - 14', 'T - 13', 'T - 12', 'T - 11',' T - 10', 'T - 9',
                  'T - 8', 'T - 7', 'T - 6', 'T - 5', 'T - 4', 'T - 3', 'T - 2', 'T - 1', 'Actual', 'Predicted']
result3['Predicted'] = round(result3['Predicted'])

final3 = result3.dropna()

round(result['Actual'].mean()), round(final3['Predicted'].mean())

y = final3['Actual']
yhat = final3['Predicted']
d = y - yhat
mse_f = np.mean(d**2)
mae_f = np.mean(abs(d))
rmse_f = np.sqrt(mse_f)
r2_f = 1-(sum(d**2)/sum((y-np.mean(y))**2))

print("Results by manual calculation:")
print("MAE:",mae_f)
print("MSE:", mse_f)
print("RMSE:", rmse_f)
print("R-Squared:", r2_f)

plt.figure(figsize=[15,8])
plt.grid(True)
plt.plot(result['Actual'],label='Actual')
plt.plot(result['Predicted'],label='Predicted')
# plt.plot(df['SMA_4'],label='SMA 4 Months')
plt.legend(loc=2)