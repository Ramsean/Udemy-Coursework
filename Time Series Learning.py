# TIME SERIES

#%% Importing libraries

import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARMA, ARIMA
from scipy.stats.distributions import chi2
import statsmodels.graphics.tsaplots as sgt
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
import pylab
import scipy.stats

#%% Loading the data

dfRaw = pd.read_csv('C:/Users/Sean Montgomery/OneDrive - Anne Holland Ventures/Sean Montgomery Unshared Work/Documents/index2018.csv')
dfComp = dfRaw.copy()

#%% Data exploration

# Head of data
dfComp.head()

# Summary stats of data
dfComp.describe()

# Detecting missing values
dfComp.isna()

# Counts of missing values
dfComp.isna().sum()

#%% Ploting Data

# plotting specific columns in df and showing graph
dfComp.spx.plot()
plt.show()

# adding a title & stretching graph
dfComp.spx.plot(figsize=(20,5),title="S&P Prices")

# Same thing but for different columns
dfComp.dax.plot(figsize=(20,5),title="DAX Prices")

# Overwriting the old titles and showing graph
plt.title("S&P vs DAX (new title)")
plt.show()

#%% QQ Plot (normality diagnostics)

scipy.stats.probplot(dfComp.spx, plot = pylab)
pylab.show()

# Might wanna review qq plot diagnostics.. (aprrox normal)

#%% Coverting df into Time Series (Woop)

dfComp.date.describe()

# conveting dates into date time type
dfComp.date = pd.to_datetime(dfComp.date, dayfirst= True)

# setting the date as an index
dfComp.set_index("date", inplace=True)

# Defining the frequency of the data set
dfComp = dfComp.asfreq('b') # the b here sets only on buisness days (excluding sat/sun)
dfComp.head()

#%% Dling with missing data
dfComp.isna().sum()

# Front filling NA values: Assigns the value of the previous period
dfComp.spx = dfComp.spx.fillna(method="ffill")
dfComp.nikkei = dfComp.nikkei.fillna(method="ffill")

# Back filling NA values: Assigns the value of the next period
dfComp.ftse = dfComp.ftse.fillna(method="bfill")

# Assigning the same value: Assign average to all missing values (risky if trend)
dfComp.dax =dfComp.dax.fillna(value = dfComp.dax.mean())


dfComp.isna().sum() # checking to see that missing values are gone


#%% Subsetting the data

# creating new column 'marketValue'
dfComp['marketValue'] =dfComp.spx


#%% ML splitting data

# Avoid shuffling for TS data (duh)
# An 80/20 split will be used here....

size = len(dfComp)
train = int(.8*size) # int makes it a whole number

# training data set (80%)
dfTrain = dfComp.iloc[:train]

# testing set (20%)
dfTest = dfComp[train:]

#%% WHITE NOISE AND YOU

# White noise is just data the doesn't follow a pattern
# White noise has: constant mean & var, and no autocorrelation (relationship of past and present values)

# generating random white noise
wn = np.random.normal(loc=df.marketValue.mean(), scale=df.marketValue.std(), size=len(df)) # loc = mean & scale = std 

# attaching the wn to the df
df['wn'] = wn

# plotting the white noise
df.wn.plot(figsize = (15,5))
plt.title("White Noise", size = 24)
plt.ylim(0,2300)
plt.show()

# plotting the market value prices
df.marketValue.plot(figsize = (15,5))
plt.title("Market Value",size = 24)
plt.ylim(0,2300)
plt.show()

#%% RANDOM WALK

# A random walk is a TS where values tend to persist over time and the spaces inbetween are just white noise

# uploading the random walk csv
rw = pd.read_csv('C:/Users/Sean Montgomery/OneDrive - Anne Holland Ventures/Sean Montgomery Unshared Work/Documents/RandWalk.csv')
rw.date = pd.to_datetime(rw.date,dayfirst=True)
rw.set_index('date', inplace = True)
rw = rw.asfreq('b')

# adding the random walk to the df
df['rw'] = rw

df.rw.plot(figsize=(20,5))
plt.title("Random Walk")
plt.show()

#%% STATIONARITY

# WEAK stationarity: let S1 and S2 be random segments of equal size from the same dataset, then Cov(S1) ~=~ Cov(S2)
# Strict Stationarity: When Cov(S1) = Cov(S2) absolutely.
# for analysis purposes, we use WEAK stationarity as strict is an ideal case.

# copying the data:
df = dfComp.copy()


# For testing stationality, we use the Dicky-Fuller Test (DF Test)
sts.adfuller(df.marketValue)

# OUTPUT:
# (-1.7369847452352438, <--- test statistic, compare it to 1-10%, if > we FTR and data is NOT stationary
#  0.4121645696770621,  <--- P-val (% of of observing a test statistic as extream as our critical value
#  18,
#  5002,
#  {'1%': -3.431658008603046,
#   '5%': -2.862117998412982,
#   '10%': -2.567077669247375},
#  39904.880607487445)

# DF test for white noise
sts.adfuller(df.wn)

# DF test for random walk
sts.adfuller(df.rw)

#%% Seasonality

# Naive Decomposition:
# additive or multiplicative

# additive
s_dec_additive = seasonal_decompose(dfComp.marketValue, model = 'additve')
s_dec_additive.plot()
plt.show()

# multiplicative
s_dec_multiplicative = seasonal_decompose(df.marketValue, model = 'multiplicative')
s_dec_multiplicative.plot()
plt.show()

#%% AUTOCORRELATION & PARTIAL AUTOCORRELATION

# The autocorr function (ACF) Shows the relationship between past values and present values.

sgt.plot_acf(df.marketValue, lags = 40, zero = False)
plt.title("ACF S&P")
plt.show()

# ^ In this plot, if autocorr values are above the blue bubble, then we say autocorrelation is significance!

# ACF of WN
sgt.plot_acf(df.wn, lags = 40, zero = False)
plt.title("ACF WN")
plt.show()

# PARTIAL AUTOCORR (PACF)

sgt.plot_pacf(df.marketValue, lags = 40, zero = False, method = ('ols'))
plt.title("PACF S&P")
plt.show()

sgt.plot_pacf(df.wn, lags = 40, zero = False, method = ('ols'))
plt.title("PACF WN")
plt.show()

#%% MODEL SELECTION

# Parsimonius (simple) model vs complex model

# Use AIC or BIC for model selection, and a higher log likelihood

#####################################################################################################################################

### Autoregressive model (AR) ###

## A linear model where current period values are a sum of past outcomes multiplied by a numeric factor

## The Model: x_t = C + phi * x_t-1 + epsilon
# C is a constant
# -1 < phi < 1

## the AR model with number of lags (n) is denoted as: "AR (n)"
# where AR (n): x_t = C + phi_1 * x_t-1 + phi_2 * x_t-2 + ... + phi_n * x_t-n + epsilon
# Phi's have to be statistically different from zero to be included!

#####################################################################################################################################

df1 = pd.DataFrame(dfRaw.ftse)
df1.columns = ["marketValue"]
df = df1.copy()


### Fitting the AR(1) ###
model_AR1 = ARMA(df.marketValue, order=(1,0)) 
#                                  ^ order=(1,0), the 1 here means how many past values we want to use, and 0 is the number of residual values we use

# use .fit() to store the model results
results_AR1 = model_AR1.fit()

# check the results of model
results_AR1.summary()

# check to see if and where 0 is included in confidence intervals, and check AIC, use LogLikilihood to pick model from comparison


### Fitting AR(2)
model_AR2 = ARMA(df.marketValue, order=(2,0)) 
results_AR2 = model_AR2.fit()
results_AR2.summary()


### Fitting AR(3)
model_AR3 = ARMA(df.marketValue, order=(3,0)) 
results_AR3 = model_AR3.fit()
results_AR3.summary()


### Fitting AR(4)
model_AR4 = ARMA(df.marketValue, order=(4,0)) 
results_AR4 = model_AR4.fit()
results_AR4.summary()

#...

### Fitting AR(7)
model_AR7 = ARMA(df.marketValue, order=(7,0)) 
results_AR7 = model_AR7.fit()
results_AR7.summary()


### Testing LL Ratio using a function:

def LLRTest(r1,r2,DF=1):
    L1 = r1.llf
    L2 = r2.llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR,DF).round(3)
    return p


LLRTest(results_AR2,results_AR3,DF=2)
# result is significant ( < CV ) so we opt for more complex model

LLRTest(results_AR3,results_AR4,DF=3)
# result is  significant. opt for complex model

# A cleaner output for some of the LLR values:
print('LLR p-value: ' + str(LLRTest(results_AR3,results_AR4,DF=3)))

# Continue to compare models increasing the Lag until:
# 1.) Non-significant p-value for the LLR test
# 2.) Non-significant p-value for the highest lag coeffiecnts

#   As example, the AR(8) model will give us a non-significant LLR test AND Non significant coeff for the 8th period.
#   Therefore, AR(7) is our optimal model.


#%% AR MODELS AND NONSTATIONARY DATA

# Let's say we have some non stationary data...
# AR models don't work when data is non-stationary, so we have to use a different approtch
# Instead, we will build an AR model on the RETURNS

df['returns'] = df.marketValue.pct_change(1).mul(100)

# since the first observation in "returns" is nan, we have to get rid of it:
df = df.iloc[1:]

# DF test
sts.adfuller(df.returns) # shows that data is now stationary

#%% AR MODEL ON RETURNS

returnModel_AR1 = ARMA(df.returns,order=(1,0))
returnResults_AR1 = returnModel_AR1.fit()
returnResults_AR1.summary()

# for sake brevity, I will change the name to rm and rr

# AR(1)
rm1 = ARMA(df.returns,order=(1,0))
rr1 = rm1.fit()
rr1.summary()

#AR(2)
rm2 = ARMA(df.returns,order=(2,0))
rr2 = rm2.fit()
rr2.summary()

#AR(3)
rm3 = ARMA(df.returns,order=(3,0))
rr3 = rm3.fit()
rr3.summary()

#AR(4)
rm4 = ARMA(df.returns,order=(4,0))
rr4 = rm4.fit()
rr4.summary()

#AR(4)
rm4 = ARMA(df.returns,order=(4,0))
rr4 = rm4.fit()
rr4.summary()
#....


LLRTest(rr1,rr2)
LLRTest(rr2,rr3)
LLRTest(rr3,rr4)
#....

#%% NORMALIZING DATA IN TS

# consider two time series; X & Y
# compare x_t to period x_1 & compare y_t to period y_1

# This makes comparing time series easier, to see what model is more reliable


# Normalizing market value
benchmark = df.marketValue.iloc[0]
df['norm'] = df.marketValue.div(benchmark).mul(100)

sts.adfuller(df.norm) # non-stationary data, no AR :(

# Normalizing returns
benchmark = df.returns.iloc[0]
df['norm'] = df.returns.div(benchmark).mul(100)

sts.adfuller(df.norm) # Stationary! perform AR :)


# **** Usually, we normalize returms! AND Normalizing HAS NO IMPACT ON STATIONARITY OR MODEL SELECTION **** #


#%% TESTING RESIDUALS FOR IID ~ N(0,constant)

df['res_price'] = results_AR7.resid

df.res_price.mean() # ~ 0

sts.adfuller(df.res_price) # stantionary!

sgt.plot_acf(df.res_price, zero =False, lags = 40) # residuals are following a cone shape! better model may exist?

#%% Accounting for unpredictable shocks (avoiding ML lag)

# Moving Averages (MA) gives the model a better chance to account for 'sudden shocks'

### The MA Model:
# r_t = c  +  theta_1 * epsilon_t-1  +  epsilon_t

# The key difference between AR and MA model is the MA model relies on the residual.
# AND the absolute value of the coeffiecent is less than 1

# MA model also uses the ACF since it should include the most recent value.

sgt.plot_acf(df.returns[1:],zero=False,lags=40)
plt.title("ACF for Returns", size=24)
plt.show()

# Simple MA model
maReturns1 = ARMA(df.returns[1:], order = (0,1))
maResults1 = maReturns1.fit()
maResults1.summary()

#%% HIGHER MA MODELS

# MA(2)
maReturns2 = ARMA(df.returns[1:], order = (0,2))
maResults2 = maReturns2.fit()
maResults2.summary()

# MA(3)
maReturns3 = ARMA(df.returns[1:], order = (0,3))
maResults3 = maReturns3.fit()
maResults3.summary()
print(maResults3.summary())
print(LLRTest(maResults2,maResults3))

#...

# MA(6)
maReturns6 = ARMA(df.returns[1:], order = (0,6))
maResults6 = maReturns6.fit()
maResults6.summary()

# MA(7)
maReturns7 = ARMA(df.returns[1:], order = (0,7))
maResults7 = maReturns7.fit()
maResults7.summary()

# MA(8)
maReturns8 = ARMA(df.returns[1:], order = (0,8))
maResults8 = maReturns8.fit()
maResults8.summary()

# Compare MA(8) to the ACF graph and notice the 1st and 7th points are insignificant.
# MA(8) > MA(6) > MA(7)

# Comparing MA(6) to MA(8)

LLRTest(maResults6,maResults8, DF=2) # DF = 2 because the difference is 2 lags

#%% RESIDUALS OF RETURNS MODEL

# Extracting the residuals of the returns model
maResults8_residuals = maResults8.resid[1:]

maResults8_residuals.mean() # near zero
maResults8_residuals.var() # near 1

# visualization of residuals
maResults8_residuals.plot() # random

# checking stationarity (wn resuiduals = good)
sts.adfuller(maResults8_residuals[2:]) # significant, so yes stationary

# checking acf
sgt.plot_acf(maResults8_residuals[2:],zero=False,lags=40)
plt.title("ACF of residuals MA(8)")
plt.show()

# Notice that early lags are not significant (good)

#%% Normalized Returns

benchReturns = df.returns.iloc[1]
df["normReturns"] = df.returns.div(benchReturns).mul(100)

sgt.plot_acf(df.normReturns[1:] , zero=False,lags=40) # same as the previous non-normalized model

# Again, normalizing the data does not affect the model selection


#%% WHEN TO PICK AR OR MA

sgt.plot_acf(df.marketValue, lags = 40, zero = False)
plt.title("ACF FTSE")
plt.show()
# When ACF coefficents don't "die off" and all the lags seem significant (as seen above) we should use an AR model instead of an MA model 

#%% THE ARMA MODEL

# Takes into account past values (AR) and past errors (MA)
# great model because it takes int account the actual values and how far off we were, allows to calibrate expectations


### The ARMA model:
# r_t = c  +  phi_1 * r_t-1  +  theta_1 * epsilon_t-1  +  epsilon_t


# Fitting an ARMA(1,1) model to the returns data
arma11Model = ARMA(df.returns[1:],order=(1,1))
arma11Results = arma11Model.fit()
arma11Results.summary()

# ^ above we see the coeff for AR is postive. This means there is a positive tendency between past and present values.

# before moving on to higher ARMA models, compare LLRTest to AR(1) and MA(1)

LLRTest(maResults1,arma11Results) # small p-value
LLRTest(rr1,arma11Results) # small p-value

# ^ arma model is better!

### When selecting an ARMA model, we start with a complicated model. How do we do that?
# start with your optimal AR and MA models, then "Half" them... example:
#previously we stated that the AR(6) and MA(8) model was best, therefore, we should take half of each and fit an:
# ARMA(3,4) (or an ARMA(3,3) also works...)

arma33Model = ARMA(df.returns[1:],order=(3,3))
arma33Results = arma33Model.fit()
arma33Results.summary()

LLRTest(arma11Results,arma33Results,DF=4) # difference of 2 for AR and 2 for MA. hence DF = 4

# ^ The LLRTest here tells us that the ARMA(3,3) is better... but where do we go from here?

# Since not all coeffiecents are significant, our "optimal model" is somewhere bewteen (1,1) & (3,3)..

# ARMA(3,2)
arma32Model = ARMA(df.returns[1:],order=(3,2))
arma32Results = arma32Model.fit()
arma32Results.summary()

# ^ constant is zero (good) and coefficents are all significant!
# Also, notice that the trend is decreasing for the AR and MA coeffiecents, therfore the further back we gp, the less relevant values and errors become

# ARMA(2,3)
arma23Model = ARMA(df.returns[1:],order=(2,3))
arma23Results = arma23Model.fit()
arma23Results.summary()

# some coefficents are insignificant, therefore we should avoid using this model

# ARMA(3,1)
arma31Model = ARMA(df.returns[1:],order=(3,1))
arma31Results = arma31Model.fit()
arma31Results.summary()

# ^ also a good model, let's use LLRTest to compre (3,2) & (3,1)
# ** use LLRTest when comparing two models that have all significant coefficents ** #
LLRTest(arma31Results,arma32Results)

# ARMA(1,3)
arma13Model = ARMA(df.returns[1:],order=(1,3))
arma13Results = arma13Model.fit()
arma13Results.summary()

# ^ all coeff significant, but can't compare to ARMA(3,1) because ARMA(1,3) is not nested in ARMA(3,1)
# When we can't compare, we manually check the Log Likelihoods and AIC's

print(arma13Results.llf,arma31Results.llf)
print(arma13Results.aic,arma31Results.aic)
# Since ARMA(1,3) has lower AIC and higher LL, ARMA(1,3) is better

#%% PLOT RESIDUALS FOR ARMA MODEL

# Selecting the arma 3,2 residuals and putting them in df
df['arma32Residuals'] = arma32Results.resid[1:]

# Plotting residuals
df.arma32Residuals.plot(figsize=(20,5))

sgt.plot_acf(df.arma32Residuals[2:], zero = False, lags = 40)

# ^ notice that the 4th,5th, and 6th lag are significant, this means we should continue to explore ARMA(4:6,4:6)
# (not gonna do that right now)

# just be sure to examine the resiudals to pick your model.

#%% ARMA MODELS AND NON STATIONARY DATA

# plot acf
sgt.plot_acf(df.marketValue,unbiased=True,zero=False,lags=40)
# ^ all significant

# plot pacf
sgt.plot_pacf(df.marketValue,zero=False,lags=40,method=('ols'))
# ^ first 6 are significant


# ARMA(1,1)
arma11Model = ARMA(df.marketValue,order=(1,1))
arma11Results = arma11Model.fit()
arma11Results.summary()

sgt.plot_acf(arma11Results.resid, zero=False,lags=40)
# ^ up to 6 are significant, let's start there

# we get an error & long story short, ARMA the way we previously used it DOES NOT WORK ON NON STATIONARY DATA

#%% ADAPTING ARMA TO NON STATIONARY DATA

# USE THE ARIMA MODEL
# the I stands for integration and it dictated the number of times we have to integrate the TS to ensure stationarity

# ARIMA(p,d,q) -> p = AR , d = Integration , q = MA 
# EX: ARIMA(p,0,q) = ARMA(p,q) 

# ARIMA(1,1,1)
arima111Model = ARIMA(df.marketValue,order=(1,1,1))
arima111Results = arima111Model.fit()
arima111Results.summary()

df['arima11Residuals'] = arima111Results.resid

sgt.plot_acf(df.arima11Residuals[1:], zero= False, lags = 40)
# ^ since up to the 4th lag is significant, so at most ARIMA(4,1,4) is significant.

# ARIMA(3,1,2)
arima312Model = ARIMA(df.marketValue, order=(3,1,2))
arima312Results = arima312Model.fit(start_ar_lags=5) # the argument here must be > the AR number to provide starting lags

# ARIMA(1,1,2)
arima112Model = ARIMA(df.marketValue, order=(1,1,2))
arima112Results = arima112Model.fit()

# ARIMA(1,1,3)
arima113Model = ARIMA(df.marketValue, order=(1,1,3))
arima113Results = arima113Model.fit()

# ARIMA(2,1,1)
arima211Model = ARIMA(df.marketValue, order=(2,1,1))
arima211Results = arima211Model.fit()

# ARIMA(3,1,1)
arima311Model = ARIMA(df.marketValue, order=(3,1,1))
arima311Results = arima311Model.fit()

print("ARIMA(1,1,1) \t ",arima111Results.llf,arima111Results.aic)
print("ARIMA(1,1,2) \t ",arima112Results.llf,arima112Results.aic)
print("ARIMA(1,1,3) \t ",arima113Results.llf,arima113Results.aic)
print("ARIMA(1,1,1) \t ",arima211Results.llf,arima211Results.aic)
print("ARIMA(1,1,1) \t ",arima311Results.llf,arima311Results.aic)

# ARIMA(1,1,3) is the top contender
sgt.plot_acf(arima113Results.resid,zero=False,lags=40)
# ^ up to 6 are significant, so need to check up to ARIMA(6,1,6)...

# Based on the same processes from before, ARIMA(5,1,3) was best...

#%% HIGHER LVLS OF INTEGRATION

df['deltaPrices'] = df.marketValue.diff(1)

# Checking to make sure ARMA(1,1) = ARIMA(1,0,1) with new var
arma11Model = ARMA(df.deltaPrices[1:], order = (1,1))
arma11Results = arma11Model.fit()
arma11Results.summary()

arima11Model = ARIMA(df.deltaPrices[1:], order = (1,0,1))
arima11Results = arima11Model.fit()
arima11Results.summary()

# stationarity DF test for delta prices
sts.adfuller(df.deltaPrices[1:])

# Now how do we fit this stationary data? (as shown above)

# If the data is stationary, we should use ARMA because using integration is computationally heavy, unessecary, and harder to interpret

#%% ARIMAX MODEL

# ARIMAX MODEL:
# deltaP_t = c + beta*X + phi_1*deltaP_t-1 + theta_1*epsilon_t-1 + epsilon_t

# the beta and the X are new here compared to the ARIMA Model..
# X is any variable(s) and beta are the coefficent(s)..
# We can use any other type of variable (cont,categorical,booliean) so long as we have an observation for every day of the week
# Furthermore, the exogenous variable should ALSO be stationary, just like the endogenous variable

# these additinal variables are called "exogenous" variables

# when using 'exog' variables, the exog = array_type

# EX: using S&P prices:
df['spx'] = dfRaw.spx

arimax111Model = ARIMA(df.marketValue, order=(1,1,1), exog = df.spx)
arimax111Results = arimax111Model.fit()
arimax111Results.summary()

# ^ S&p prices are not significant, but still, this is how you can add other variables to predict in TS!

#%% SEASONALITY MODELS

# SARMA , SARIMA , SARIMAX

# consider the vegas show, it only occurs during certian parts of the year... so in order to account for this, we ONLY select those parts of the year. NOT all the other months.

# only consider the sections of time that are "active"

# Consider the following: SARIMAX (p,d,q)(P,D,Q,s)
# P = seasonal auto regressive order
# D = seasonal integration
# Q = seasonal moving average order
# s = Length of cycle or, the number of periods needed to pass before the tendency reappears
#     ex: if s = 1, then this means NO SEASONALITY


modelSARIMAX = SARIMAX(df.marketValue, exog = df.spx, order=(1,0,1), seasonal_order = (2,0,1,5))
resultsSARIMAX = modelSARIMAX.fit()
resultsSARIMAX.summary()

#%% Volatility / Stability

# Volatility - the magnitude of the resiudals

# Volatility Clustering: When low follows low values or high follows high values

# We measure Volatility by squaring the values - it penalizes high differences.

# ARCH - Autoregressive Conditional Hederoschedastic model

#%% ARCH

# the ARCH(q) model: where q = the number of previous values we include in the model

# Creating the squared returns:
df['returnsSQ'] = np.square(df.returns)

# Plotting the reuturns vs the squared returns:
plt.subplot(211)
plt.plot(df.returns)
plt.title('Returns')
plt.subplot(212)
plt.plot(df.returnsSQ)
plt.title('Volatility')
plt.show()

# PACF of Returns and Squared Returns:
sgt.plot_pacf(df.returns[1:], lags = 40, alpha = 0.05, zero=False, method= ('ols'))
plt.title('PACF Returns')
sgt.plot_pacf(df.returnsSQ[1:], lags = 40, alpha = 0.05, zero=False, method= ('ols'))
plt.title('PACF Squared Returns')
#  ^  high significance in PACF for squared returns shows us that Volatility clustering is a thing! high follows high or there exists short term trends in variance (clustering)
# this tells us that using an ARCH model!
# * If TS data resembes WN, & squared version suggests using AR(N), then we choose ARCH(N) model * #

# Simple ARCH(1) model
modelARCH1 = arch_model(df.returns[1:], mean = "Constant", vol = "ARCH", p = 1) # p is the order of the model ('N' as above)
resultsARCH1 = modelARCH1.fit()
resultsARCH1.summary()

# 6 iterations means this model is computationally efficent
# Rsquared values are very small
# High LL compared to previous models (GOOD)
# mu coeff is significant in the mean model
# all significant coeff in volatility model

# more complex arch model: ARCH(2)
modelARCH2 = arch_model(df.returns[1:], mean = "Constant", vol = "ARCH", p = 2) # p is the order of the model ('N' as above)
resultsARCH2 = modelARCH2.fit()
resultsARCH2.summary()

# ARCH(3)
modelARCH3 = arch_model(df.returns[1:], mean = "Constant", vol = "ARCH", p = 3) # p is the order of the model ('N' as above)
resultsARCH3 = modelARCH3.fit()
resultsARCH3.summary()

#...

# ARCH(13)
modelARCH13 = arch_model(df.returns[1:], mean = "Constant", vol = "ARCH", p = 13) # p is the order of the model ('N' as above)
resultsARCH13 = modelARCH13.fit()
resultsARCH13.summary()

# ^ 10th alpha is non-sig so we opt for the ARCH(12) as our best model

#%% GARCH MMODEL

# GARCH ~ ARCH

# GARCH(p,q) where p = ARCH order (past squared residuals) & q = GARCH order (past squared variences)

# Garch(1,1)
modelGARCH11 = arch_model(df.returns[1:],mean="Constant", vol = "GARCH", p = 1, q = 1)
resultsGARCH11 = modelGARCH11.fit()
resultsGARCH11.summary()

# ^ Better model for measuring volatility, even more so than the ARCH(12) model

# NOTE: Including higehr order GARCH models are proven to NEVER out-perform the GARCH(1,1)

#%% AUTOMATED MODEL SELECTION (OH THANK GOD!!!!!!!!)

# PROS: FAST OMG, Removes ambiguity, Reduces human error in interpretation
# CONS: Blind method, never get to compare model, topic expertise may prefer simpler model

# Go look....


#%% Forecasting

# There are two parts to forecasting: FInd the pattern, predict the future...

ARmodel = ARIMA(dfTrain.ftse, order=(1,0,0))
ARresults = ARmodel.fit()
dfTrain.tail()
start_date = '2013-04-08'
end_date = '2015-01-01'

df_pred = ARresults.predict(start = start_date, end = end_date)

df_pred[start_date:end_date].plot(figsize = (20,5), color = 'red')
dfTest.ftse[start_date:end_date].plot(figsize = (20,5), color = 'blue')
plt.plot()

# ^ no good, so we will use returns because we stationary data

##### Using returns
### Computing returns:
dfTrain['returns'] = dfTrain.ftse.pct_change(1).mul(100)

ARmodel = ARIMA(dfTrain.returns[1:], order=(1,0,0))
ARresults = ARmodel.fit()
dfTrain.tail()

df_pred = ARresults.predict(start = start_date, end = end_date)


df_pred[start_date:end_date].plot(figsize = (20,5), color = 'red')



# dfTest.ftse[start_date:end_date].plot(figsize = (20,5), color = 'blue')
plt.plot()


(5/6)








































