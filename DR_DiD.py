'''
Wonjin Lee
08-21-2022
-------------------------------------------------------------------------------
DR DiD
Data: Example 13.4 from Wooldridge textbook (p.442)

The effect of workers' compensation on time out of work.
- Treatment group: High-income workers.
- Control group: Low-income workers.
-------------------------------------------------------------------------------
'''

import wooldridge as woo
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

#------------------------------------------------------------------------------
# 0. Load data set
#------------------------------------------------------------------------------
df = woo.dataWoo('injury')
df = df.assign(id = range(1,len(df)+1))
df_ky = df.query("ky==1")


#------------------------------------------------------------------------------
# 1. Unconditional DiD in 2x2 design - TWFE
#------------------------------------------------------------------------------
print(
    smf.ols('ldurat ~ afchnge*highearn', data = df_ky)
    .fit(cov_type="cluster", cov_kwds={"groups":df_ky['id']})
    .summary().tables[1]
    )


#------------------------------------------------------------------------------
# 2. Conditional DiD in 2x2 design - Naive TWFE
#------------------------------------------------------------------------------
## Drop NaN
print([i for i in df_ky.columns if df_ky[i].isnull().any()])
print(df_ky.isnull().sum())
df_ky = df_ky.dropna()

print(
    smf.ols('ldurat ~ afchnge*highearn + male + married + manuf + construc + lage',
        data = df_ky)
    .fit(cov_type="cluster", cov_kwds={"groups":df_ky['id']})
    .summary().tables[1]
    )


#------------------------------------------------------------------------------
# 3. Conditional DiD in 2x2 design - Sant’Anna and Zhao (2020)
#------------------------------------------------------------------------------
'''
Assume that this is the stationary repeated cross-section data.
'''
Y = 'ldurat'
G = 'highearn'
T = 'afchnge'
X = ['male', 'married', 'manuf', 'construc', 'lage']

def bootstrap_parallel(fnc, df, X, G, T, Y, sample_size_bootstrap=1000):
    ATTs = Parallel(n_jobs = 8)(
        delayed(fnc)(df.sample(frac=1, replace=True), X, G, T, Y)
        for i in range(sample_size_bootstrap))
    return ATTs

def fig_ATTs(ATTs):
    sns.histplot(ATTs, kde=False)
    plt.vlines(np.percentile(ATTs, 2.5), 0, 100, color = 'red',linestyles="--")
    plt.vlines(np.percentile(ATTs, 97.5), 0, 100, color = 'red',linestyles="--",
            label="95% Conf. Interval")
    plt.title("ATT Dist by Bootstrap")
    plt.legend()
    plt.show()

## (1) Output regression DiD
#------------------------------------------------------------------------------
def output_reg_DiD(df, X, G, T, Y):
    m0_before = LinearRegression().fit(
        df.query(f"{G}==0 & {T}==0")[X], df.query(f"{G}==0 & {T}==0")[Y]
        ).predict(df[X])
    m0_after = LinearRegression().fit(
        df.query(f"{G}==0 & {T}==1")[X], df.query(f"{G}==0 & {T}==1")[Y]
        ).predict(df[X])
    m0 = (df.assign(m0_after = m0_after).assign(m0_before = m0_before))
    ATT = (
        np.mean(df.query(f"{G}==1 & {T}==1")['ldurat'])
        - np.mean(df.query(f"{G}==1 & {T}==0")['ldurat'])
        - ( np.mean(m0.query(f"{G}==1")['m0_after']) 
        - np.mean(m0.query(f"{G}==1")['m0_before']) ) 
        )
    return ATT

print('ATT = ', output_reg_DiD(df_ky, X, G, T, Y))

# Confidence interval and histogram by bootstrap
np.random.seed(1004)
ATTs = bootstrap_parallel(output_reg_DiD, df_ky, X, G, T, Y)
print('95% Conf. Interval of ATT:', (np.percentile(ATTs, 2.5), np.percentile(ATTs, 97.5)))
fig_ATTs(ATTs)


## (2) IPW DiD (Abadie (2005))
#------------------------------------------------------------------------------
'''
I use the equation (2.3) in the Sant’Anna and Zhao (2020).
'''
def ipw_DiD(df, X, G, T, Y):
    pscore = LogisticRegression(C=1e6, max_iter=1000).fit(
        df[X], df[G]).predict_proba(df[X])[:, 1]
    numer = ((df[G] - (1-df[G])*pscore/(1-pscore))
            *(df[T] - np.mean(df[T]))/(np.mean(df[T])*(1-np.mean(df[T]))))
    ATT = np.mean(numer*df[Y])/np.mean(df[G])
    return ATT

print('ATT = ', ipw_DiD(df_ky, X, G, T, Y))

# Confidence interval and histogram by bootstrap
np.random.seed(1004)
ATTs = bootstrap_parallel(ipw_DiD, df_ky, X, G, T, Y)
print('95% Conf. Interval of ATT:', (np.percentile(ATTs, 2.5), np.percentile(ATTs, 97.5)))
fig_ATTs(ATTs)


## (3) Doubly Robust DiD
#------------------------------------------------------------------------------
'''
I weight using the equation (2.3) in the Sant’Anna and Zhao (2020) for simplicity.
'''
def doubly_robust_DiD(df, X, G, T, Y):
    pscore = LogisticRegression(C=1e6, max_iter=1000).fit(
        df[X], df[G]).predict_proba(df[X])[:, 1]
    m0_before = LinearRegression().fit(
        df.query(f"{G}==0 & {T}==0")[X], df.query(f"{G}==0 & {T}==0")[Y]
        ).predict(df[X])
    m0_after = LinearRegression().fit(
        df.query(f"{G}==0 & {T}==1")[X], df.query(f"{G}==0 & {T}==1")[Y]
        ).predict(df[X])
    numer = ((df[G] - (1-df[G])*pscore/(1-pscore))
        *(df[T] - np.mean(df[T]))/(np.mean(df[T])*(1-np.mean(df[T]))))
    ATT = np.mean(numer*(df[Y] - ( m0_after - m0_before))/np.mean(df[G]))
    return ATT

print('ATT = ', doubly_robust_DiD(df_ky, X, G, T, Y))

# Confidence interval and histogram by bootstrap
np.random.seed(1004)
ATTs = bootstrap_parallel(doubly_robust_DiD, df_ky, X, G, T, Y)
print('95% Conf. Interval of ATT:', (np.percentile(ATTs, 2.5), np.percentile(ATTs, 97.5)))
fig_ATTs(ATTs)
