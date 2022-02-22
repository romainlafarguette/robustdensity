# -*- coding: utf-8 -*-
"""
Class to Estimate a Robust Normal Skewed Density 
Romain Lafarguette, https://romainlafarguette.github.io/
Time-stamp: "2022-01-24 00:47:49 RLafarguette"
"""
###############################################################################
#%% Modules
###############################################################################
import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns

# Functional imports
from scipy.stats import skewnorm
from sklearn.linear_model import TheilSenRegressor, LogisticRegression

###############################################################################
#%% Ancillary function: parametrize a Skewnormal via mode and variance
###############################################################################
def skn_params(mean, variance, cdf_target):
    """ 
    Retrieve the parameters from a Skewnormal from: 
    the mean, variance and CDF of the mode
    The CDF of the mode is evaluated via Logisitic Firth regression

    """

    # Shape derives from the cdf evaluated at the location (balance of risks)
    # The Owen's T function T(0, alpha) = (2/Pi)*arctan(alpha)
    shape = np.tan(np.pi*(0.5-cdf_target)) # Evaluate the Skewnorm CDF at 0

    # Find the scale from shape
    delta = shape/(np.sqrt(1 + shape**2))
    scale_factor = (1-(2*(delta**2)/np.pi))
    scale = np.sqrt(variance/scale_factor) # Scale is always positive

    # Location     
    location = mean - scale*delta*np.sqrt(2/np.pi)
    
    res_d = {'shape': shape, 'location': location, 'scale': scale}

    return(res_d)

def z_scale(x):
    """ Z-score """
    z = (x-np.mean(x))/np.std(x, ddof=1)
    return(z)

def z_rescale(x, m, std):
    """ From Z-score to the original value """
    original = float(m + x*std)
    return(original)

###############################################################################
#%% Robust Density Fit Class
###############################################################################
class RobustDensity(object):

    """ 
    Robust Density Estimation 

    Inputs
    ------
    X: np.array
        Regressors

    Output
    ------
    A robust fit class object

    Usage:
    dens_fit = RobustDensity().fit(X, y)

    """
    __description = "Robust Density Fit using Theil-Sen and Firth Regressions"
    __author = "Romain Lafarguette, https://romainlafarguette.github.io/"
    __date = "August 2021"
    
    # Class initializer
    def __init__(self, depvar, reg_l, data):

        # Data treatment
        # To be sure, remove intercept from the data and add it later
        for col in data.columns:
            if len(data[col].value_counts())==1:
                data = data.drop(col, axis=1).copy()
                reg_l = [x for x in reg_l if x != col]
                print(f'{col} constant, removed from frame, '
                      'model intercept automatically fitted')
        data['intercept'] = 1 # To make sure the intercept is in the dataframe

        # Attributes
        self.depvar = depvar
        self.reg_l = ['intercept'] + reg_l # Make sure intercept is included
        self.all_vars_l = [self.depvar] + self.reg_l
        self.data = data[self.all_vars_l].dropna().copy()
        
        # Estimate and save the mean and std of each variable
        # Will help to scale back after
        self.mean_d = {var:np.mean(self.data[var]) for var in self.all_vars_l}
        self.std_d = {var:np.std(self.data[var]) for var in self.all_vars_l}

        # Normalize all the variables
        self.ndata = pd.DataFrame(index=self.data.index) # Normalized frame
        
        # Demean and center
        for var in [x for x in self.all_vars_l if x != 'intercept']: 
            self.ndata[var] = (self.data[var]-self.mean_d[var])/self.std_d[var]

        self.ndata['intercept'] = 1 # Because of Zscore, intercept=0
            
        self.X = self.ndata[self.reg_l].copy() # Normalized
        self.y = self.ndata[self.depvar].copy() # Normalized
        
    # Class-methods (methods which return a class defined below)    
    def fit(self, random_state=42, model='TS', threshold=0.1, input_var=None):
                
        # Note that the 'self' arg below is the parent class
        return(RobustFit(self, random_state, model, threshold, input_var))

class RobustFit(object):
    # Initialization and inheritance
    def __init__(self, RobustDensity, random_state, model, threshold,
                 input_var):        
        self.__dict__.update(RobustDensity.__dict__) # Import from parent
        self.random_state = random_state
        self.model = model
        self.threshold = threshold
        
        # Model fits
        self.ts_fit = self.__ts_fit() 
        self.ols_fit = self.__ols_fit() 
        self.linear_coeff = self.__coeff_frame()
        # Variance
        self.ts_var = self.__ts_variance()
        self.ols_var = self.__ols_variance()
        if model=='TS':
            self.fit_var = self.ts_var
        elif model=='OLS':
            self.fit_var = self.ols_var
        else: 
            raise ValueError('Model values should be TS or OLS')

        # Possibility to input the variance directly
        if input_var:
            self.fit_var = input_var        
                           
    # Fit the linear models
    def __ts_fit(self):
        """ Theil-Sen Fit """
        tfit = TheilSenRegressor(
            random_state=self.random_state).fit(self.X, self.y)
        return(tfit)

    def __ols_fit(self):
        """ OLS Fit """
        fit = sm.OLS(self.y, self.X).fit(cov_type='HC1')
        print(fit.summary())
        return(fit)

    def __ts_variance(self):
        """ Estimate the variance based on the Theil-Sen fit residuals """
        ts_y_fitted = self.ts_fit.predict(self.X)
        ts_res = self.y - ts_y_fitted
        ts_var = np.var(ts_res) # Variance of the residuals
        return(ts_var)

    def __ols_variance(self):
        """ Estimate the variance based on the OLS fit residuals """
        ols_y_fitted = self.ols_fit.predict(self.X)
        ols_res = self.y - ols_y_fitted
        ols_var = np.var(ols_res) # Variance of the residuals
        return(ols_var)


    def __boot_TS(self):
        """ Bootstrapped an OLS to infer the Theil Sen Confidence Interval """
        boot_ci_l = list() # Container

        for idx, _ in enumerate(self.X.index): # Drop one row each time
            boot_X = self.X.drop(index=self.X.index[idx])
            boot_y = self.y.drop(index=self.y.index[idx]) 

            boot_ols = sm.OLS(boot_y, boot_X).fit(cov_type='HC1') # OLS
            boot_ci = pd.DataFrame(boot_ols.conf_int(self.threshold),
                                   index=self.reg_l)
            boot_ci.columns = ['ci_left', 'ci_right']            
            boot_ci_l.append(boot_ci.T)

        # Rearrange the frames    
        dboot = pd.concat(boot_ci_l)
        dleft = dboot.loc['ci_left', :].copy()
        dright = dboot.loc['ci_right', :].copy()

        # Compute the median of the left and right bootstrapped CI
        dci = pd.concat([dleft.median(), dright.median()], axis=1)
        dci.columns = ['Theil_Sen_ci_left', 'Theil_Sen_ci_right']

        return(dci)
    
    def __coeff_frame(self):
        # Coefficients from the regressions
        ts_coeff = pd.DataFrame(self.ts_fit.coef_, index=self.reg_l)
        ols_coeff = pd.DataFrame(self.ols_fit.params, index=self.reg_l) 

        # OLS Confidence interval 
        ols_ci = pd.DataFrame(self.ols_fit.conf_int(self.threshold),
                              index=self.reg_l)
        
        dcoeff = pd.concat([ols_coeff, ts_coeff, ols_ci], axis=1)
        dcoeff.columns = ['OLS', 'Theil Sen', 'OLS_ci_left', 'OLS_ci_right']

        # Add the Theil Sen bootstrapped confidence interval
        dci = self.__boot_TS()
        dcoeff_final = pd.concat([dcoeff, dci], axis=1)
        return(dcoeff_final)
    
    # Plot function (public)
    def linear_coeff_plot(self, font_scale=2, left=0.15, right=0.85,
                          bottom=0.15,
                          legend_top=f'OLS - CI at',
                          legend_bottom= 'Theil Sen - CI at'):
        """ Plot the coefficients of the linear regression """
        
        sns.set(style='white', font_scale=font_scale, palette='deep',
            font='arial')

        # Plot the normalized coefficients
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        ols_ci = (self.linear_coeff['OLS_ci_right']
                  - self.linear_coeff['OLS_ci_left'])/2

        th_ci = (self.linear_coeff['Theil_Sen_ci_right']
                  - self.linear_coeff['Theil_Sen_ci_left'])/2
        
        ax1.bar(self.linear_coeff.index, self.linear_coeff['OLS'],
                yerr=ols_ci,align='center',
                label=f'{legend_top} {int(100*self.threshold)}%')
        ax1.axhline(y=0, color='black')
        ax1.legend(frameon=False, handlelength=0, loc='upper left')

        ax2.bar(self.linear_coeff.index, self.linear_coeff['Theil Sen'],
                yerr=th_ci, align='center',
                label=f'{legend_bottom} {int(100*self.threshold)}%')
        ax2.axhline(y=0, color='black')
        ax2.legend(frameon=False, handlelength=0, loc='upper left')

        ax2.set_xticks(range(len(self.linear_coeff.index)))
        ax2.set_xticklabels(self.linear_coeff.index, fontsize='small',
                            rotation=45)
        
        # plt.suptitle('Coefficients of the Standard '
        #              'and Robust Linear Regressions', fontsize='medium')

        plt.subplots_adjust(left=left, right=right, bottom=bottom)
        return(None)

    # Class-methods (methods returning a class defined below)
    def project(self, cond, input_proj=None, min_prob=None):
        
        # Note that the 'self' arg below is the parent class
        return(RobustProjection(self, cond, input_proj, min_prob))

class RobustProjection(object):
    def __init__(self, RobustFit, cond, input_proj, min_prob): 
        self.__dict__.update(RobustFit.__dict__) # Import from parent

        assert isinstance(cond, pd.DataFrame), ('Cond vector should be pandas '
                                                'dataframe')
        self.input_proj = input_proj
        
        self.cond = cond.copy()
        for var in [x for x in self.reg_l if x != 'intercept']: # no constant
            self.cond[var]=(self.cond[var] - self.mean_d[var])/self.std_d[var]
        self.cond['intercept'] = 1 # Because intercept can't be Zscored
                        
        # Linear projections
        self.ts_proj = float(np.dot(self.cond[self.reg_l],
                                    self.linear_coeff.loc[self.reg_l,
                                                          'Theil Sen']))
        self.ols_proj = float(np.dot(self.cond[self.reg_l],
                                     self.linear_coeff.loc[self.reg_l, 'OLS']))

        if self.model == 'TS':
            self.proj = self.ts_proj
        elif self.model == 'OLS':
            self.proj = self.ols_proj
        else:
            raise ValueError('Model parameter has to be in {TS, OLS}')

        if self.input_proj: # In case user wants to force a given projection
            self.proj = (self.input_proj
                         - self.mean_d[self.depvar])/self.std_d[self.depvar]
        
        # Logit fit
        self.logit_fit = self.__logit_fit() # Defined below
        self.prob_d = self.__relative_prob() # Defined below

        # Adjustment to the balance of risk if too large distortions
        if min_prob: # Make sure that both observations have min probability
            assert (min_prob >0 and min_prob <1), 'min_prob should be in (0,1)'
            self.prob_d[False] = np.clip(self.prob_d[False],
                                         min_prob, 1-min_prob)
            self.prob_d[True] = 1 - self.prob_d[False]
        else: # Without adjustment, ignore
            pass
        
        # Density fit
        self.skn_params = skn_params(mean=self.proj, variance=self.fit_var,
                                     cdf_target=self.prob_d[True])        
        self.skn = skewnorm(self.skn_params['shape'],
                            self.skn_params['location'],
                            self.skn_params['scale'])

        # Main quantities of interest
        m, v, s = self.skn.stats(moments='mvs')
        self.skn_mean, self.skn_var, self.skn_skew = float(m),float(v),float(s)

        _support = np.linspace(np.min(self.y), np.max(self.y), 10000)
        _y_pdf = self.skn.pdf(_support)
        self.mode = float(_support[np.argmax(_y_pdf)])

        # Unscaled skewnorm estimated by MLE (no link z-moments and moments)  
        z_sample = self.skn.rvs(size=1000)
        u_sample = [z_rescale(x, self.mean_d[self.depvar],
                              self.std_d[self.depvar]) for x in z_sample]
        unscaled_params = skewnorm.fit(u_sample)
        self.unscaled_skn = skewnorm(*unscaled_params)
        
    def __logit_fit(self):
        """ Based on a discretization determined by the projection """     
        self.ndata['bool_dep'] = (self.ndata[self.depvar] <= self.proj)
        
        if np.min(self.y) > self.proj:
            # Choose the min as "0" to have two categories
            self.ndata['bool_dep'] = (self.ndata[self.depvar] > np.min(self.y))
            print('Projection below historical minimum')

        if np.max(self.y) < self.proj:
            # Choose the max as "1" to have two categories
            self.ndata['bool_dep'] = (self.ndata[self.depvar] < np.max(self.y))
            print('Projection above historical maximum')
            
        self.binary_y = self.ndata['bool_dep'].values.ravel()
            
        # Fit the logistic model
        logit_fit = LogisticRegression(random_state=self.random_state,
                                       solver='lbfgs').fit(self.X,
                                                           self.binary_y)
        return(logit_fit)
    
    def __relative_prob(self):
        """ Compute the relative probability to determine the skewness """
        prob = self.logit_fit.predict_proba(self.cond).ravel()
        label = self.logit_fit.classes_
        prob_d = {l:p for l,p in zip(label, prob)}
        return(prob_d)

    # Plot functions
    def linear_contribution_plot(self, fscale=2):
        """ Plot the linear contribution of each variable to the mean """
        
        # Compute the linear contribution
        decomp = pd.DataFrame(index=self.reg_l)

        if self.model=='TS':
            decomp['coeff'] = self.linear_coeff.loc[self.reg_l, 'Theil Sen']
            model_name = 'Theil Sen'
        elif self.model=='OLS':
            decomp['coeff'] = self.linear_coeff.loc[self.reg_l, 'OLS']
            model_name = 'OLS'
            
        decomp['cond'] = self.cond[self.reg_l].T
        decomp['contribution'] = decomp['coeff']*decomp['cond']
        np.testing.assert_allclose(np.sum(decomp['contribution']),self.ts_proj)

        # Remove intercept for better charts
        dd = pd.DataFrame(decomp['contribution']).T.drop(columns=['intercept'])

        # Stacked bar Chart
        sns.set(style='white', font_scale=fscale, palette='deep', font='arial')
        fig, ax1 = plt.subplots()
        ax1 = dd.plot.bar(stacked=True, ax=ax1)
        ax1.axhline(y=0, color='black')
        ax1.legend(frameon=False)
    
        ax1.set_xticklabels([])
        ax1.set_ylabel('Contribution GDP p.p.', labelpad=20)
        ax1.set_title(f'Contribution to the Mean {model_name} '
                      'GDP Projection (excl. Intercept)', y=1.02)

    def odds_ratio_plot(self, title=None):
        d_or = pd.DataFrame(np.exp(self.logit_fit.coef_),
                            columns=self.reg_l).transpose()
        d_or.columns = ['odds_ratio']
                
        fig, ax = plt.subplots()
        ax.bar(d_or.index, 100*d_or['odds_ratio'], align='center')
        ax.axhline(y=100, ls='--', color='tab:red')
        ax.set_ylabel('Percentage', labelpad=20)
        ax.set_xticks(range(len(d_or.index)))
        ax.set_xticklabels(d_or.index, fontsize='small', rotation=45)

        if title:
            ttl = title
        else:
            ttl1 = 'Risk Drivers: Odds Ratio from Logistic Regression'
            ttl2 = 'Prob[(GDP <= Conditional Mean) vs (GDP > Conditional Mean)]'
            ttl = f'{ttl1} \n {ttl2}'
        ax.set_title(ttl, y=1.02)

        
    def density_plot(self, title='Robust Density Fit',
                     xlabel='GDP growth (p.p.)', ylabel='Density',
                     percentile=0.05, perc_direction='left', central='mode',
                     y_support_min=1.5, y_support_max=1.5, font_scale=2,
                     ticks_l=None, ax=None):
        
        # Fix the style
        sns.set(style='white', font_scale=font_scale, palette='deep',
                font='arial')

        ax = ax or plt.gca()
                
        y_support = np.linspace(y_support_min*np.min(self.y),
                                y_support_max*np.max(self.y), 1000)
        
        y_pdf = self.skn.pdf(y_support)

        # Need to rescale the y to be in line with the original value (no zsc)
        z_rsc = lambda x:z_rescale(x, self.mean_d[self.depvar],
                                   self.std_d[self.depvar])
        y_support = np.array([z_rsc(y) for y in y_support])

        re_proj = z_rescale(self.proj, self.mean_d[self.depvar],
                            self.std_d[self.depvar])
        
        # Some quantities of interest        
        pdf_mean = self.skn.pdf(self.skn_mean)
        pdf_mean_l = y_pdf[y_pdf<pdf_mean]

        r_mode = z_rsc(self.mode)
        
        pdf_mode = self.skn.pdf(self.mode)
        pdf_mode_l = y_pdf[y_pdf<pdf_mode]

        # Balance of risks
        p0 = int(round(100*self.prob_d[True], 0))
        p1 = int(100-p0)
        
        ax.plot(y_support, y_pdf, color='tab:blue', lw=3,
                 label='GDP Growth Density')

        # Vertical lines: either mode or mean
        if central=='mean':
            ax.plot([z_rsc(self.skn_mean)]*len(pdf_mean_l), pdf_mean_l,
                 color='tab:green', ls='--', label='Theil-Sen Mean')
        elif central=='mode':
            ax.plot([r_mode]*len(pdf_mode_l), pdf_mode_l,
                     color='tab:red', ls=':',
                     label='Parametrized Mode')
        elif central=='both':
            ax.plot([z_rsc(self.skn_mean)]*len(pdf_mean_l), pdf_mean_l,
                 color='tab:green', ls='--', label='Theil-Sen Mean')

            ax.plot([r_mode]*len(pdf_mode_l), pdf_mode_l,
                     color='tab:red', ls=':',
                     label='Parametrized Mode')            
        else:
            raise ValueError('central should be "mean" "mode", "both"')

        # Plot the expected shortfall for a given percentile
        y_per = self.skn.ppf(percentile)
        pdf_per = self.skn.pdf(y_per)
        y_per = z_rsc(y_per) # Rescale after computing the pdf
        
        pdf_per_l = y_pdf[y_pdf<pdf_per]
        
        ax.plot([y_per]*len(pdf_per_l), pdf_per_l, color='black', ls='-.')

        if perc_direction=='left':
            ax.fill_between(y_support, y_pdf, 0, where=y_support<y_per,
                             alpha=0.75, color='tab:red',
                            label=f'{int(100*percentile)}% Expected Shortfall')
        elif perc_direction=='right':
            ax.fill_between(y_support, y_pdf, 0, where=y_support>y_per,
                             alpha=0.75, color='tab:red',
                            label=f'{int(100*percentile)}% Expected Shortfall')
        else:
            raise ValueError('perc_direction should be "left" or "right"')

        ax.legend(frameon=False, fontsize='small')

        # Labels
        ax.set_xlabel(xlabel, labelpad=10, fontsize='small')
        ax.set_ylabel(ylabel, labelpad=10, fontsize='small')
        ax.set_title(f'{title} \n Balance of risks below vs. above mean: '
                      f'{p0}% vs. {p1}%',
                      y=1.02, fontsize='medium')

        # Xticks
        if ticks_l:
            ax.set_xticks(ticks_l)
        else:
            if central=='mean':
                ax.set_xticks([round(z_rsc(self.skn_mean),1)]
                               + [round(y_per, 1)])
            elif central=='mode':
                ax.set_xticks([round(r_mode,1)] + [round(y_per, 1)])
            elif central=='both':
                ax.set_xticks([round(re_proj,1)] + [round(y_per, 1)])
            else:
                raise ValueError('central should be "mode", "mean", "both"')
            
        return(ax)
