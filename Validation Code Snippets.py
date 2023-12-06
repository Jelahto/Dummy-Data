



########################################
# WARNING
# WARNING
# WARNING
# WARNING
# WARNING

# This is code that I used for the validation study
# Therefore, some of the hard-coded numbers are wrong
# For example

# Prepare the datasets
df_reliability = df.iloc[:, 23:36]
df_skills_factor_items = df.iloc[:, 23:26]
df_blue_lps_factor_items = df.iloc[:, 26:29]
df_level_fit_factor_items = df.iloc[:, 29:32]
df_overall_rate_factor_items = df.iloc[:, 32:36]

# The column numbers in df.iloc[ ] are going to be different in a different dataset
# Some of the variable names might be different too 

#######################################




import pandas as pd
import numpy as np
import pingouin as pg
from datetime import datetime as dt
from datetime import timedelta
from datetime import date

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import seaborn as sns


########################################
# create scores, show ranges, show by org
#######################################

# logic to create quality of hire scores ... 
# Start here

df['skills_factor'] = (df['has_background_num'] +
                       df['has_skills_num'] + df['produced_qualty_num'])
skills = df.skills_factor.describe()

df['blue_lps_factor'] = (df['works_well_oth_num'] +
                         df['is_motivated_num'] + df['demonstrates_LPs_num'])
bluelps = df.blue_lps_factor.describe()

df['level_fit_factor'] = (df['correct_job_level_num'] +
                          df['role_good_fit_num'] + df['will_succeed_num'])
levelfit = df.level_fit_factor.describe()

df['overall_rate_factor'] = (df['hire_again_num'] + df['raises_bar_num'] +
                             df['was_good_hire_num'] + df['top_performer_num'])
overall = df.overall_rate_factor.describe()

df['total_qoh_score'] = (df['skills_factor'] + df['blue_lps_factor'] +
                         df['level_fit_factor'] + df['overall_rate_factor'])
qoh_total_score = df.total_qoh_score.describe()

score_ranges = pd.concat(
    [skills, bluelps, levelfit, overall, qoh_total_score], axis=1)
score_ranges.columns = ['Skills', 'Blue LPs',
                        'Level Fit', 'Rating', 'Overall QoH']

# score_ranges.to_clipboard()

# create scores by Visier organization
df['organization'] = df['cost_center_tier1']
df.loc[df['organization'] == "Environmental Health Safety",
       "organization"] = "Safety and Mission Assurance"
df.loc[(df['organization'].isin(['Security', 'Facilities',
        'Integrated Supply Chain', 'Quality'])), "organization"] = "Operations"
df['organization'].value_counts()

scores_by_org = pd.concat([
    df.groupby('organization').total_qoh_score.count(),
    df.groupby('organization').total_qoh_score.mean(),
    df.groupby('organization').total_qoh_score.std(),
], axis=1)

# scores_by_org.to_clipboard()

########################################
########################################
# reliability analysis
#######################################
########################################

# Prepare the datasets
df_reliability = df.iloc[:, 23:36]
df_skills_factor_items = df.iloc[:, 23:26]
df_blue_lps_factor_items = df.iloc[:, 26:29]
df_level_fit_factor_items = df.iloc[:, 29:32]
df_overall_rate_factor_items = df.iloc[:, 32:36]

# Now run reliability (Cronbach's alpha)
print("The reliability of the overall scale = ",
      pg.cronbach_alpha(df_reliability))
print("skills factor = ", pg.cronbach_alpha(df_skills_factor_items))
print("blue lps factor = ", pg.cronbach_alpha(df_blue_lps_factor_items))
print("level fit factor = ", pg.cronbach_alpha(df_level_fit_factor_items))
print("overall rating factor = ", pg.cronbach_alpha(df_overall_rate_factor_items))

########################################
########################################
# efa and cfa factor analysis
#######################################
########################################

# exploratory factor analysis

from factor_analyzer import (
    ConfirmatoryFactorAnalyzer, ModelSpecificationParser)
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer import FactorAnalyzer

# get the dataset
df_factor = df.iloc[:, 23:36]
df_factor.isna().value_counts()
df_factor.info()

# Bartlett's test of sphericity
chi_square_value, p_value = calculate_bartlett_sphericity(df_factor)
chi_square_value, p_value

# Kaiser-Meyer-Olkin (KMO) Test
kmo_all, kmo_model = calculate_kmo(df_factor)
kmo_all, kmo_model

# Choosing number of factors
fa = FactorAnalyzer(rotation=None)
fa.fit(df_factor)

eigenvals, vals = fa.get_eigenvalues()
eigenvals

# diagnosis
df_factor.corr()

########################################
########################################
########################################

# confirmatory factor analysis

model_dict = {
    "F1": ["has_background_num", "has_skills_num", "produced_qualty_num"],
    "F2": ["works_well_oth_num", "is_motivated_num", "demonstrates_LPs_num"],
    "F3": ["correct_job_level_num", "role_good_fit_num", "will_succeed_num"],
    "F4": ["hire_again_num", "raises_bar_num", "was_good_hire_num", "top_performer_num"]
}

model_spec = ModelSpecificationParser.parse_model_specification_from_dict(
    df_factor, model_dict)

cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
cfa.fit(df_factor.values)

cfa.loadings_
cfa.aic_
cfa.log_likelihood_

########################################
########################################
# correlations
#######################################
########################################

# correlation matrix
pd.DataFrame(df.columns)
items_list = list(range(23, 36))
dimensions_list = list(range(41, 46))
corr_matrix = df.iloc[:, [*items_list, *dimensions_list]].corr()

################
################
# corr_matrix.to_csv("corr_matrix.csv")
################
################

# with performance
df_hrbp_prep = df_hrbp[['employee_email', 'worker_id']]

# link QoH datafile to hrbp file
df_empid = pd.merge(df, df_hrbp_prep, how="left",
                     left_on="newhire_email", right_on="employee_email", sort=False)

# link new file to performance file
df_predict = pd.merge(df_empid, df_perf, how="left",
                     left_on="worker_id", right_on="employee_id", sort=False)

# Wow, no missing data! 
df_predict.rating_num.isnull().sum()

# prepare columns we want to correlate
pd.DataFrame(df_predict.columns)
items_list = list(range(23, 36))
dimensions_list = list(range(41, 46))

# check that we have all integers
df_predict.iloc[:, [*items_list, *dimensions_list, 52]].dtypes

# histogram for overall qualilty of hire scores
fig, ax = plt.subplots(layout='constrained')
ax.hist(df_predict['rating_num'], bins=20,
        facecolor='mediumslateblue', alpha=0.75, edgecolor="black")
ax.set_xlabel('Performance Rating')
ax.set_ylabel('Count')
ax.set_title('Year-End Review Dist')
ax.grid(False)
plt.show()

df_predict['rating_num'].describe()
df_predict['rating'].value_counts()
df_predict['rating_num'].value_counts()
pg.normality(df_predict['rating_num'])

# run matrix
corr_matrix = df_predict.iloc[:, [*items_list, *dimensions_list, 52]].corr()
corr_matrix.to_csv("corr_matrix_perf.csv")

# normalized ratings
# 16% 1 - 55 cases
# 68% 2 - 234 cases
# 16% 3 - 55 cases
normalized_yearly = np.concatenate([np.ones(55), np.full(234, 2), np.full(55, 3)])
pd.Series(normalized_yearly).describe()

# normalized QoH ratings
# min would be 13, max would be 65
# Set the mean and standard deviation of the normal distribution
mu, sigma = 39, 10
dist = np.random.normal(mu, sigma, size=344)
dist = np.clip(dist, 13, 65)
normalized_qoh = pd.Series(dist)
normalized_qoh.describe()

# rxy_corrected = (rxy * (SDx / SDy)) * (SDy' / SDx')
rxy = df_predict.loc[:, ['total_qoh_score', 'rating_num']].corr().iat[0,1]
SDx = df_predict['total_qoh_score'].std()
SDy = df_predict['rating_num'].std()
SDxi = normalized_qoh.describe()['std']
SDyi = pd.Series(normalized_yearly).describe()['std']
rxy_corrected = (rxy * (SDx / SDy)) * (SDyi / SDxi)
print(f"The corrected correlation is {rxy_corrected}")

# clean up variables
del mu, sigma, dist, normalized_yearly, normalized_qoh, rxy, SDx, SDy, SDxi, SDyi

########################################
########################################
# attrition analysis
#######################################
########################################

# link new file to performance file
df_attrit = pd.merge(df_predict, df_attrition, how="left",
                     left_on="employee_id", right_on="employee_id", sort=False)

df_attrit.attrition_status.notnull().sum()
df_attrit.attrition_status.isnull().sum()

# Only two people have left the company
df_attrit.iloc[
    df_attrit.loc[df_attrit['attrition_status'].notnull()].index.tolist(), 
    [41, 42, 43, 44, 45, 52]
    ]

# It seemed to be a personality issue
df_attrit.iloc[
    df_attrit.loc[df_attrit['attrition_status'].notnull()].index.tolist(), 
    [12, 13, 14]
    ]

########################################
########################################
# group differences
#######################################
########################################

# link new file to performance file
df_dem = pd.merge(df_predict, df_demo, how="left",
                     left_on="employee_id", right_on="employee_id", sort=False)

df_dem.gender.value_counts()
df_dem.ethnicity.value_counts()

# t-test for gender
df_dem.groupby("gender")["total_qoh_score"].mean()
x = df_dem[df_dem['gender'] == 'Male']['total_qoh_score']
y = df_dem[df_dem['gender'] == 'Female']['total_qoh_score']
pg.ttest(x, y)

# t-test for ethnicity
condition = df_dem['ethnicity'].isin(['White (Not Hispanic or Latino) (United States of America)', 'Asian (Not Hispanic or Latino) (United States of America)', 'Hispanic or Latino (United States of America)'])

filtered_df = df_dem[condition]
filtered_df.groupby("ethnicity")["total_qoh_score"].count()
filtered_df.groupby("ethnicity")["total_qoh_score"].mean()

# Test for means differences White, Hispanic
x = filtered_df[filtered_df['ethnicity'] == 'Hispanic or Latino (United States of America)']['total_qoh_score']
y = filtered_df[filtered_df['ethnicity'] == 'White (Not Hispanic or Latino) (United States of America)']['total_qoh_score']
pg.ttest(x, y)

# Test for means differences White, Asian
x = filtered_df[filtered_df['ethnicity'] == 'Asian (Not Hispanic or Latino) (United States of America)']['total_qoh_score']
y = filtered_df[filtered_df['ethnicity'] == 'White (Not Hispanic or Latino) (United States of America)']['total_qoh_score']
pg.ttest(x, y)

########################################
########################################
# survey duration
#######################################
########################################

# inspect how long it took for managers to complete
time_delt_15 = timedelta(days=0, hours=0, minutes=15)
time_delt_1 = timedelta(days=0, hours=0, minutes=1.5)
df_time = df[(df['total_time'] < time_delt_15) &
             (df['total_time'] > time_delt_1)]
result_time = df_time['total_time'].describe()
print(f"The mean trimmed survey duration is {result_time['mean']}")

# turn time into an int
df['timetaken_min'] = df.apply(
    lambda x: x['total_time'].total_seconds()/60, axis=1)
df['timetaken_min'].describe()

# something is wrong with the timer in MS Forms? Consider id = 329
# for id = 39 you have response variance and free text, yet duration == 13 seconds?
df[df['timetaken_min'] < 1].to_clipboard()
df[df['timetaken_min'] > 60].to_clipboard()

# use the integer value to inspect time
df_time_int = df[(df['timetaken_min'] < 15) & (df['timetaken_min'] > 1.5)]
df_time_int.timetaken_min.describe()

########################################
########################################
# data visualization
#######################################
########################################

# histogram for overall qualilty of hire scores
fig, ax = plt.subplots(layout='constrained')
ax.hist(df['total_qoh_score'], bins=20,
        facecolor='mediumslateblue', alpha=0.75, edgecolor="black")
ax.set_xlabel('Overall Score')
ax.set_ylabel('Count')
ax.set_title('Quality of Hire Scores')
ax.grid(False)
plt.show()

# Check normality of total score
pg.normality(df['total_qoh_score'])

# helper fucntion to plot endorsement of options for an individual item
# need to move this

def plot_item(item):
    """creates a bar plot of response categories for a single item in 'df' """

    to_plot = df[item].value_counts(normalize=True)

    if not 'Neither Agree or Disagree' in to_plot:
        concat_response = pd.Series([0], ["Neither Agree or Disagree"])
        to_plot = pd.concat([to_plot, concat_response])
    if not 'Disagree' in to_plot:
        concat_response = pd.Series([0], ["Disagree"])
        to_plot = pd.concat([to_plot, concat_response])
    if not 'Strongly Disagree' in to_plot:
        concat_response = pd.Series([0], ["Strongly Disagree"])
        to_plot = pd.concat([to_plot, concat_response])

    fig, ax = plt.subplots(layout='constrained')
    ax.bar(to_plot.index, to_plot.values)
    plt.show()
    return "... plot completed !"

# percentage response for each category, by question
# this creates a dataframe
responses = df.iloc[:, 9:22].apply(lambda x: x.value_counts(normalize=True))
export_responses = responses.T
export_responses = export_responses.iloc[:, [3, 0, 2, 1, 4]]
export_responses = export_responses.fillna(0)
export_responses = export_responses.reset_index()

names = export_responses['index']
plot_data = export_responses['Strongly Agree']

fig, ax = plt.subplots()
ax.barh(names, plot_data)
plt.style.use('seaborn-bright')
plt.show()

names = export_responses['index']
plot_data = export_responses['Disagree']

fig, ax = plt.subplots()
ax.barh(names, plot_data)
plt.style.use('seaborn-bright')
plt.show()

# histogram for time taken to complete
fig, ax = plt.subplots()  # layout = 'constrained' stretches it
fig.set_figheight(5)
fig.set_figwidth(10)
ax.hist(df_time_int.timetaken_min, bins=15,
        facecolor='mediumslateblue', alpha=0.75, edgecolor="black")
ax.set_xlabel('Time Taken', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title('Time to Complete Survey (N = 268)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(False)
plt.show()
### fig.savefig('test5.png', dpi=150, transparent=True),

