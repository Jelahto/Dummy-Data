

########################
# Prepare the Survey Data File
########################

import pandas as pd
import re
import datetime as dt

# display all columns in dataframe
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# import the data
df = pd.read_csv("Quality of Hire Survey - Version 2_August 17, 2023_09.24.csv", header=0, skiprows=[1,2])
df.info()

########################
# Correct Variable Names
########################

# change all variables to snake_case
snake_case = re.compile(r'(?<!^)(?=[A-Z])')
new_columns = [snake_case.sub('_', item).lower() for item in list(df.columns)]

# create a list of variable names to modify
vars_to_change = [("i_p_address", "ip_address"),("duration (in seconds)", "duration_seconds"), ("recipient_last_name", "employee_last_name" ), ("recipient_first_name", "employee_first_name"), ("recipient_email", "manager_email"), ("external_reference", "employee_id"), ("rater_name_1", "rater_firstname"), ("rater_name_2", "rater_lastname"), ("rater_name_3", "rater_email_entered"), ("manager_email", "manager_email_auto"), ("employee_email", "employee_email_auto"), ("manager_i_d", "manager_id_auto"), ("external_data_reference", "employee_id_auto"), ("demonstrates__l_ps", "demonstrates_lps")]

# helper function to do the modification
def changeNames(new_columns, vars_to_change):
    for idx, name in enumerate(new_columns):
      newVar = [item[1] for item in vars_to_change if item[0] == name]
      if(newVar):
          print("New Var is", newVar)
          new_columns[idx] = newVar[0]
    return new_columns

# apply the function
df.columns = changeNames(new_columns, vars_to_change)
df.columns

########################
# Converting to datetime objects
########################

print("Converting start_date and end_date to datetime")
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])

########################
# Data Cleaning and Variable Creation
########################



