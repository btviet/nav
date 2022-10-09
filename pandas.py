import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pandas.plotting import register_matplotlib_converters
from sqlalchemy import all_
register_matplotlib_converters()
from datetime import datetime, timedelta
import statistics
from scipy.signal import savgol_filter
import os
import sys
import scipy.stats
from matplotlib.backends.backend_pdf import PdfPages

## Locked parameters
raw_window = 5 # The time bin for calculating the rate for the EDAC curve to be normalized
smooth_window = 11 # The time bin for calculating the rate for the curve that is to be smoothed
savgolwindow = 1095  # The size of the window of the Savitzky-Golay filter. 3 years
polyorder = 3 # Order of the polynomial for Savitzky-Golay filter

## The noise interval
upper_noiselimit = 0.8885549424705822 # Q3+1.5*IQR 
lower_noiselimit = -0.7007099616105935 # Q1-0.6*IQR

upper_boundary_range = 0.18 # Normal distribution
lower_boundary_range = 0.06 # Normal distribution
upper_noiselimit =  0.7187784869509977 


''' Zero set correction'''
def create_rawedac_df(): # Creates a dataframe from the raw data provided by MEX
    df = pd.read_csv(path + '/raw_files/MEX_EDAC.txt',skiprows=12, sep="\t",parse_dates = ['# DATE TIME'])
    df2 = pd.read_csv(path + '/raw_files/MEX_NDMW0D0G_2022_02_17_16_13_50.116.txt', skiprows=15,  sep="\t",parse_dates = ['# DATE TIME'])
    df.rename(columns={'# DATE TIME': 'date', 'NDMW0D0G [MEX]': 'edac'}, inplace=True) # Changing the name of the columns, old_name: new_name
    df2.rename(columns={'# DATE TIME': 'date', 'NDMW0D0G - AVG - 1 Non [MEX]': 'edac'}, inplace=True) # Changing the name of the columns, old_name: new_name
    df = df.append(df2)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    df = df.sort_values(by="date")
    df.set_index('date')
    return df 
  
  def zero_set_correct(): # Returns the zero-set corrected dataframe of the raw EDAC counter
    df = create_rawedac_df()
    diffs = df.edac.diff() # The difference in counter from row to row
    indices = np.where(diffs<0)[0] #  Finding the indices where the EDAC counter decreases instead of increases or stays the same
    for i in range(0, len(indices)):
        prev_value = df.loc[[indices[i]-1]].values[-1][-1]
        if i == len(indices)-1: # The last time the EDAC counter goes to zero
            df.loc[indices[i]:,'edac'] = df.loc[indices[i]:,'edac'] + prev_value # Add the previous
        else:
            df.loc[indices[i]:indices[i+1]-1,'edac'] = df.loc[indices[i]:indices[i+1]-1,'edac'] + prev_value
    return df 
  
  def resample_corrected_edac(): # Resamples the zero set corrected EDAC counter to have one reading each day, and save the resulting dataframe to a textfile
    df = zero_set_correct() # Retrieve the zero-set corrected dataframe
    df_list = [] # Initialize list to have the resampled values
    lastdate = df['date'][df.index[-1]].date() # The last date of the EDAC dataset
    currentdate = df['date'][0].date() # Set currentdate to be the first date of the EDAC dataset
    end_reached = False # Variable to be changed to True when the function has reached the last date
    while end_reached != True:
        day_indices = np.where(df['date'].dt.date == currentdate) # The indices in df for where same date as currentdate
        if len(day_indices[0]) == 0: # If date does not exist 
            df_list.append([currentdate, lastreading]) # Let the missing date have the reading of the previous date          
        else:
            lastreading = df['edac'][day_indices[0][-1]]  # The last reading of the current date
            df_list.append([currentdate, lastreading]) # Add the date and the last reading to the list
        currentdate =  currentdate + pd.Timedelta(days=1) # go to next day
        if lastdate-currentdate < pd.Timedelta('0hr0m0s'): # if the currentdate is past the lastdate
            end_reached = True 
    df_resampled =  pd.DataFrame(df_list, columns=['date', 'edac'])
    #df_resampled.to_csv(path + "resampled_corrected_edac.txt", sep='\t') # Save to file
    
    
    
''' Calculating the daily rates from the EDAC counter'''
def creating_df(): # Retrieves the zero set corrected raw EDAC counter
    df = pd.read_csv(path +'resampled_corrected_edac.txt',skiprows=0, sep="\t",parse_dates = ['date'])
    df.set_index('date') # set index to be the dates, instead of 0 1 2 ...
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True) # Removing the extra column with indices
    return df 

def create_rate_df(days_window): # Function to calculate a rate for each date
    # day window is the an odd number
    df_list = [] # Initialize list to keep dates and daily rates
    df = creating_df() # df with date and the value of the EDAC counter
    startdate = df['date'][df.index[days_window//2]].date() # The starting date in the data, includes the time
    lastdate = df['date'][df.index[-1]].date() # The last date and time in the dataset
    print("The starting date is ", startdate, "\nThe last date is ", lastdate)
    currentdate = startdate # Start from the first date
    end_reached = False
    while not end_reached:
        beginning_window = (currentdate-pd.Timedelta(days=days_window//2)) # Place the current date to be the middle of the window
        end_window = (currentdate + pd.Timedelta(days=days_window//2))
        beginning_values = df.iloc[np.where(df['date'].dt.date == beginning_window)[0][0]] # Date and EDAC value of beginning window
        end_values = df.iloc[np.where(df['date'].dt.date == end_window)[0][0]] # Date and EDAC of ending window
        diff = end_values-beginning_values
        diff_edac = diff[1] # Difference in EDAC value
        diff_days = diff[0] # Difference in days
        number_of_days = diff_days.days+1 # Number of days in the window
        current_edac_rate = diff_edac/number_of_days # Calculate the EDAC rate
        df_list.append([currentdate, current_edac_rate])
        currentdate = currentdate + pd.Timedelta(days=1) # Iterate to next day
        if lastdate-currentdate < pd.Timedelta(days=days_window//2): # Stopping condition for while-loop
            end_reached = True
        if currentdate.year != (currentdate-pd.Timedelta(days=1)).year: # For observation while running code
            print("Year reached: ", currentdate.year)
    df_rate =  pd.DataFrame(df_list, columns=['date', 'rate']) # Convert the list to a dataframe
    return df_rate # Return date, rate dataframe

def save_df_rate_to_txt(days_window): # Saves the calculated daily rates to a .txt-file
    df_rate = create_rate_df(days_window)
    df_rate.to_csv(path + 'daily_edac_rate_'+ str(days_window) + '.txt', sep='\t')
    
def initialize_dfs(raw_window, smooth_window): # Initialize the dataframes before normalizing
    # raw_df_rate is the dataframe with daily edac rates with the raw window, unaltered
    # smooth_df_rate is the dataframe with daily edac rates with the smooth window, unaltered
    # spikeless_df_rate is smooth_df_rate with the three largest spikes removed
    # y_filtered is the smoothed curve, based on spikeless_df_rate, with savitzky-golay filter
    
    raw_df_rate = pd.read_csv(path + 'daily_edac_rate_'+ str(raw_window) + '.txt', sep="\t", parse_dates=['date'])
    raw_df_rate.drop(raw_df_rate.columns[raw_df_rate.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True) # Removing the extra column with indices
    smooth_df_rate = pd.read_csv(path + 'daily_edac_rate_' + str(smooth_window)+ '.txt', sep="\t", parse_dates=['date'])
    smooth_df_rate.drop(smooth_df_rate.columns[smooth_df_rate.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True) # Removing the extra column with indices
    smooth_copy = smooth_df_rate.copy(deep=True) # Avoid changing the original dataframe, by creating a copy instead
    spikeless_df_rate, y_filtered = smooth(smooth_copy) 
    firstdate_smooth = smooth_df_rate['date'][smooth_df_rate.index[0]] # The first date of the smooth df 2005-01-06
    firstdate_raw = raw_df_rate['date'][raw_df_rate.index[0]] # The first date of the raw df 2005-01-03
    
    if firstdate_raw > firstdate_smooth: # If the raw df starts later than the smooth
        firstdate_index = np.where(smooth_df_rate['date']== firstdate_smooth)[0][0]  # Find where the smooth df is the early date
        smooth_df_rate=smooth_df_rate.iloc[firstdate_index:]
        spikeless_df_rate = spikeless_df_rate.iloc[firstdate_index:]
    else: # If the smooth df starts later than the raw df
        firstdate_index = np.where(raw_df_rate['date']== firstdate_smooth)[0][0]
        raw_df_rate = raw_df_rate.iloc[firstdate_index:] # Slice the raw df to start at the same time

    # Make sure the dataframes end at the same date 
    if len(raw_df_rate['rate']) >  len(y_filtered): # If the raw df ends at a later date 
        raw_df_rate = raw_df_rate.iloc[:len(y_filtered)] # Slice it to end earlier, same time as y_filtered
    elif len(raw_df_rate['rate']) <  len(y_filtered):
        y_filtered = y_filtered[:len(raw_df_rate['rate'])]
        spikeless_df_rate = spikeless_df_rate.iloc[:len(raw_df_rate['rate'])]
        smooth_df_rate = smooth_df_rate.iloc[:len(raw_df_rate['rate'])]
        
    spikeless_df_rate.reset_index(drop=True,inplace=True)
    raw_df_rate.reset_index(drop=True, inplace=True)
    smooth_df_rate.reset_index(drop=True, inplace=True)
    return raw_df_rate, smooth_df_rate, spikeless_df_rate, y_filtered
