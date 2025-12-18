"""
Adapted from the data preprocessing code of I. Teinemaa et al., 
"Outcome-Oriented Predictive Process Monitoring: Review and Benchmark".
"""
import pm4py
import pandas as pd
import numpy as np
import os

input_data_folder = "../orig_logs"
output_data_folder = "../data_logs/"
in_filename = "Sepsis Cases - Event Log.csv"

case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "time:timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

category_freq_threshold = 10

dynamic_cat_cols = ["Activity", 'org:group'] # i.e. event attributes
static_cat_cols = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
       'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
       'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
       'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
       'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
       'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
       'SIRSCritTemperature', 'SIRSCriteria2OrMore'] # i.e. case attributes that are known from the start
dynamic_num_cols = ['CRP', 'LacticAcid', 'Leucocytes']
static_num_cols = ['Age']

static_cols = static_cat_cols + static_num_cols + [case_id_col]
dynamic_cols = dynamic_cat_cols + dynamic_num_cols + [timestamp_col]
cat_cols = dynamic_cat_cols + static_cat_cols

def extract_timestamp_features(group):
    # Sort descending to compute time since last event
    group = group.sort_values(timestamp_col, ascending=False, kind='mergesort')
    
    # Time since last event in minutes
    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
    tmp = tmp.fillna(pd.Timedelta(0))
    group["timesincelastevent"] = tmp / pd.Timedelta(minutes=1)
    
    # Time since case start in minutes
    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1]
    tmp = tmp.fillna(pd.Timedelta(0))
    group["timesincecasestart"] = tmp / pd.Timedelta(minutes=1)
    
    # Restore chronological order
    group = group.sort_values(timestamp_col, ascending=True, kind='mergesort')
    
    # Event number within case
    group["event_nr"] = range(1, len(group) + 1)

    releases = {"Release A", "Release B", "Release C", "Release D", "Release E"}
    last_release_time = None
    timesince_last_release = []
    timesince_last_release_binary = []
    
    for _, row in group.iterrows():
        if last_release_time is None:
            delta = pd.Timedelta(0)  # <-- keep as Timedelta
        else:
            delta = row[timestamp_col] - last_release_time
        timesince_last_release.append(delta / pd.Timedelta(days=1))
        # convert to binary: 1 if < 28 days, else 0
        timesince_last_release_binary.append(int((delta < pd.Timedelta(days=28)) and (row["Activity"] == "Return ER")))
        
        # Update if current activity is a release
        if row["Activity"] in releases:
            last_release_time = row[timestamp_col]
    
    group["timesincelastrelease"] = timesince_last_release
    group['recent_release'] = timesince_last_release_binary
    
    return group

def check_if_activity_exists(group, activity):
    relevant_activity_idxs = np.where(group[activity_col] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        group[label_col] = pos_label
        return group
    else:
        group[label_col] = neg_label
        return group
    
def check_if_activity_exists_and_time_less_than(group, activity):
    relevant_activity_idxs = np.where(group[activity_col] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        if group["timesincelastevent"].iloc[idx] <= 28 * 1440: # return in less than 28 days
            group[label_col] = pos_label
            return group
        else:
            group[label_col] = neg_label
            return group
    else:
        group[label_col] = neg_label
        return group

def check_if_any_of_activities_exist(group, activities):
    if np.sum(group[activity_col].isin(activities)) > 0:
        return True
    else:
        return False
    
data = pd.read_csv(os.path.join(input_data_folder, in_filename), sep=";")
data = data.rename(columns={
        "case:concept:name": case_id_col,
        "concept:name": "Activity",
        "org:resource": "Resource",
        "time:timestamp": timestamp_col,
    })
data[case_id_col] = data[case_id_col].fillna("missing_caseid")

# remove incomplete cases
tmp = data.groupby(case_id_col).apply(check_if_any_of_activities_exist, activities=["Release A", "Release B", "Release C", "Release D", "Release E"])
incomplete_cases = tmp.index[tmp==False]
data = data[~data[case_id_col].isin(incomplete_cases)]


data = data[static_cols + dynamic_cols]
# add features extracted from timestamp
data[timestamp_col] = pd.to_datetime(data[timestamp_col])
data["timesincemidnight"] = data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute
data["month"] = data[timestamp_col].dt.month
data["weekday"] = data[timestamp_col].dt.weekday
data["hour"] = data[timestamp_col].dt.hour
data = data.groupby(case_id_col, group_keys=False).apply(extract_timestamp_features)

data = data.sort_values([case_id_col, timestamp_col], ascending=[True, True], kind="mergesort").reset_index(drop=True)

# Fill forward missing values within each case
grouped = data.groupby(case_id_col)

# impute missing values
for col in static_cols + dynamic_cols:
    data[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))
        
data[cat_cols] = data[cat_cols].fillna('missing')
data = data.fillna(0)
    
# set infrequent factor levels to "other"
for col in cat_cols:
    if col != activity_col:
        counts = data[col].value_counts()
        mask = data[col].isin(counts[counts >= category_freq_threshold].index)
        data.loc[~mask, col] = "other"

# first labeling
dt_labeled = data.sort_values(timestamp_col, ascending=True, kind="mergesort").groupby(case_id_col).apply(check_if_activity_exists_and_time_less_than, activity="Return ER")
dt_labeled.to_csv(os.path.join(output_data_folder, "sepsis_cases_1.csv"), sep=";", index=False)
    
# second labeling
dt_labeled = data.sort_values(timestamp_col, ascending=True, kind="mergesort").groupby(case_id_col).apply(check_if_activity_exists, activity="Admission IC")
dt_labeled.to_csv(os.path.join(output_data_folder, "sepsis_cases_2.csv"), sep=";", index=False)