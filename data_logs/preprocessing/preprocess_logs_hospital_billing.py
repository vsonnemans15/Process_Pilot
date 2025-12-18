"""
Adapted from the data preprocessing code of I. Teinemaa et al., 
"Outcome-Oriented Predictive Process Monitoring: Review and Benchmark".
"""
import pandas as pd
import numpy as np
import os
import sys

input_data_folder = "../orig_logs"
output_data_folder = "../data_logs"
in_filename = "Hospital_Billing_Event_Log.csv"

case_id_col = "Case ID"
activity_col = "Activity"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

category_freq_threshold = 10

dynamic_cat_cols = ["Activity", 'Resource', 'actOrange', 'actRed', 'blocked', 'caseType', 'diagnosis', 'flagC', 'flagD', 'msgCode', 'msgType', 'state', 'version', 'isCancelled', 'isClosed', 'closeCode'] 
static_cat_cols = ['speciality']
dynamic_num_cols = ['msgCount']
static_num_cols = []

static_cols = static_cat_cols + static_num_cols + [case_id_col]
dynamic_cols = dynamic_cat_cols + dynamic_num_cols + [timestamp_col]
cat_cols = dynamic_cat_cols + static_cat_cols


def extract_timestamp_features(group):
    
    group = group.sort_values(timestamp_col, ascending=False, kind='mergesort')
    
    tmp = group[timestamp_col] - group[timestamp_col].shift(-1)
    tmp = tmp.fillna(0)
    group["timesincelastevent"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes

    tmp = group[timestamp_col] - group[timestamp_col].iloc[-1]
    tmp = tmp.fillna(0)
    group["timesincecasestart"] = tmp.apply(lambda x: float(x / np.timedelta64(1, 'm'))) # m is for minutes

    group = group.sort_values(timestamp_col, ascending=True, kind='mergesort')
    group["event_nr"] = range(1, len(group) + 1)
    
    return group

def check_if_activity_exists(group, activity, cut_from_idx=True):
    relevant_activity_idxs = np.where(group[activity_col] == activity)[0]
    if len(relevant_activity_idxs) > 0:
        idx = relevant_activity_idxs[0]
        group[label_col] = pos_label
        if cut_from_idx:
            return group[:idx]
        else:
            return group
    else:
        group[label_col] = neg_label
        return group   

def check_if_attribute_exists(group, attribute, cut_from_idx=True):
    group[label_col] = neg_label if True in list(group[attribute]) else pos_label
    relevant_idxs = np.where(group[attribute]==True)[0]
    if len(relevant_idxs) > 0:
        cut_idx = relevant_idxs[0]
        if cut_from_idx:
            return group[:idx]
        else:
            return group
    else:
        return group
    
    
data = pd.read_csv(os.path.join(input_data_folder, in_filename), sep=";")
data = data.rename(columns={
        "case:concept:name": "Case ID",
        "concept:name": "Activity",
        "org:resource": "Resource",
        "time:timestamp": "Complete Timestamp"
    })
data[case_id_col] = data[case_id_col].fillna("missing_caseid")
data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)

valid_end_activities = ["BILLED", "DELETE", "FIN"]
cases_with_terminal = data.groupby(case_id_col)[activity_col].apply(lambda acts: acts.isin(valid_end_activities).any())
incomplete_cases = cases_with_terminal.index[~cases_with_terminal]
data = data[~data[case_id_col].isin(incomplete_cases)]

data = data[static_cols + dynamic_cols]

# add features extracted from timestamp
data[timestamp_col] = pd.to_datetime(data[timestamp_col])

print("Imputing missing values...")
sys.stdout.flush()
# impute missing values
grouped = data.sort_values(timestamp_col, ascending=True, kind='mergesort').groupby(case_id_col)
for col in static_cols + dynamic_cols:
    data[col] = grouped[col].transform(lambda grp: grp.fillna(method='ffill'))
        
data[cat_cols] = data[cat_cols].fillna('missing')
data = data.fillna(0)
    
# set infrequent factor levels to "other"
for col in cat_cols:
    counts = data[col].value_counts()
    mask = data[col].isin(counts[counts >= category_freq_threshold].index)
    data.loc[~mask, col] = "other"
    
data = data.sort_values(timestamp_col, ascending=True, kind="mergesort")    
    
# second labeling
dt_labeled = data.groupby(case_id_col).apply(check_if_attribute_exists, attribute="isClosed", cut_from_idx=False)
#dt_labeled.drop(['isClosed'], axis=1).to_csv(os.path.join(output_data_folder, "hospital_billing_2.csv"), sep=";", index=False)
dt_labeled.to_csv(os.path.join(output_data_folder, "hospital_billing_2.csv"), sep=";", index=False)