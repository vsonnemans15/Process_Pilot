"""
Adapted from the data preprocessing code of I. Teinemaa et al., 
"Outcome-Oriented Predictive Process Monitoring: Review and Benchmark".
"""
import pandas as pd
import os
import numpy as np
import sys
import pm4py
import os
import sys
import pandas as pd
import numpy as np

# Input and output
input_file = "../orig_logs/Road_Traffic_Fine_Management_Process.xes.gz"
output_file = "../orig_logs/Road_Traffic_Fine_Management_Process.csv"

# Load the XES log
log = pm4py.read_xes(input_file)

# Convert to pandas DataFrame
df = pm4py.convert_to_dataframe(log)

# Save as CSV
df.to_csv(output_file, sep=";", index=False)

print(f"Saved CSV to {output_file}")

input_data_folder = "../orig_logs"
output_data_folder = "../data_logs/"
filenames = ["Road_Traffic_Fine_Management_Process.csv"]

# changed the column names
case_id_col = "Case ID"
activity_col = "Activity"
resource_col = "Resource"
timestamp_col = "Complete Timestamp"
label_col = "label"
pos_label = "deviant"
neg_label = "regular"

freq_threshold = 10

# features for classifier
dynamic_cat_cols = ["Activity", "Resource", "lastSent", "notificationType", "dismissal"]
static_cat_cols = ["article", "vehicleClass"]
dynamic_num_cols = ["expense", "timesincecasestart"]
static_num_cols = ["amount", "points"]

static_cols = static_cat_cols + static_num_cols + [case_id_col]
dynamic_cols = dynamic_cat_cols + dynamic_num_cols + [timestamp_col]
cat_cols = dynamic_cat_cols + static_cat_cols


def extract_timestamp_features(group):
    # Sort descending by timestamp
    group = group.sort_values(timestamp_col, ascending=False, kind='mergesort')

    # Time since last event in months
    tmp = pd.to_datetime(group[timestamp_col]) - pd.to_datetime(group[timestamp_col]).shift(1)
    tmp = tmp.fillna(pd.Timedelta(0))
    group["timesincelastevent"] = tmp.dt.total_seconds() / (30*24*3600)

    # Time since case start in months
    tmp = pd.to_datetime(group[timestamp_col]) - pd.to_datetime(group[timestamp_col].iloc[-1])
    tmp = tmp.fillna(pd.Timedelta(0))
    group["timesincecasestart"] = (tmp.dt.total_seconds() // (30*24*3600)).astype(int)

    # Restore ascending order and assign event numbers
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


for filename in filenames:
    print("Starting...")
    sys.stdout.flush()
    data = pd.read_csv(os.path.join(input_data_folder, filename), sep=";")

    data.rename(columns=lambda x: x.replace('(case) ', ''), inplace=True)
    data = data.rename(columns={
        "case:concept:name": "Case ID",
        "concept:name": "Activity",
        "org:resource": "Resource",
        "time:timestamp": "Complete Timestamp"
    })

    # discard cases that never have "Payment" or "Send for Credit Collection"
    valid_end_activities = ["Payment", "Send for Credit Collection"]
    cases_with_terminal = data.groupby(case_id_col)[activity_col].apply(lambda acts: acts.isin(valid_end_activities).any())
    incomplete_cases = cases_with_terminal.index[~cases_with_terminal]
    data = data[~data[case_id_col].isin(incomplete_cases)]

    # add event duration
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data["timesincemidnight"] = data[timestamp_col].dt.hour * 60 + data[timestamp_col].dt.minute
    data["month"] = data[timestamp_col].dt.month
    data["weekday"] = data[timestamp_col].dt.weekday
    data["hour"] = data[timestamp_col].dt.hour

    # add features extracted from timestamp
    print("Extracting timestamp features...")
    sys.stdout.flush()
    data = data.groupby(case_id_col, group_keys=False).apply(extract_timestamp_features)
    
    data = data.sort_values([case_id_col, timestamp_col], ascending=[True, True], kind="mergesort").reset_index(drop=True)

    # Fill forward missing values within each case
    grouped = data.groupby(case_id_col)
    for col in static_cols + dynamic_cols:
        data[col] = grouped[col].transform(lambda grp: grp.ffill())

    # Handle categorical and remaining missing values
    data[cat_cols] = data[cat_cols].fillna('missing')
    data = data.fillna(0)
    # set infrequent factor levels to "other"
    for col in cat_cols:
        if col != activity_col:
            counts = data[col].value_counts()
            mask = data[col].isin(counts[counts >= freq_threshold].index)
            data.loc[~mask, col] = "other"

    # keep only final columns
    data = data[static_cols + dynamic_cols]

    # assign class labels
    print("Assigning class labels...")
    sys.stdout.flush()
    data = data.sort_values([case_id_col, timestamp_col], ascending=[True, True], kind="mergesort").reset_index(drop=True)

    # Apply labeling
    dt_labeled = data.groupby(case_id_col, group_keys=False).apply(
        check_if_activity_exists,
        activity="Send for Credit Collection",
        cut_from_idx=False
    )

    # save output
    dt_labeled.to_csv(os.path.join(output_data_folder, "traffic_fines_1.csv"), sep=";", index=False)