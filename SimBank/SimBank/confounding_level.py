import pandas as pd
import numpy as np
np.random.seed(42)

def set_delta(data, data_RCT, delta):
    RCT_percentage = 1 - delta
    if RCT_percentage == 1:
        data_RCT["RCT"]==1
        return data_RCT
    #Normal
    case_nr_unique = data["case_nr"].unique()
    np.random.shuffle(case_nr_unique)
    cutoff = int((RCT_percentage) * len(case_nr_unique))
    idx = data["case_nr"].isin(case_nr_unique[cutoff:])
    data_new = data[idx].copy()
    data_new["RCT"] = 0  # policy-based data is confounded

    #RCT
    case_nr_RCT_unique = data_RCT["case_nr"].unique()
    np.random.shuffle(case_nr_RCT_unique)
    cutoff_RCT = int((1 - RCT_percentage) * len(case_nr_RCT_unique))
    RCT_idx = data_RCT["case_nr"].isin(case_nr_RCT_unique[cutoff_RCT:])
    data_new_RCT = data_RCT[RCT_idx].copy()
    data_new_RCT["RCT"] = 1  # RCT data is unconfounded

    max_case_nr = data_new["case_nr"].max()
    min_case_nr_RCT = data_new_RCT["case_nr"].min()
    

    #Add max_case_nr + 1 to case_nr of RCT data
    data_new_RCT["case_nr"] = data_new_RCT["case_nr"] - min_case_nr_RCT + max_case_nr + 1
    #Add RCT data to normal data
    data_combined = pd.concat([data_new, data_new_RCT], ignore_index=True)
    # #Shuffle data
    unique_cases_combined = data_combined['case_nr'].unique()
    np.random.shuffle(unique_cases_combined)
    data_combined = data_combined[data_combined['case_nr'].isin(unique_cases_combined)]

    case_mapping = {case: idx for idx, case in enumerate(sorted(unique_cases_combined))}
    data_combined['case_nr'] = data_combined['case_nr'].map(case_mapping)
    return data_combined