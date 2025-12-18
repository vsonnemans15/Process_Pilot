"""
Adapted from the data preprocessing code of I. Teinemaa et al., 
"Outcome-Oriented Predictive Process Monitoring: Review and Benchmark".
"""
import os

case_id_col = {}
activity_col = {}
resource_col = {}
timestamp_col = {}
label_col = {}
pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
control_flow_var = {}
control_flow_var_incremental = {}
control_flow_var_binary = {}
environmental_actions = {}
control_flow_var_attribute = {}
filename = {}

base_dir = os.path.dirname(os.path.abspath(__file__))  # folder where the script is
logs_dir = os.path.join(base_dir, "..", "data_logs")

#### SimBank settings ####
dataset = "SimBank"
filename[dataset] = os.path.join(logs_dir, "SimBank.csv")
case_id_col[dataset] = "case_nr"
activity_col[dataset] = "activity"
resource_col[dataset] = ""
timestamp_col[dataset] = "timestamp"
label_col[dataset] = "label"
pos_label[dataset] = "deviant"
neg_label[dataset] = "regular"

dynamic_cat_cols[dataset] = ["activity"]
static_cat_cols[dataset] = []
dynamic_num_cols[dataset] = ["unc_quality", "est_quality", "cum_cost", "elapsed_time", "interest_rate", "discount_factor", "min_interest_rate"]
static_num_cols[dataset] = ["amount"]

environmental_actions[dataset] = ['receive_acceptance', 'receive_refusal']
control_flow_var_incremental[dataset] = []
control_flow_var_binary[dataset]= []
control_flow_var_attribute[dataset] = ["noc", "nor"]
control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]


#### Traffic fines settings ####

for formula in range(1,3):
    dataset = "traffic_fines_%s"%formula
    
    filename[dataset] = os.path.join(logs_dir, "traffic_fines_%s.csv"%formula)
    
    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "Resource"
    timestamp_col[dataset] = "Complete Timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", 'Resource', "lastSent", "notificationType", "dismissal"]
    static_cat_cols[dataset] = ["article",  "vehicleClass"]
    dynamic_num_cols[dataset] = ["expense", "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"]
    static_num_cols[dataset] = ["amount", "points"]

    environmental_actions[dataset] = ['Payment']
    control_flow_var_incremental[dataset] = []
    control_flow_var_binary[dataset]= []
    control_flow_var_attribute[dataset] = []
    control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]
    
  
    

#### Sepsis Cases settings ####
datasets = ["sepsis_cases_%s" % i for i in range(1, 5)]

for i in [1,2,4]:
    
    dataset = f"sepsis_cases_{i}"
    filename[dataset] = os.path.join(logs_dir, "%s.csv" % dataset)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "org:group"
    timestamp_col[dataset] = "time:timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", 'org:group'] # i.e. event attributes
    static_cat_cols[dataset] = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                       'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                       'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                       'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                       'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
                       'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
                       'SIRSCritTemperature', 'SIRSCriteria2OrMore'] # i.e. case attributes that are known from the start
    dynamic_num_cols[dataset] = ['CRP', 'LacticAcid', 'Leucocytes', "hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ['Age']
    if i == 1:
        environmental_actions[dataset] = ['Return ER']
        control_flow_var_attribute[dataset] = ['recent_release'] #cannot be abstracted or transformed because defines the outcome of the case
        control_flow_var_incremental[dataset] = []
        control_flow_var_binary[dataset]= []   
      
    elif i == 2 or i == 4: 
        environmental_actions[dataset] = [] #return ER won't be in all_actions because we cut traces after Release activity
        control_flow_var_attribute[dataset] = [] 
        control_flow_var_incremental[dataset] = []
        control_flow_var_binary[dataset]= []   
     
    control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]
    

#### BPIC2017 settings ####

bpic2017_dict = {"bpic2017_cancelled": "BPIC17_O_Cancelled.csv",
                 "bpic2017_accepted": "BPIC17_O_Accepted.csv",
                 "bpic2017_refused": "BPIC17_O_Refused.csv"
                }

for dataset, fname in bpic2017_dict.items():

    filename[dataset] = os.path.join(logs_dir, fname)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = 'org:resource'
    timestamp_col[dataset] = 'time:timestamp'
    label_col[dataset] = "label"
    neg_label[dataset] = "regular"
    pos_label[dataset] = "deviant"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", "Accepted", "Selected"] 
    static_cat_cols[dataset] = ['ApplicationType', 'LoanGoal']
    dynamic_num_cols[dataset] = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', 'CreditScore',  "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"]
    static_num_cols[dataset] = ['RequestedAmount']
    environmental_actions[dataset] = ['O_Accepted', 'O_Cancelled', 'O_Returned']
    control_flow_var_incremental[dataset] = ['O_Create Offer', 'W_Call after offers']
    control_flow_var_binary[dataset]= []
    control_flow_var_attribute[dataset] = []
    control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]
    
  
    
#### Hospital billing settings ####
for i in range(1, 7):
    #for suffix in ["", "_sample10000", "_sample30000"]:
        dataset = "hospital_billing_%s" % (i)

        filename[dataset] = os.path.join(logs_dir, "hospital_billing_%s.csv" % (i))

        case_id_col[dataset] = "Case ID"
        activity_col[dataset] = "Activity"
        resource_col[dataset] = "Resource"
        timestamp_col[dataset] = "Complete Timestamp"
        label_col[dataset] = "label"
        neg_label[dataset] = "regular"
        pos_label[dataset] = "deviant"

        if i == 1:
            neg_label[dataset] = "deviant"
            pos_label[dataset] = "regular"

        # features for classifier
        dynamic_cat_cols[dataset] = ["Activity", 'Resource', 'actOrange', 'actRed', 'blocked', 'caseType', 'diagnosis', 'flagC', 'flagD', 'msgCode', 'msgType', 'state', 'version']#, 'isCancelled', 'isClosed', 'closeCode'] 
        static_cat_cols[dataset] = ['speciality']
        dynamic_num_cols[dataset] = ['msgCount', "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour"]#, "open_cases"]
        static_num_cols[dataset] = []

        if i == 2: 
            environmental_actions[dataset] = []
            control_flow_var_incremental[dataset] = []
            control_flow_var_binary[dataset]= []
            control_flow_var_attribute[dataset] = ["isCancelled"]
            control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]

        if i == 1: # label is created based on isCancelled attribute
            dynamic_cat_cols[dataset] = [col for col in dynamic_cat_cols[dataset] if col != "isCancelled"]
        elif i == 2:
            dynamic_cat_cols[dataset] = [col for col in dynamic_cat_cols[dataset] if col != "isClosed"]
    
            
#### BPIC2012 settings ####
bpic2012_dict = {"bpic2012_cancelled": "bpic2012_O_CANCELLED-COMPLETE.csv",
                 "bpic2012_accepted": "bpic2012_O_ACCEPTED-COMPLETE.csv",
                 "bpic2012_declined": "bpic2012_O_DECLINED-COMPLETE.csv"
                }

for dataset, fname in bpic2012_dict.items():

    filename[dataset] = os.path.join(logs_dir, fname)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "Resource"
    timestamp_col[dataset] = "Complete Timestamp"
    label_col[dataset] = "label"
    neg_label[dataset] = "regular"
    pos_label[dataset] = "deviant"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", "Resource"]
    static_cat_cols[dataset] = []
    dynamic_num_cols[dataset] = ["hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ['AMOUNT_REQ']

    environmental_actions[dataset] = ['O_CANCELLED-COMPLETE', "O_ACCEPTED-COMPLETE", 'O_SENT_BACK-COMPLETE']
    control_flow_var_incremental[dataset] = ['O_CREATED-COMPLETE']
    control_flow_var_binary[dataset]= []
    control_flow_var_attribute[dataset] = []
    control_flow_var[dataset] = environmental_actions[dataset] + control_flow_var_incremental[dataset] + control_flow_var_attribute[dataset] + control_flow_var_binary[dataset]
    