import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import Counter
import sys
import pickle
import time
import math
import random
from scipy.sparse import dok_matrix
import os
import gymnasium as gym
from collections import defaultdict
from gym import Env 
from gym import spaces 
from utils import *
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset_manager.DatasetManager import *

np.random.seed(0)

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

class Environment(Env):

    def __init__(self, dataset, k, state_abstraction, knn_mapping, binary_outcome):

        self.state_abstraction, self.k = state_abstraction, k
        self.knn_mapping = knn_mapping
        self.binary_outcome = binary_outcome
        self.dataset = dataset
        self.df, self.dataset_manager = get_real_data(dataset)
        self.config = DatasetMDP(dataset, self.df, self.dataset_manager)
        self.df = self.config.df
        self.all_cases = self.config.all_cases
        self.all_actions = self.config.all_actions
        self.activity_index = self.config.activity_index
        self.n_actions = self.config.n_actions
      
        self.df = self.config.cut_at_terminal_states(self.df,self.dataset_manager)
        
        if self.dataset == 'bpic2012_accepted': # for Branchi et al. (2022) baseline
            bins = [-float('inf'), 6000, 15000, float('inf')]
            labels = ['low', 'medium', 'high']
            self.df['AMOUNT_BUCKET'] = pd.cut(self.df['AMOUNT_REQ'], bins=bins, labels=labels, right=True)
        elif self.dataset == 'traffic_fines_1': # for Branchi et al. (2022) baseline
            bins = [-float('inf'), 50, float('inf')] #create the buckets before normalization
            labels = ['low', 'high']
            self.df['AMOUNT_BUCKET'] = pd.cut(self.df['amount'], bins=bins, labels=labels, right=False)
        
        train_size = int(0.8 * len(self.all_cases)) #chronological split, not random 
        self.train_case_ids = self.all_cases[:train_size] 
        self.test_case_ids = self.all_cases[train_size:] 
        self.train_df = self.df[self.df['ID'].isin(self.train_case_ids)].copy()
        self.test_df = self.df[self.df['ID'].isin(self.test_case_ids)].copy()

        self.state_cols_simulation, self.control_flow_var = define_real_state_cols(dataset, self.dataset_manager)
        self.reward = 0
        self.reward_scale = 10000
        self.unscaled_outcome = 0
        self.case = 0
        self.state = 0
        self.action = 0
        self.costs_dic = self.config.costs_dic

        # scale numeric columns
        self.numeric_cols = self.dataset_manager.static_num_cols + self.dataset_manager.dynamic_num_cols
        self.scaler = StandardScaler()
        self.df[self.numeric_cols] = self.scaler.fit_transform(self.df[self.numeric_cols])
        self.train_df[self.numeric_cols] = self.scaler.fit_transform(self.train_df[self.numeric_cols]) #fit on train set
        self.test_df[self.numeric_cols] = self.scaler.transform(self.test_df[self.numeric_cols]) #transform test set (without fitting to avoid data leakage)
        
         # --- encode categorical columns ---
        #self.dataset_manager.dynamic_cat_cols.append('last_action')
        self.categorical_cols = self.dataset_manager.static_cat_cols + self.dataset_manager.dynamic_cat_cols
        
        from sklearn.preprocessing import OneHotEncoder
        self.onehot_cols = []
        self.onehot_encoders = {}
        self.embedding_cols = {}
        self.label_encoders = {}  
    
        for col in self.categorical_cols:
            n_unique = self.train_df[col].nunique()
            if n_unique <= 100:
                # One-hot encode 
                ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
                reshaped = self.train_df[[col]] 
                encoded = ohe.fit_transform(reshaped)
                ohe_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                train_df_ohe = pd.DataFrame(encoded, columns=ohe_cols, index=self.train_df.index)
                self.train_df = pd.concat([self.train_df, train_df_ohe], axis=1)
                if col in self.state_cols_simulation:
                    self.state_cols_simulation.remove(col)
                self.state_cols_simulation += ohe_cols
                self.onehot_encoders[col] = ohe  
                self.onehot_cols.append(col)
                
                # Transform testing set using the same encoder
                reshaped_test = self.test_df[[col]]
                encoded_test = ohe.transform(reshaped_test)
                df_ohe_test = pd.DataFrame(encoded_test, columns=ohe_cols, index=self.test_df.index)
                self.test_df = pd.concat([self.test_df, df_ohe_test], axis=1)

            else:
                # Label encode
                le = LabelEncoder()
                self.train_df[col] = le.fit_transform(self.train_df[col])
                self.label_encoders[col] = le
                self.embedding_cols[col] = len(le.classes_)
                self.test_df[col] = self.test_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        # Unabstracted training state space for simulation 
        self.all_states_unabs = self.train_df[self.state_cols_simulation].drop_duplicates().reset_index(drop=True)
        self.n_states_unabs = len(self.all_states_unabs)
        self.last_state_unabs_offline = self.all_states_unabs.shape[0] - 1
        self.all_state_unabs_index = {tuple(row): idx for idx, row in self.all_states_unabs.iterrows()}
        print(f"Simulating the environment with {self.n_states_unabs} original states observed in training event log ({self.n_states_unabs/len(self.train_df)})")

        self.transition_proba = transition_probabilities_faster(self.train_df, self.state_cols_simulation, self.all_states_unabs , self.activity_index, self.n_actions)
        self.all_states_unabs = self.all_states_unabs.drop(columns=["state_index"])

        if self.knn_mapping: 
            all_states_unabs = np.array(self.all_states_unabs)  
            self.knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
            self.knn.fit(all_states_unabs)

        print("Abstracting the state space")
        self.abstract_state_space()  # Abstracted training state space 

        print("Preparing action mask")
        self.build_abs_transition_proba()

        self.illegal_trace_total = 0
        if self.binary_outcome:
            self.penalty_illegal = 100
        else:
            self.penalty_illegal = 1.1 * max(v for v in self.costs_dic.values() if isinstance(v, (int, float)))
            if self.penalty_illegal == 0:
                self.penalty_illegal = 1
   
    def step(self, action):
            info = {}
            done = False
            granted = False
            possible_actions = [action for action in range(self.n_actions) # According to the original MDP model
                                if (self.state_unabs, action) in self.transition_proba and self.transition_proba[(self.state_unabs, action)].sum() > 0]
            
            if action not in possible_actions or len(self.current_trace)==99: # Stopping criteria after 100 actions
                self.reward = -self.penalty_illegal
                done = True
                granted = False
                info["invalid_action"] = True
                self.illegal_action_count += 1
                self.current_trace_illegal = True
                self.impossible_action_taken = True
                return self.state_unabs, self.state, self.reward, done, granted, info
                
            # Get the next state
            previous_state_unabs = self.state_unabs
            transition = self.transition_proba.get((previous_state_unabs, action), None)
            prob_vector = transition.toarray().flatten()
            states = np.arange(self.n_states_unabs)
            next_state_unabs = np.random.choice(states, size=1, p=prob_vector)[0] 
            self.state_unabs = next_state_unabs
            self.full_state_unabs = self.all_states_unabs.iloc[self.state_unabs]
            # Map to abstracted state
            next_state = self.unabs_to_abs_state.get(tuple(self.all_states_unabs.iloc[next_state_unabs]), None)
            self.state = next_state
            self.full_state = self.all_states.iloc[self.state]         
            
            action_name = list(self.activity_index.keys())[list(self.activity_index.values()).index(action)]
            self.current_trace.append(action_name)
            self.current_path.append((action_name, self.full_state))
            self.cost = self.costs_dic.get(action_name, 0)
            if action_name == 'contact_headquarters': # Update the cost of contact_hq in SimBank depending on the customer's quality
                dummy = pd.DataFrame([self.all_states_unabs.iloc[self.state_unabs].to_dict()])[self.state_cols_simulation].copy()
                unscaled_numeric = self.scaler.inverse_transform(dummy[self.numeric_cols])
                dummy.loc[:, self.numeric_cols] = unscaled_numeric
                unscaled_event = dummy.iloc[0].to_dict()
                self.cost += 1000*unscaled_event['unc_quality']
     
            self.cum_costs += self.cost

            # Check if next_state is terminal state
            done, granted = self.config.is_terminal_successful_state(self.full_state_unabs)
            
            # Terminal actions
            if done:
                if self.binary_outcome == False: # Rewards based on case profitabitility
                    self.reward = np.float32(
                    self.config.outcome_function(granted,
                                                self.all_states_unabs.iloc[self.state_unabs].to_dict(),
                                                self.cum_costs, 
                                                self.scaler,
                                                self.numeric_cols,
                                                self.state_cols_simulation))
                else: # Rewards based on binary outcome (not used in our evaluation)
                    self.reward = np.float32(
                    self.config.binary_outcome_function(granted))
    
            else:
                self.reward = 0.0  # Intermediate action
                
            return self.state_unabs, self.state, self.reward, done, granted, info
    
    def reset(self, case):
        self.cum_costs = 0
        self.outcome = 0
        self.reward = 0.0
        self.current_trace = []
        self.current_path = []
        self.illegal_action_count = 0
        self.current_trace_illegal = False
        self.impossible_action_taken = False

        self.case = case
        # abstract state
        state_values = np.atleast_1d(self.case.iloc[0][self.state_cols])
        self.state = self.all_state_index[tuple(state_values)] 
        self.full_state = np.array(self.all_states.iloc[self.state])
        self.current_path.append(("start",self.full_state))
        #unabstracted state
        self.state_unabs = self.all_state_unabs_index[tuple(self.case.iloc[0][self.state_cols_simulation])]
        self.full_state_unabs = self.all_states_unabs.iloc[self.state_unabs]
        return self.state_unabs, self.state
    
    def abstract_state_space(self):
        if self.state_abstraction in ["full_k_means", "partial_k_means"]:
            # run k-means clustering
            self.clustering_state_cols = (
                [c for c in self.state_cols_simulation if c not in (["last_action"] + self.control_flow_var)]
                if self.state_abstraction == "partial_k_means" #partial preserves last action and control flow variables
                else self.state_cols_simulation)
            self.train_df, self.kmeanModel = k_means(self.train_df, self.clustering_state_cols, self.k)
            self.state_cols = (
                ["last_action", "cluster"] + self.control_flow_var
                if self.state_abstraction == "partial_k_means"
                else ["cluster"]
            )

        elif self.state_abstraction == 'branchi': # Baseline representation
            if self.dataset in ['bpic2012_accepted', 'traffic_fines_1']:
                self.state_cols = ['last_action', "AMOUNT_BUCKET"] + self.control_flow_var
                 
        elif self.state_abstraction == False:
            self.state_cols = self.state_cols_simulation
            
        elif self.state_abstraction == "contextual": # Baseline representation
            self.state_cols = ['last_action']

        elif self.state_abstraction == "structural":
            df_structural = self.train_df.copy()
            self.all_states_unabs_copy = self.all_states_unabs.copy()
            self.all_states_unabs_copy['state_index'] = self.all_states_unabs_copy.index
            df_structural["state"] = df_structural[self.state_cols_simulation].apply(lambda row: self.all_state_unabs_index.get(tuple(row), -1), axis=1)
            df_structural["next_state"] = df_structural.groupby("ID")["state"].shift(-1).fillna(-1).astype(int)
            #build transition sets per state
            state_action_map = defaultdict(set)
            for _, row in df_structural.iterrows():
                state_action_map[row["state"]].add(f"{row['action']};{row['next_state']}")

            #identify unique state groups
            unique_state_groups = {}
            merged_state_index = 0
            state_to_cluster = {}

            for state, action_next_pairs in state_action_map.items():
                    action_next_pairs = tuple(sorted(action_next_pairs))
                    if action_next_pairs not in unique_state_groups:
                        unique_state_groups[action_next_pairs] = merged_state_index
                        merged_state_index += 1
                    state_to_cluster[state] = unique_state_groups[action_next_pairs]  

            df_structural["cluster"] = df_structural["state"].map(state_to_cluster) 
            self.train_df = df_structural.copy()    
            self.state_cols = ["cluster"]

        elif self.state_abstraction == 'action-set':
            state_to_cluster = {}
            unique_sets = {}
            merged_state_index = 0

            # store action sets for each row in train_df
            action_sets = []

            for idx, row in self.train_df.iterrows():
                state = self.all_state_unabs_index[tuple(row[self.state_cols_simulation])]
                next_actions = [
                    a for a in range(self.n_actions)
                    if (state, a) in self.transition_proba and self.transition_proba[(state, a)].sum() > 0
                ]
                action_sets.append(next_actions)

                key = tuple(sorted(next_actions))
                if key not in unique_sets:
                    unique_sets[key] = merged_state_index
                    merged_state_index += 1
                state_to_cluster[state] = unique_sets[key]

            # add to train_df
            self.train_df['action_set'] = action_sets
            self.train_df['cluster'] = self.train_df[self.state_cols_simulation].apply(
                lambda row: state_to_cluster[self.all_state_unabs_index[tuple(row)]], axis=1
            )
            self.state_cols = ['cluster']


        elif self.state_abstraction == 'action-count':
            state_to_cluster = {}
            unique_counts = {}
            merged_state_index = 0

            action_counts_list = []

            for idx, row in self.train_df.iterrows():
                state = self.all_state_unabs_index[tuple(row[self.state_cols_simulation])]
                counts = np.zeros(self.n_actions, dtype=int)
                for (s, a), probs in self.transition_proba.items():
                    if s == state and probs.sum() > 0:
                        counts[a] += 1
                action_counts_list.append(counts)

                key = tuple(counts)
                if key not in unique_counts:
                    unique_counts[key] = merged_state_index
                    merged_state_index += 1
                state_to_cluster[state] = unique_counts[key]

            self.train_df['action_count'] = [list(c) for c in action_counts_list]
            self.train_df['cluster'] = self.train_df[self.state_cols_simulation].apply(
                lambda row: state_to_cluster[self.all_state_unabs_index[tuple(row)]], axis=1
            )
            self.state_cols = ['cluster']
           

        # build abstract state space
        self.all_states = self.train_df[self.state_cols].copy()
        if isinstance(self.all_states, pd.Series):
            self.all_states = self.all_states.to_frame()
        self.all_states = self.all_states.drop_duplicates().reset_index(drop=True)
        self.n_states = len(self.all_states)
        self.all_state_index = {tuple(row): idx for idx, row in self.all_states.iterrows()} 
        self.unabs_to_abs_state = {
                tuple(row[self.state_cols_simulation]) if isinstance(row[self.state_cols_simulation], (list, tuple, np.ndarray, pd.Series))
                else (row[self.state_cols_simulation],):  # wrap scalar in a tuple
                self.all_state_index.get(
                    tuple(row[self.state_cols]) if isinstance(row[self.state_cols], (list, tuple, np.ndarray, pd.Series))
                    else (row[self.state_cols],),
                    None
                )
                for _, row in self.train_df.iterrows()
            }
        self.last_state_offline = self.all_states.shape[0] - 1
        print(f"Training with {self.n_states} abstracted states (k={self.k})({self.n_states/len(self.train_df)})")
        return 
    
    def build_abs_transition_proba(self):
        """
        Build transition probability matrix for abstracted states.
        This lets us know which actions are ever possible from each abstract state.
        """
        self.abs_transition_proba = transition_probabilities_faster(self.train_df, self.state_cols, self.all_states , self.activity_index, self.n_actions)
        self.all_states = self.all_states.drop(columns=["state_index"])

        # Precompute mask: valid_actions[abs_state] = list of legal actions
        self.valid_actions_abs = {
            abs_state: [
                a for a in range(self.n_actions)
                if (abs_state, a) in self.abs_transition_proba and self.abs_transition_proba[(abs_state, a)].sum() > 0
            ]
            for abs_state in range(self.n_states)
        }
    
        
    def extend_state_space_with_test(self): # When a test set is available
        """
        Extend the original state space with unseen test states.
        Map test states into the abstract state space learned from training.
        Does NOT refit clustering → prevents test leakage.
        """
        test_states_unabs = (self.test_df[self.state_cols_simulation].drop_duplicates().reset_index(drop=True))

        # Find new unseen states
        merged = test_states_unabs.merge(
            self.all_states_unabs,
            how="left",
            indicator=True
        )
        self.unseen_states = merged.loc[merged["_merge"] == "left_only", self.state_cols_simulation]

        if len(self.unseen_states) > 0:
            print(f"Extending state space with {len(self.unseen_states)} unseen original test states")
            # Append to the unabstracted state space
            self.all_states_unabs = pd.concat([self.all_states_unabs, self.unseen_states]).reset_index(drop=True)
            self.n_states_unabs = len(self.all_states_unabs)

            # Update the transition matrix with new states and new transitions
            full_df = pd.concat([self.train_df, self.test_df], ignore_index=True)
            self.transition_proba = transition_probabilities_faster(full_df, self.state_cols_simulation, self.all_states_unabs , self.activity_index, self.n_actions)
            self.all_states_unabs = self.all_states_unabs.drop(columns=["state_index"])
            self.all_state_unabs_index = {
                tuple(row): idx for idx, row in self.all_states_unabs.iterrows()
            }
        
        else:
            print("No unseen states found in test set")

        if self.state_abstraction in ["full_k_means", "partial_k_means"]:
                # get cluster assignments for test states
                test_clusters = self.kmeanModel.predict(self.test_df[self.clustering_state_cols])
                self.test_df["cluster"] = test_clusters # store cluster index as new column

        elif self.state_abstraction in ["structural", "action-set", "action-count"]:
            # At runtime, transitions are unknown.
            # Structural abstraction reduces to "new unabs state → new abs state cluster idx" (except if knn-mapping).
            cluster_indices = []
            for _, row in self.unseen_states.iterrows():
                key = tuple(row)  # unabstracted state
                if key not in self.unabs_to_abs_state:
                    new_idx = len(self.all_states)
                    self.all_states = pd.concat([self.all_states, pd.DataFrame({'cluster': [new_idx]})], ignore_index=True)
                    # Update mapping from unabstracted state → abstract index
                    self.unabs_to_abs_state[key] = new_idx
                cluster_indices.append(self.unabs_to_abs_state[key])
            
            # Add the cluster column to test_df
            self.test_df["cluster"] = [
                self.unabs_to_abs_state[tuple(row)] 
                for _, row in self.test_df[self.state_cols_simulation].iterrows()
            ]
            
            self.n_states = len(self.all_states)
            self.all_state_index = {tuple(row): idx for idx, row in self.all_states.iterrows()}
            return f"Extending state space with {len(self.unseen_states)} unseen abstracted test states"
                

        test_states = (self.test_df[self.state_cols].drop_duplicates().reset_index(drop=True))
        merged = test_states.merge(
            self.all_states,
            how="left",
            indicator=True
        )
        unseen_states_abstract = merged.loc[merged["_merge"] == "left_only", self.state_cols]
        
        
        self.all_states = pd.concat([self.all_states, unseen_states_abstract]).reset_index(drop=True)
        self.n_states = len(self.all_states)
        self.all_state_index = {tuple(row): idx for idx, row in self.all_states.iterrows()}
        print(f"Extending state space with {len(unseen_states_abstract)} unseen abstracted test states")
        print('update unabstracted to abstracted state mapping')

        for _, row in self.test_df.iterrows():
            key = tuple(row[self.state_cols_simulation])
            abs_idx = self.all_state_index.get(tuple(row[self.state_cols]), None)
            if abs_idx is not None:
                self.unabs_to_abs_state[key] = abs_idx  
    
    def extend_state_space_with_one_event_online(self, event): #When online deployment is available (SimBank)
            """
            Extend the original state space with one online event.
            Map state into the abstract state space learned from training.
            Does NOT refit clustering → prevents test leakage.
            """
            event_row = pd.DataFrame(
                [[np.float64(event[col]) for col in self.state_cols_simulation]],
                columns=self.state_cols_simulation
            )
            # Convert event_row to a tuple for dictionary lookup
            full_state_unabs = tuple(event_row.iloc[0].values)

            # Check if seen
            seen_state = ((self.all_states_unabs[self.state_cols_simulation] == event_row.iloc[0]).all(axis=1)).any()

            if seen_state:
                state = self.unabs_to_abs_state[full_state_unabs]  # now works
                mapped_to_abs_state = True
                return state, seen_state, mapped_to_abs_state

            #for unseen original states
            elif self.knn_mapping:
                candidate_mask = (
                    (self.all_states_unabs['last_action'] == event['last_action']) &
                    (self.all_states_unabs['noc'] == event.get('noc', 0)) &
                    (self.all_states_unabs['nor'] == event.get('nor', 0))
                )
                candidate_states = self.all_states_unabs[candidate_mask]

                knn_local = NearestNeighbors(n_neighbors=1)
                knn_local.fit(candidate_states.values)
                nn_idx_local = knn_local.kneighbors([full_state_unabs], return_distance=False)[0, 0]
                nearest_state = candidate_states.iloc[nn_idx_local]
                state = self.unabs_to_abs_state[tuple(nearest_state)]
                mapped_to_abs_state = True
                return state, seen_state, mapped_to_abs_state
                
            elif self.state_abstraction in ["full_k_means", "partial_k_means"]:
                    # get cluster assignments for test states
                    X = np.array([event[col] for col in self.clustering_state_cols]).reshape(1, -1)
                    assigned_cluster = self.kmeanModel.predict(X)[0]
                    event["cluster"] = assigned_cluster # store cluster index as new column

            elif self.state_abstraction in ["structural", "action-set", "action-count"]:
                    event["cluster"] = self.n_states  # assign new cluster index

            #add new original state
            self.all_states_unabs = pd.concat(
                    [self.all_states_unabs, pd.DataFrame([full_state_unabs], columns=self.state_cols_simulation)],
                    ignore_index=True
                )
            self.n_states_unabs = len(self.all_states_unabs)

            #map to new or seen abstract state
            full_state_abs = tuple(np.atleast_1d([event[col] for col in self.state_cols]))
            existing_rows = [tuple(row) for row in self.all_states.values]
            if full_state_abs in existing_rows:
                mapped_to_abs_state = True  # already exists
                state = existing_rows.index(full_state_abs)  
            else:
                mapped_to_abs_state = False  # new abstracted state
                self.all_states = pd.concat(
                    [self.all_states, pd.DataFrame([full_state_abs], columns=self.state_cols)],
                    ignore_index=True
                )
                self.n_states = len(self.all_states)
                state = self.n_states - 1
                
            self.unabs_to_abs_state[full_state_unabs] = state  

            return state, seen_state, mapped_to_abs_state


class DatasetMDP:
    def __init__(self, dataset, df, dataset_manager):
        self.dataset = dataset
        self.df = df
        self.control_flow_var = dataset_manager.control_flow_var #includes env actions + other control flow variables
        self.environmental_actions = dataset_manager.environmental_actions
        self.control_flow_var_incremental = dataset_manager.control_flow_var_incremental
        self.control_flow_var_binary = dataset_manager.control_flow_var_binary
        self.control_flow_var_attribute = dataset_manager.control_flow_var_attribute
        if self.dataset == 'SimBank': #add the interest rate in the action space
            mask = self.df["action"] == "calculate_offer"
            self.df.loc[mask, "action"] = self.df.loc[mask, "interest_rate"]
        self.df, self.all_cases, self.all_actions, self.activity_index, self.n_actions = self.remove_env_actions(self.df) #remove environmental actions that are not controlled by the business
        self.costs_dic = self.define_costs_from_log() # costs of individual actions
        self.win_action, self.loss_actions = self.define_terminal_actions() 
    
    def cut_at_terminal_states(self, df, dataset_manager):
        """
        Cut traces at the first terminal state per case,
        but also keep the last row of the case (e.g., archive_application).
        """
        def cut_case(case_df):
            for idx, row in case_df.iterrows():
                state = row.to_dict()
                is_terminal, _ = self.is_terminal_successful_state(state)
                if is_terminal:
                    last_idx = case_df.index[-1]
                    if idx == last_idx:
                        return case_df.loc[:idx]  
                    else:
                        return pd.concat([case_df.loc[:idx], case_df.loc[[last_idx]]])
            return case_df  
        
        df = df.groupby("ID", group_keys=False).apply(cut_case).reset_index(drop=True)
        df = df.sort_values(by=['ID', dataset_manager.timestamp_col]).reset_index(drop=True)
        return df
    

    def remove_env_actions(self, df):
        for env_activity in self.control_flow_var: # Control_flow_var_attribute remains the same
            if env_activity in self.environmental_actions or env_activity in self.control_flow_var_binary:
                print("Transforming environmental or control flow action as binary state var:", env_activity)
                # Create a binary column: 1 if the action occurs at this row
                df[env_activity] = (df['action'] == env_activity).astype(int)
                # Compute cumulative max per case to keep 1 after first occurrence
                df[env_activity] = df.groupby('ID')[env_activity].cummax()
            elif env_activity in self.control_flow_var_incremental:
                print("Transforming control flow action as incremental state var:", env_activity)
                # Create an incremental column: count occurrences of the action per case
                df[env_activity] = (df['action'] == env_activity).astype(int)
                df[env_activity] = df.groupby('ID')[env_activity].cumsum()
        
        # Remove environmental actions by replacing them with the previous action in the trace
        # Boolean mask where actions are control-flow
        mask = df['action'].isin(self.environmental_actions)
        prev_values = df[['action', 'last_action']].shift(1)
        df.loc[mask, ['action', 'last_action']] = prev_values.loc[mask]
        prev_idx = mask.shift(-1, fill_value=False)
        df = df.loc[~prev_idx].reset_index(drop=True)
        all_cases = df['ID'].unique()
        
        if self.dataset == 'hospital_billing_2': 
            df['isCancelled'] = df['isCancelled'].map({False: 0, True: 1})

        all_actions = df["action"].unique() 
        activity_index = {activity: idx for idx, activity in enumerate(all_actions)}
        n_actions = len(all_actions)
        df['last_action'] = df['action'].map(activity_index)
        
        return df, all_cases, all_actions, activity_index, n_actions



    def define_costs_from_log(self):
        if self.dataset == 'SimBank':
            return {
                "initiate_application": 0,
                    "start_standard": 10,
                    "start_priority": 5000,
                    "validate_application": 20,
                    "contact_headquarters": 1000,
                    "skip_contact": 0,
                    "email_customer": 10,
                    "call_customer": 20,
                    "calculate_offer": 400,
                    0.07: 400,
                    0.08: 400,
                    0.09: 400,
                    "cancel_application": 30,
                    "receive_acceptance": 10,
                    "receive_refusal": 10,
                    "stop_application": 0
                }
        

        if self.dataset == 'traffic_fines_1': 
            return {
                'Create Fine': 0.0,
                'Send Fine': 0.0,
                'Add Penalty': 0.0,
                'Send for Credit Collection': 0.0,
                'Insert Fine Notification': 0.0,
                'Insert Date Appeal to Prefecture': 0.0, 
                'Send Appeal to Prefecture': -1, 
                'Receive Result Appeal from Prefecture': 0.0, 
                'Notify Result Appeal to Offender': 0.0, 
                'Appeal to Judge': -1.0,
                "start": 0,
                'archive_application': 0,
            }
        if self.dataset in ['hospital_billing_2', 'bpic2012_accepted', 'bpic2017_accepted']:
            unique_activities = self.df['action'].unique()
            return {
                act: 0.0 if act in ["start", "archive_application"] else 10.0
                for act in unique_activities
            }
        if self.dataset in ['sepsis_cases_1', 'sepsis_cases_2']:
            return {
                "start": 0,
                'archive_application': 0,
                "ER Registration": 150,
                "Leucocytes": 25,
                "CRP": 20,
                "LacticAcid": 35,
                "ER Triage": 250,
                "ER Sepsis Triage": 350,
                "IV Liquid": 75,
                "IV Antibiotics": 200,
                "Admission NC": 2000,   
                "Release A": 50,
                "Admission IC": 7000,   
                "Release B": 50,
                "Release E": 50,
                "Release C": 50,
                "Release D": 50,
            }
        
    def is_terminal_successful_state(self, state): 
        """
        Determine whether the current state is terminal and whether it is successful (granted).
        """
        is_terminal = False
        is_successful = False

        # ----- Action-based terminal states ----- 
        if self.dataset in ['bpic2017_accepted', 'bpic2012_accepted']:
            last_action = int(state.get('last_action'))
            terminal_actions = self.win_action + self.loss_actions
            if last_action in terminal_actions:
                is_terminal = True
                is_successful = last_action in self.win_action  
        elif self.dataset == 'SimBank':
            last_action = state.get('last_action') 
            if state.get('receive_acceptance')==1: # Defined by environmental state variable
                is_terminal, is_successful = True, True
            elif state.get('receive_refusal') ==1:
                is_terminal, is_successful = True, False
            elif last_action == self.activity_index['cancel_application']:
                is_terminal, is_successful = True, False

        elif self.dataset == 'sepsis_cases_2' or self.dataset == 'sepsis_cases_4':
            last_action = int(state.get('last_action'))
            terminal_actions = self.win_action + self.loss_actions
            if last_action in terminal_actions:
                is_terminal = True
                is_successful = last_action in self.win_action  
        # ----- Attribute-based terminal states -----
        elif self.dataset == 'sepsis_cases_1':
            last_action = int(state.get('last_action'))
            recent_release = state.get('recent_release', 0)
            return_ER = state.get('Return ER')
            if last_action in [self.activity_index['Release A'], self.activity_index['Release B'], self.activity_index['Release C'], self.activity_index['Release D'], self.activity_index['Release E']]:
                is_terminal = True
                # Successful if Return ER did not happened within 28 days or no Return ER after Release.
                if return_ER == 1 and recent_release == 0:
                     is_successful = True
                elif return_ER == 0:
                     is_successful = True                    
        elif self.dataset  == 'traffic_fines_1':
            if state.get('Payment')==1:
                is_terminal, is_successful = True, True
            elif int(state.get('last_action')) in self.loss_actions:
                is_terminal, is_successful = True, False
        
        elif self.dataset == 'hospital_billing_2':
            last_action = int(state.get('last_action'))
            cancellation_status = state.get('isCancelled')
            if cancellation_status==1: # Unsuccessful if billing package is cancelled
                is_terminal, is_successful = True, False
            elif last_action == self.activity_index['archive_application'] and cancellation_status==0:
                is_terminal, is_successful = True, True

        return is_terminal, is_successful

        
    def define_terminal_actions(self): # Used to determine terminal states above
        if self.dataset == 'bpic2017_accepted':
            win = [self.activity_index['A_Pending']]
            loss = [self.activity_index['A_Cancelled'],
                    self.activity_index['A_Denied'], self.activity_index['archive_application']]
            
        elif self.dataset == 'bpic2012_accepted':
            win = [self.activity_index['A_APPROVED-COMPLETE']]
            loss = [self.activity_index['A_CANCELLED-COMPLETE'],
                    self.activity_index['A_DECLINED-COMPLETE']]
            
        elif self.dataset == 'traffic_fines_1':
            win = []
            loss = [self.activity_index['Send for Credit Collection']]
        
        elif self.dataset == 'sepsis_cases_1':
            win = [self.activity_index['Release A'], self.activity_index['Release B'], self.activity_index['Release C'], self.activity_index['Release D'], self.activity_index['Release E']]
            loss = []
             
        elif self.dataset == 'sepsis_cases_2':
            win = [self.activity_index['Release A'], self.activity_index['Release B'], self.activity_index['Release C'], self.activity_index['Release D'], self.activity_index['Release E']]
            loss = [self.activity_index['Admission IC']]
             
        elif self.dataset == 'sepsis_cases_4':
            win = [self.activity_index['Release A']]
            loss = [self.activity_index['Release B'], self.activity_index['Release C'], self.activity_index['Release D'], self.activity_index['Release E']]

        elif self.dataset == 'hospital_billing_2':
            win = [] # Need to go until the end of trace to determine if success
            loss = [self.activity_index['DELETE']]

        elif self.dataset == 'SimBank':
            win = []
            loss = [self.activity_index['cancel_application']]
            
        else:
            win, loss = [], []
        return win, loss
    
    def binary_outcome_function(self, granted): #no used in the paper
        if granted: 
            r = 1000
        else:
            r = 0
        return r

    def outcome_function(self, granted, current_event, cum_cost, scaler, numeric_cols, state_cols):
   
        if self.dataset == 'SimBank':
            if granted:
                # Unscale the event 
                dummy = pd.DataFrame([current_event])[state_cols].copy()
                unscaled_numeric = scaler.inverse_transform(dummy[numeric_cols])
                dummy.loc[:, numeric_cols] = unscaled_numeric
                unscaled_event = dummy.iloc[0].to_dict()
                
                # Profit function in the SimBank paper
                i = unscaled_event["interest_rate"]
                A = unscaled_event["amount"]
                risk_factor = (10 - unscaled_event['est_quality']) / 200 
                risk_free_rate = 0.03 # around the 10 year Belgian government bond yield 16/11/2023, also accounts for inflation
                df = risk_free_rate + risk_factor
                n = 10 # number of years
                future_earnings = A * (1 + i)**n
                discount_future_earnings = future_earnings / (1 + df)**n
                exp_profit = discount_future_earnings - cum_cost - A - 100
                return exp_profit
            else:
                return -cum_cost - 100

        elif self.dataset == 'bpic2017_accepted' and granted:
            dummy = pd.DataFrame([current_event])[state_cols].copy()
            unscaled_numeric = scaler.inverse_transform(dummy[numeric_cols])
            dummy.loc[:, numeric_cols] = unscaled_numeric
            unscaled_event = dummy.iloc[0].to_dict()

            A = unscaled_event["OfferedAmount"]
            # Determine interest rate by offer amount class (inspired by Branchi et al. (2022))
            if A <= 6000:
                r = 0.16
            elif A <= 15000:
                r = 0.18
            else:
                r = 0.20
            interest = A * r
            return interest - cum_cost
        
        elif self.dataset == 'bpic2012_accepted' and granted:
            dummy = pd.DataFrame([current_event])[state_cols].copy()
            unscaled_numeric = scaler.inverse_transform(dummy[numeric_cols])
            dummy.loc[:, numeric_cols] = unscaled_numeric
            unscaled_event = dummy.iloc[0].to_dict()
            A = unscaled_event["AMOUNT_REQ"]
            # Determine interest rate by class (inspired by Branchi et al. (2022))
            if A <= 6000: 
                r = 0.16
            elif A <= 15000:
                r = 0.18
            else:
                r = 0.20
            interest = A * r
            return interest - cum_cost
        
        elif self.dataset == 'traffic_fines_1' and granted:
                dummy = pd.DataFrame([current_event])[state_cols].copy()
                unscaled_numeric = scaler.inverse_transform(dummy[numeric_cols])
                dummy.loc[:, numeric_cols] = unscaled_numeric
                unscaled_event = dummy.iloc[0].to_dict()
                months = unscaled_event["timesincecasestart"] / (60 * 24 * 30)  # minutes → months
                if months <= 6: #(inspired by Branchi et al. (2022))
                    credits = 3
                elif months <= 12:
                    credits = 2
                else:
                    credits = 1
                return credits - cum_cost
        elif self.dataset in ['sepsis_cases_1', 'sepsis_cases_2']:
            if granted:  # successful discharge
                health_benefit = 10000  
                return health_benefit - cum_cost
            else:
                return -cum_cost

        elif self.dataset == 'hospital_billing_2' and granted:
                revenue = 1000
                return revenue - cum_cost
    
        else: #for all datasets for unsuccessful cases
            return -cum_cost 

def get_real_data(dataset):
      
        dataset_manager = DatasetManager(dataset)
        df = dataset_manager.read_dataset()
        df = df.rename(columns={dataset_manager.activity_col: "action", 
                        dataset_manager.case_id_col: "ID"})
        #dataset_manager.dynamic_cat_cols.remove(dataset_manager.activity_col)

        all_actions = df["action"].unique() 

        activity_index = {activity: idx for idx, activity in enumerate(all_actions)}
        df['last_action'] = df['action'].map(activity_index)
        
        return df, dataset_manager 

def define_real_state_cols(dataset, dataset_manager):

    dataset_manager.dynamic_cat_cols.remove(dataset_manager.activity_col)

    for col in ['hour', 'weekday', 'month', 'timesincemidnight', 
            'timesincelastevent', 'event_nr', 'open_cases']:
        if col in dataset_manager.dynamic_num_cols:
            dataset_manager.dynamic_num_cols.remove(col)

    # Only drop timesincecasestart for non-traffic datasets
    if dataset != 'traffic_fines_1' and 'timesincecasestart' in dataset_manager.dynamic_num_cols:
        dataset_manager.dynamic_num_cols.remove('timesincecasestart')
    
    control_flow_var = dataset_manager.control_flow_var
    state_cols_simulation = ['last_action'] + dataset_manager.dynamic_cat_cols + dataset_manager.dynamic_num_cols + dataset_manager.static_cat_cols + dataset_manager.static_num_cols + control_flow_var
    
    return state_cols_simulation, control_flow_var



def find_next_state(current_state, action, transition_proba, n_states_unabs): #using unabstracted original states

        prob_vector = transition_proba.get((current_state, action), None).toarray().flatten()
        states = np.arange(n_states_unabs)
        next_state = np.random.choice(states, size=1, p=prob_vector)[0] 
        proba = prob_vector[next_state]
                
        return next_state, proba



# Estimate the Transition Probabilities in the Abstract MDP
def transition_probabilities_faster_abs(df, state_cols, all_states, activity_index, n_actions):

    if isinstance(state_cols, (str, int)):
        state_cols = [state_cols]

    # Assign an index to all abstracted states
    all_states = all_states.copy()
    all_states['state_index'] = all_states.index

    # Helper function to convert a row to a tuple key
    def make_state_tuple(state_row):
        values = []
        for col in state_cols:
            val = state_row[col]
            if isinstance(val, (list, tuple, np.ndarray, np.generic)):
                values.extend(np.ravel(val))
            else:
                values.append(val)
        return tuple(values)

    # Map state tuples to indices
    state_index_map = {make_state_tuple(row): int(row['state_index'])
                       for _, row in all_states.iterrows()}

    n_states = len(all_states)
    state_action_freq = dok_matrix((n_states, n_actions), dtype=np.float64)
    state_freq = {}

    # Process each case
    for case_id, group in df.groupby('ID'):
        group = group.reset_index(drop=True)

        action_sequence = group['action'].values

        for i in range(1, len(group)):
            current_state = state_index_map[make_state_tuple(group.iloc[i-1])]
            current_action = activity_index[action_sequence[i]]
            next_state = state_index_map[make_state_tuple(group.iloc[i])]

            # Update counts
            state_action_freq[current_state, current_action] += 1
            if (current_state, current_action) not in state_freq:
                state_freq[(current_state, current_action)] = dok_matrix((n_states, 1), dtype=np.float64)
            state_freq[(current_state, current_action)][next_state, 0] += 1

    # Compute transition probabilities
    transition_proba = {}
    for (state, action), freq_matrix in state_freq.items():
        if state_action_freq[state, action] > 0:
            transition_proba[(state, action)] = freq_matrix / state_action_freq[state, action]

    return transition_proba


# Estimate the Transition Probabilities in the original MDP
def transition_probabilities_faster(df, state_cols, all_states, activity_index, n_actions):

    all_states['state_index'] = all_states.index
    state_index_map = {
                (tuple(row[state_cols])): int(row['state_index'])
                for _, row in all_states.iterrows()
            }


    n_states = len(all_states)
    state_action_freq = dok_matrix((n_states, n_actions), dtype=np.float64)
    state_freq = {}

    for case_id, group in df.groupby('ID'):
                group = group.reset_index(drop=True)

                state_sequence = group[state_cols].to_records(index=False)
                action_sequence = group['action'].values

            
                for i in range(1,len(state_sequence)):
                    
                    current_state = state_index_map[
                            (tuple(state_sequence[i-1][col] for col in state_cols))
                    ]
                    current_action = activity_index[action_sequence[i]]
                    next_state = state_index_map[
                            (tuple(state_sequence[i][col] for col in state_cols))
                    ]
                          
                    state_action_freq[current_state, current_action] += 1
                    if (current_state, current_action) not in state_freq:
                        state_freq[(current_state, current_action)] = dok_matrix((n_states, 1), dtype=np.float64)
                    state_freq[(current_state, current_action)][next_state, 0] += 1

    transition_proba = {}  

    for (state, action), freq_matrix in state_freq.items():
        if state_action_freq[state, action] > 0:
            transition_proba[(state, action)] = freq_matrix / state_action_freq[state, action]

    return transition_proba
