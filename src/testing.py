from MDP_functions import *
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


def run_testing_phase(env, Q_table, num_episodes_test, knn_mapping, args):
    offline_env = deepcopy(env)
    print(f'Number of observed states : {len(offline_env.all_states_unabs)}')
    learning_rate = 0.8
    discount_factor = 0.9
    stopping_criteria = 100 #max length of traces
    env.extend_state_space_with_test()
    optimal_paths_test = []
    recommended_traces_test = []
    n_granted_test = 0
    cumulative_reward_test = []
    list_granted_test = []
    list_granted_percent_test = []
    list_len_ep_test = []
    case_id_tracker = []
    n_stopping_cases_test = 0
    n_stopping_cases_long_test = 0
    test_cases_with_impossible_actions = []

    rewards_new_cases = []
    rewards_seen_cases = []
    start_time = time.time()
    Q_test = Q_table.copy()
    if not knn_mapping: 
        n_new_rows = env.n_states - Q_table.shape[0]
        new_rows = np.zeros((n_new_rows, Q_test.shape[1]))
        Q_test = np.vstack([Q_test, new_rows])

    #for generalization evaluation
    n_new_states = 0
    n_unmapped_states = 0
    total_visited_states = 0

    n_seen_cases = 0 
    n_new_cases = 0
    new_starting_states = 0

    n_seen_granted_cases = 0
    n_new_granted_cases = 0
    ids_new_cases = []
    ids_seen_cases = []
    n_stopping_error = 0

    print(f"Testing the agent on {env.dataset} with {num_episodes_test} episodes")
    print(f"State abstraction function: {env.state_abstraction}, parameter: {env.k}, State cols: {env.state_cols}")


    #Q-learning implementation
    with tqdm(range(num_episodes_test), desc="Testing Progress", unit="episode") as pbar:
        for episode in pbar:
            done, granted = False, False
            reward_episode, len_epis = 0, 0
            Q_table_previous = Q_test
            new_case = False

            case_id = env.test_df[env.test_df['ID'] == env.test_case_ids[episode]]  
            current_state_unabs, current_state = env.reset(case_id) #gives id of states
            if knn_mapping: 
                candidate_mask = (offline_env.all_states_unabs['last_action'] == env.full_state_unabs['last_action']) #using offline_env to retrieve closest observed state
                for var in env.dataset_manager.control_flow_var:
                                if var in offline_env.all_states_unabs.columns and var in env.full_state_unabs:
                                    candidate_mask &= (offline_env.all_states_unabs[var] == env.full_state_unabs[var])
                # Filter candidate states
                candidate_states = offline_env.all_states_unabs[candidate_mask]
                if candidate_states.empty:
                    # fallback: use all states if no candidate found
                    candidate_states = offline_env.all_states_unabs  
                knn_local = NearestNeighbors(n_neighbors=1)
                knn_local.fit(candidate_states.values)
                new_state = np.array(env.full_state_unabs).reshape(1, -1)  # shape (1, num_features)
                nn_idx_local = knn_local.kneighbors(new_state, return_distance=False)[0, 0]
                nearest_state = candidate_states.iloc[nn_idx_local]
                current_state = offline_env.unabs_to_abs_state[tuple(nearest_state)] #use the neighbor's abstracted state
                

            seen_starting_state, mapped_to_abs_state = seen_offline(env, current_state_unabs, current_state)
            if not seen_starting_state:
                new_starting_states +=1
                new_case = True
            
            total_visited_states +=1


            while done == False:

                    possible_actions = [action for action in range(env.n_actions)
                                        if (env.state_unabs, action) in env.transition_proba and env.transition_proba[(env.state_unabs, action)].sum() > 0]
                    
                    if len(possible_actions) == 0 :
                        Q_test = Q_table_previous
                        n_stopping_cases_test +=1
                        break

                    if len_epis >= stopping_criteria:
                        Q_test = Q_table_previous
                        n_stopping_cases_long_test +=1
                        break
                    
                    max_value_indices = np.where(Q_test[current_state] == np.max(Q_test[current_state]))[0]
                    if len(max_value_indices) == 0:
                        n_stopping_error +=1
                        break
                    else:
                        action = np.random.choice(max_value_indices)
                    #action = np.random.choice(max_value_indices)
                    
                    next_state_unabs, next_state, reward, done, granted, _ = env.step(action)
                    if knn_mapping and next_state_unabs is not None: 
                        candidate_mask = (offline_env.all_states_unabs['last_action'] == env.full_state_unabs['last_action'])
                        for var in env.dataset_manager.control_flow_var:
                            if var in offline_env.all_states_unabs.columns and var in env.full_state_unabs:
                                candidate_mask &= (offline_env.all_states_unabs[var] == env.full_state_unabs[var])
                        # Filter candidate states
                        candidate_states = offline_env.all_states_unabs[candidate_mask]
                        if candidate_states.empty:
                            # fallback: use all states if no candidate found
                            candidate_states = offline_env.all_states_unabs  
                        knn_local = NearestNeighbors(n_neighbors=1)
                        knn_local.fit(candidate_states.values)
                        next_state = np.array(env.full_state_unabs).reshape(1, -1)  # shape (1, num_features)
                        nn_idx_local = knn_local.kneighbors(next_state, return_distance=False)[0, 0]
                        nearest_next_state = candidate_states.iloc[nn_idx_local]
                        next_state = offline_env.unabs_to_abs_state[tuple(nearest_next_state)] #use the neighbor's abstracted state       
                    
                    seen_state, mapped_to_abs_state = seen_offline(env, next_state_unabs, next_state)
                    if not seen_state: 
                            n_new_states +=1
                            new_case = True
                            if not mapped_to_abs_state:
                                n_unmapped_states +=1
                    total_visited_states +=1

                    #update Q-value using the Bellman Equation
                    if done:  # next_state is terminal
                        Q_test[current_state, action] += learning_rate * (reward - Q_test[current_state, action])
                    else:
                        Q_test[current_state, action] += learning_rate * (reward + discount_factor * np.max(Q_test[next_state]) - Q_test[current_state, action])

                    current_state = next_state  
                    current_state_unabs = next_state_unabs
                    reward_episode += reward
                    len_epis += 1
                   
                        
            if new_case:
                    n_new_cases += 1
                    rewards_new_cases.append(reward_episode)
                    ids_new_cases.append(env.test_case_ids[episode])
                    if granted:
                        n_new_granted_cases += 1 #to assess generalization
                        n_granted_test +=1
            else:
                    n_seen_cases += 1
                    rewards_seen_cases.append(reward_episode)
                    ids_seen_cases.append(env.test_case_ids[episode])
                    if granted:
                        n_seen_granted_cases += 1
                        n_granted_test +=1

            test_cases_with_impossible_actions.append(env.impossible_action_taken)
            case_id_tracker.append(case_id['ID'].iloc[0])
            cumulative_reward_test.append(reward_episode)
            list_len_ep_test.append(len_epis)
            list_granted_test.append(granted)
            list_granted_percent_test.append(100*(n_granted_test/(episode+1)))
            recommended_traces_test.append(env.current_trace)
            optimal_paths_test.append(env.current_path)

            pbar.set_postfix(success=f"{100 * (n_granted_test / (episode + 1)):.2f}%", len_episode=f"{np.mean(list_len_ep_test):.1f}, impossible action taken: {sum(test_cases_with_impossible_actions)}, stopping cases: {n_stopping_cases_test+n_stopping_cases_long_test}")
            
    end_time = time.time()
    runtime = end_time - start_time 

    # ========= Compute Log Policy Performance ===========
    test_df = env.test_df  

    test_log_policy_rewards = []
    test_log_policy_granted = []
    test_log_policy_lengths = []
    test_log_policy_case_ids = []

    for case_id, case_df in test_df.groupby('ID'):
        cum_cost = 0.0
        reward = 0.0
        granted = False
    
        for _, row in case_df.iterrows():
            state = row.to_dict()
            action_name = state['action']
            cum_cost += env.costs_dic.get(action_name, 0)
            is_terminal, granted_flag = env.config.is_terminal_successful_state(state)
            granted = granted_flag or granted
            
            if is_terminal:
                # Extract only the columns needed for simulation
                row_subset = {k: state[k] for k in env.state_cols_simulation}
                #r = env.config.binary_outcome_function(granted)
                r = env.config.outcome_function(granted, row_subset, cum_cost, env.scaler, env.numeric_cols, env.state_cols_simulation)
                reward += r
                break
            
        test_log_policy_granted.append(int(granted))
        test_log_policy_lengths.append(len(case_df))
        test_log_policy_rewards.append(reward)
        test_log_policy_case_ids.append(case_id)

    # Compute mean metrics
    mean_success_rate_test = np.mean(test_log_policy_granted)
    mean_case_length_test = np.mean(test_log_policy_lengths)
    mean_case_reward_test = np.mean(test_log_policy_rewards)

    print("Mean success rate (testing log):", mean_success_rate_test)
    print("Mean case length (testing log):", mean_case_length_test)
    print("Mean case reward (testing log):", mean_case_reward_test)


    # Aggregate results
    log_policy_results = {
        "case_ids": test_log_policy_case_ids,
        "rewards": test_log_policy_rewards,
        "granted": test_log_policy_granted,
        "lengths": test_log_policy_lengths,
        "avg_reward": np.mean(test_log_policy_rewards),
        "success_rate": 100 * np.mean(test_log_policy_granted),
    }

    #optimal_paths_test = optimal_paths_test[-1000:]
    print("=====Testing results====")
    if args.dataset in ['bpic2017_accepted', 'traffic_fines_1']: # too heavy
         testing_results = {#"dataset": env.dataset,
                            "observed_states": len(offline_env.all_states_unabs),
                            "state_abstraction": args.state_abstraction,
                            "k": env.k,
                            #"Q_table": Q_test,
                            "state_cols": env.state_cols,
                            "state_cols_simulation": env.state_cols_simulation,
                            "n_states": env.n_states,
                            "n_states_unabs": env.n_states_unabs,
                            "cumulative_reward": cumulative_reward_test,
                            "len_ep": list_len_ep_test,
                            "granted": list_granted_test,
                            "granted_percent": list_granted_percent_test,
                            "n_granted": n_granted_test,
                            "n_stopping_cases": n_stopping_cases_test, #because of no possible actions
                            "n_stopping_cases_long": n_stopping_cases_long_test, #because of stopping criteria
                            "runtime": runtime,
                            "test_case_ids": env.test_case_ids,
                            "case_id_tracker": case_id_tracker,
                            "n_new_states": n_new_states,
                            "n_unmapped_states": n_unmapped_states,
                            "total_visited_states": total_visited_states,
                            "n_seen_cases": n_seen_cases,
                            "n_new_cases": n_new_cases,
                            "new_starting_states": new_starting_states, 
                            "n_seen_granted_cases": n_seen_granted_cases,
                            "n_new_granted_cases": n_new_granted_cases,
                            "runtime": runtime,
                            "cases_with_impossible_actions": test_cases_with_impossible_actions,
                            "n_episodes": num_episodes_test,
                            "rewards_new_cases": rewards_new_cases,
                            "rewards_seen_cases": rewards_seen_cases,
                            "ids_new_cases": ids_new_cases,
                            "ids_seen_cases": ids_seen_cases,
                            "n_stopping_error": n_stopping_error,
                            "log_policy_results": log_policy_results
                            }
         
    else:
    
        testing_results = {"dataset": env.dataset,
                        "observed_states": len(offline_env.all_states_unabs),
                            "state_abstraction": args.state_abstraction,
                            "k": env.k,
                            "Q_table": Q_test,
                            "state_cols": env.state_cols,
                            "state_cols_simulation": env.state_cols_simulation,
                            "n_states": env.n_states,
                            "n_states_unabs": env.n_states_unabs,
                            "abs_state_space": env.all_states,
                            "original_state_space": env.all_states_unabs,
                            "all_state_unabs_index": env.all_state_unabs_index,
                            "all_state_index": env.all_state_index,
                            "unabs_to_abs_state": env.unabs_to_abs_state,
                            "optimal_path_frequencies": optimal_paths_test,
                            "recommended_traces": recommended_traces_test, 
                            "cumulative_reward": cumulative_reward_test,
                            "len_ep": list_len_ep_test,
                            "granted": list_granted_test,
                            "granted_percent": list_granted_percent_test,
                            "n_granted": n_granted_test,
                            "n_stopping_cases": n_stopping_cases_test, #because of no possible actions
                            "n_stopping_cases_long": n_stopping_cases_long_test, #because of stopping criteria
                            "runtime": runtime,
                            "test_case_ids": env.test_case_ids,
                            "case_id_tracker": case_id_tracker,
                            "n_new_states": n_new_states,
                            "n_unmapped_states": n_unmapped_states,
                            "total_visited_states": total_visited_states,
                            "n_seen_cases": n_seen_cases,
                            "n_new_cases": n_new_cases,
                            "new_starting_states": new_starting_states, 
                            "n_seen_granted_cases": n_seen_granted_cases,
                            "n_new_granted_cases": n_new_granted_cases,
                            "runtime": runtime,
                            "cases_with_impossible_actions": test_cases_with_impossible_actions,
                            "n_episodes": num_episodes_test,
                            "rewards_new_cases": rewards_new_cases,
                            "rewards_seen_cases": rewards_seen_cases,
                            "ids_new_cases": ids_new_cases,
                            "ids_seen_cases": ids_seen_cases,
                            "n_stopping_error": n_stopping_error,
                            "log_policy_results": log_policy_results
                            }
                            #"budget": budget}
    print("Testing runtime:", runtime)
    print(f"Testing success rate: {100 * (n_granted_test / (episode + 1)):.2f}%")
    return testing_results

from SimBank.SimBank.simulation import *
import math 
from itertools import product


def measure_simplicity(env):
    # Number of abstract states (nodes)
    n_nodes = env.n_states

    # Number of edges: Get the set of all abstracted transitions allowed by the abstracted MDP
    abs_transitions_model = set()
    # abs_transition_proba: (abs_state, action) -> prob vector over next abstract states
    for (abs_state, action), probs in env.abs_transition_proba.items():
        prob_vector = probs.toarray().flatten()

        if prob_vector.sum() <= 0:
            continue
        # find indices of next abstract states with non-zero prob
        next_abs_indices = np.where(prob_vector > 0)[0]
        for next_abs in next_abs_indices:
            abs_transitions_model.add((int(abs_state), int(action), int(next_abs)))
    n_edges = len(abs_transitions_model) #number of edges in abstracted MDP

    print(f"Simplicity: {n_nodes} nodes, {n_edges} edges")
    return n_nodes, n_edges, abs_transitions_model


def preprocess_event(event, offline_env):
     # Fill NaNs with 0
    event = {k: (0 if pd.isna(v) else v) for k, v in event.items()}
    event['last_action'] = offline_env.activity_index[event['activity']]

    # Keep only the features used in scaler
    event_df = pd.DataFrame([event])
    event_df[offline_env.numeric_cols] = offline_env.scaler.transform(event_df[offline_env.numeric_cols])
    event = event_df.iloc[0].to_dict()
    return event

def online_testing_simbank(offline_env, Q_table, num_episodes_test, args):
    
    dataset_params = {}
    #general
    dataset_params["train_size"] = 100000
    dataset_params["test_size"] = 6000
    dataset_params["val_share"] = .5
    dataset_params["train_val_size"] = 10000
    dataset_params["test_val_size"] = min(int(dataset_params["val_share"] * dataset_params["test_size"]), 1000)
    dataset_params["simulation_start"] = datetime(2024, 3, 20, 8, 0)
    dataset_params["random_seed_train"] = 82
    dataset_params["random_seed_test"] = 130
    #process
    dataset_params["log_cols"] = ["case_nr", "activity", "timestamp", "elapsed_time", "cum_cost", "est_quality", "unc_quality", "amount", "interest_rate", "discount_factor", "outcome", "quality", "noc", "nor", "min_interest_rate"]
    dataset_params["case_cols"] = ["amount"]
    dataset_params["event_cols"] = ["activity", "elapsed_time", "cum_cost", "est_quality", "unc_quality", "interest_rate", "discount_factor"]
    dataset_params["cat_cols"] = ["activity"]
    dataset_params["scale_cols"] = ["amount", "elapsed_time", "cum_cost", "est_quality", "unc_quality", "interest_rate", "discount_factor", "outcome"]
    #intervention
    dataset_params["intervention_info"] = {}
    #dataset_params["intervention_info"]["name"] = ["choose_procedure"]
    # dataset_params["intervention_info"]["name"] = ["set_ir_3_levels"]
    # dataset_params["intervention_info"]["name"] = ["time_contact_HQ"]
    dataset_params["intervention_info"]["name"] = ["choose_procedure", "set_ir_3_levels"]
    if dataset_params["intervention_info"]["name"] == ["choose_procedure"]:
        dataset_params["intervention_info"]["data_impact"] = ["direct"]
        dataset_params["intervention_info"]["actions"] = [["start_standard", "start_priority"]] #If binary, last action is the 'treatment' action
        dataset_params["intervention_info"]["action_width"] = [2]
        dataset_params["intervention_info"]["action_depth"] = [1]
        dataset_params["intervention_info"]["activities"] = [["start_standard", "start_priority"]]
        dataset_params["intervention_info"]["column"] = ["activity"]
        dataset_params["intervention_info"]["start_control_activity"] = [["initiate_application"]]
        dataset_params["intervention_info"]["end_control_activity"] = [["initiate_application"]]
    elif dataset_params["intervention_info"]["name"] == ["set_ir_3_levels"]:
        dataset_params["intervention_info"]["data_impact"] = ["indirect"]
        dataset_params["intervention_info"]["actions"] = [[0.07, 0.08, 0.09]]
        dataset_params["intervention_info"]["action_width"] = [3]
        dataset_params["intervention_info"]["action_depth"] = [1]
        dataset_params["intervention_info"]["activities"] = [["calculate_offer"]]
        dataset_params["intervention_info"]["column"] = ["interest_rate"]
        dataset_params["intervention_info"]["start_control_activity"] = [[]]
        dataset_params["intervention_info"]["end_control_activity"] = [[]]
    elif dataset_params["intervention_info"]["name"] == ["time_contact_HQ"]:
        dataset_params["intervention_info"]["data_impact"] = ["direct"]
        dataset_params["intervention_info"]["actions"] = [["do_nothing","contact_headquarters"]] #If binary, last action is the 'treatment' action
        dataset_params["intervention_info"]["action_width"] = [2]
        dataset_params["intervention_info"]["action_depth"] = [4] 
        dataset_params["intervention_info"]["activities"] = [["do_nothing", "contact_headquarters"]]
        dataset_params["intervention_info"]["column"] = ["activity"]
        dataset_params["intervention_info"]["start_control_activity"] = [["start_standard"]]
        dataset_params["intervention_info"]["end_control_activity"] = [["start_standard", "email_customer", "call_customer"]]
    elif dataset_params["intervention_info"]["name"] == ["choose_procedure", "set_ir_3_levels"]:
        dataset_params["intervention_info"]["data_impact"] = ["direct", "indirect"]
        dataset_params["intervention_info"]["actions"] = [["start_standard", "start_priority"], [0.07, 0.08, 0.09]]
        dataset_params["intervention_info"]["action_width"] = [2, 3] 
        dataset_params["intervention_info"]["action_depth"] = [1, 1] 
        dataset_params["intervention_info"]["activities"] = [["start_standard", "start_priority"], ["calculate_offer"]]
        dataset_params["intervention_info"]["column"] = ["activity", "interest_rate"]
        dataset_params["intervention_info"]["start_control_activity"] = [["initiate_application"], []]
        dataset_params["intervention_info"]["end_control_activity"] = [["initiate_application"], []]

    dataset_params["intervention_info"]["retain_method"] = "precise"
    # dataset_params["intervention_info"]["retain_method"] = "non-precise"

    # Combinations
    dataset_params["intervention_info"]["action_combinations"] = list(product(*dataset_params["intervention_info"]["actions"]))
    dataset_params["intervention_info"]["action_width_combinations"] = math.prod(dataset_params["intervention_info"]["action_width"])
    dataset_params["intervention_info"]["action_depth_combinations"] = math.prod(dataset_params["intervention_info"]["action_depth"])

    dataset_params["intervention_info"]["len"] = [action_width if action_width > 2 else 1 for action_width in dataset_params["intervention_info"]["action_width"]]
    dataset_params["intervention_info"]["RCT"] = False
    dataset_params["filename"] = "loan_log_" +  str(dataset_params["intervention_info"]["name"])
    #policy
    dataset_params["policies_info"] = {}
    dataset_params["policies_info"]["general"] = "real"
    dataset_params["policies_info"]["choose_procedure"] = {"amount": 50000, "est_quality": 5}
    dataset_params["policies_info"]["time_contact_HQ"] = "real"
    dataset_params["policies_info"]["min_quality"] = 2
    dataset_params["policies_info"]["max_noc"] = 3
    dataset_params["policies_info"]["max_nor"] = 1
    dataset_params["policies_info"]["min_amount_contact_cust"] = 50000

    Q_test = Q_table.copy()
    online_env = PresProcessGenerator(dataset_params, offline_env, dataset_params["random_seed_test"])
    # ================ Testing Phase (online) ================================
    learning_rate = 0.8
    discount_factor = 0.9
    stopping_criteria = 100 #max length of traces
    optimal_paths_test = []
    recommended_traces_test = []
    n_granted_test = 0
    cumulative_reward_test = []
    list_granted_test = []
    list_granted_percent_test = []
    list_len_ep_test = []
    case_id_tracker = []
    n_stopping_cases_test = 0
    n_stopping_cases_long_test = 0
    test_cases_with_impossible_actions = []

    rewards_new_cases = []
    rewards_seen_cases = []
    start_time = time.time()

    #for generalization evaluation
    n_new_states = 0
    n_unmapped_states = 0
    total_visited_states = 0

    n_seen_cases = 0 
    n_new_cases = 0
    new_starting_states = 0

    n_seen_granted_cases = 0
    n_new_granted_cases = 0
    ids_new_cases = []
    ids_seen_cases = []

    inv_activity_index = {v: k for k, v in offline_env.activity_index.items()}  
    print(f"Testing the agent on {offline_env.dataset} with {num_episodes_test} episodes")
    print(f"State abstraction function: {offline_env.state_abstraction}, parameter: {offline_env.k}, State cols: {offline_env.state_cols}")


    #Q-learning implementation
    with tqdm(range(num_episodes_test), desc="Testing Progress", unit="episode") as pbar:
        for episode in pbar:
            done, granted = False, False
            impossible_action = False
            reward_episode, len_epis = 0, 0
            new_case = False

            event, simulation_state, event_scaled = online_env.reset_for_online(seed_to_add=episode)


            current_state, seen_state, mapped_to_abs_state = offline_env.extend_state_space_with_one_event_online(event_scaled)
          
            if not seen_state:  
                new_starting_states += 1
                n_new_states += 1
                new_case = True
            if not mapped_to_abs_state:
                n_unmapped_states += 1
                new_row = np.zeros((1, Q_test.shape[1]))
                Q_test = np.vstack([Q_test, new_row])
            
            
            total_visited_states +=1

            while not done:

                    max_value_indices = np.where(Q_test[current_state] == np.max(Q_test[current_state]))[0]
                    action = np.random.choice(max_value_indices)
                    action_name = inv_activity_index[action]
                    
                    event, simulation_state, event_scaled, reward, granted, done, impossible_action = online_env.step_for_online(simulation_state, action=action_name)
                
                    next_state_unabs = tuple(np.atleast_1d([event_scaled[col] for col in offline_env.state_cols_simulation])) if event_scaled is not None else None

                    if next_state_unabs is not None: #invalid action, no state transition
                        next_state, seen_state, mapped_to_abs_state = offline_env.extend_state_space_with_one_event_online(event_scaled)
                        if not seen_state:  
                            new_case = True
                            n_new_states += 1
                        if not mapped_to_abs_state:
                            n_unmapped_states += 1
                            new_row = np.zeros((1, Q_test.shape[1]))
                            Q_test = np.vstack([Q_test, new_row])
                        total_visited_states +=1

                    #update Q-value using the Bellman Equation
                    if done:  # next_state is terminal
                        Q_test[current_state, action] += learning_rate * (reward - Q_test[current_state, action])
                    else:
                        Q_test[current_state, action] += learning_rate * (reward + discount_factor * np.max(Q_test[next_state]) - Q_test[current_state, action])

                    current_state = next_state  
                    current_state_unabs = next_state_unabs
                    reward_episode += reward
                    len_epis += 1
                
                        
            if new_case:
                n_new_cases += 1
                rewards_new_cases.append(reward_episode)
                ids_new_cases.append(episode)
                if granted:
                    n_new_granted_cases += 1 #to assess generalization
                    n_granted_test +=1
            else:
                n_seen_cases += 1
                rewards_seen_cases.append(reward_episode)
                ids_seen_cases.append(episode)
                if granted:
                    n_seen_granted_cases += 1
                    n_granted_test +=1

            test_cases_with_impossible_actions.append(impossible_action)
            case_id_tracker.append(episode)
            cumulative_reward_test.append(reward_episode)
            list_len_ep_test.append(len_epis)
            list_granted_test.append(granted)
            list_granted_percent_test.append(100*(n_granted_test/(episode+1)))
            recommended_traces_test.append(online_env.current_trace)
            optimal_paths_test.append(online_env.current_path)

            pbar.set_postfix(success=f"{100 * (n_granted_test / (episode + 1)):.2f}%", len_episode=f"{np.mean(list_len_ep_test):.1f}, impossible action taken: {sum(test_cases_with_impossible_actions)}, stopping cases: {n_stopping_cases_test+n_stopping_cases_long_test}")
            
    end_time = time.time()
    runtime = end_time - start_time 

    optimal_paths_test = optimal_paths_test

    

    print("=====Testing results====")
    testing_results = {"dataset": offline_env.dataset,
                        "state_abstraction": args.state_abstraction, #args.state_abstraction
                        "k": offline_env.k,
                        "Q_table": Q_test,
                        "state_cols": offline_env.state_cols,
                        "state_cols_simulation": offline_env.state_cols_simulation,
                        "n_states": offline_env.n_states,
                        "n_states_unabs": offline_env.n_states_unabs,
                        "abs_state_space": offline_env.all_states,
                        "original_state_space": offline_env.all_states_unabs,
                        "all_state_unabs_index": offline_env.all_state_unabs_index,
                        "all_state_index": offline_env.all_state_index,
                        "unabs_to_abs_state": offline_env.unabs_to_abs_state,
                        "optimal_path_frequencies": optimal_paths_test,
                        "recommended_traces": recommended_traces_test, 
                        "cumulative_reward": cumulative_reward_test,
                        "len_ep": list_len_ep_test,
                        "granted": list_granted_test,
                        "granted_percent": list_granted_percent_test,
                        "n_granted": n_granted_test,
                        "n_stopping_cases": n_stopping_cases_test, #because of no possible actions
                        "n_stopping_cases_long": n_stopping_cases_long_test, #because of stopping criteria
                        "runtime": runtime,
                        "test_case_ids": offline_env.test_case_ids,
                        "case_id_tracker": case_id_tracker,
                        "n_new_states": n_new_states,
                        "n_unmapped_states": n_unmapped_states,
                        "total_visited_states": total_visited_states,
                        "n_seen_cases": n_seen_cases,
                        "n_new_cases": n_new_cases,
                        "new_starting_states": new_starting_states, 
                        "n_seen_granted_cases": n_seen_granted_cases,
                        "n_new_granted_cases": n_new_granted_cases,
                        "runtime": runtime,
                        "cases_with_impossible_actions": test_cases_with_impossible_actions,
                        "n_episodes": num_episodes_test,
                        "rewards_new_cases": rewards_new_cases,
                        "rewards_seen_cases": rewards_seen_cases,
                        "ids_new_cases": ids_new_cases,
                        "ids_seen_cases": ids_seen_cases
                        }

    print("Testing runtime:", runtime)
    print(f"Testing success rate: {100 * (n_granted_test / (episode + 1)):.2f}%")
    return testing_results
