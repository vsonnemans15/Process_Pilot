from MDP_functions import *
import argparse
import math
from sklearn.preprocessing import StandardScaler
from testing import *
from utils import *
import copy
from sklearn.neighbors import NearestNeighbors

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Run Process Pilot training with specified dataset and state abstraction.")
parser.add_argument('--dataset', type=str, required=True, help="Dataset name (e.g., 'bpic2017_accepted')")
parser.add_argument('--state_abstraction', type=str, default='False', help="State abstraction method: 'False', 'structural', 'contextual', 'k_means', or 'k_means_features'")
parser.add_argument('--k', type=int, default=0, help="Number of clusters if using k-means abstraction")
parser.add_argument('--action_masking', type=str2bool, default=False, help="Use of action masking during training")
parser.add_argument('--binary_outcome', type=str2bool, default=False, help="Reward based on binary outcome (True) or profitability (False)")
parser.add_argument('--num_episodes', type=int, default=10000, help="Number of training episodes")
parser.add_argument('--checkpoint_interval', type=int, default=100000, help="Number of training episodes before testing phase")

args = parser.parse_args()
dataset = args.dataset
state_abstraction = args.state_abstraction
k = args.k
action_masking = args.action_masking
binary_outcome = args.binary_outcome
if state_abstraction in ['False-knn', "structural-knn", "action-set-knn", 'action-count-knn']:
    state_abstraction = state_abstraction.replace("-knn", "")
    knn_mapping = True
else:
    knn_mapping = False

checkpoint_interval = args.checkpoint_interval

end_file = f"masking_{action_masking}_binaryoutcome_{binary_outcome}_final"

if state_abstraction == 'False':
    state_abstraction = False
    k = None
elif state_abstraction in ['contextual', 'structural', 'action-count','action-set']:
    k = None
else:
    pass

def convert_to_tuple(path):
    return tuple(tuple(x) if isinstance(x, list) else x for x in path)

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.abspath(os.path.join(script_dir, '..', 'final_results_long'))
os.makedirs(results_dir, exist_ok=True)


# ========== Create environment from event log ===========

env = Environment(dataset, k, state_abstraction, knn_mapping, binary_outcome)
print(f"Number of cases: {len(env.all_cases)}")
print(f"Number of events: {len(env.df)}")
print(f"Number of actions: {env.n_actions}")
print(f"Number of states: {env.n_states_unabs}")
mean_trace_length = env.df.groupby('ID').size().mean()
print(f"Mean trace length: {mean_trace_length:.2f}")

# ============== Get training log policy results ================
train_df = env.train_df
print("Computing rewards for the Log Policy (baseline)...")

log_policy_rewards = []
log_policy_granted = []
log_policy_lengths = []
log_policy_case_ids = []

for case_id, case_df in train_df.groupby('ID'):
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
            r = env.config.outcome_function(granted, row[env.state_cols_simulation], cum_cost, env.scaler, env.numeric_cols, env.state_cols_simulation)
            reward += r
            break
          
    log_policy_granted.append(int(granted))
    log_policy_lengths.append(len(case_df))
    log_policy_rewards.append(reward)
    log_policy_case_ids.append(case_id)


mean_success_rate = np.mean(log_policy_granted)
mean_case_length = np.mean(log_policy_lengths)
mean_case_reward = np.mean(log_policy_rewards)

print("Mean success rate (training log):", mean_success_rate)
print("Mean case length (training log):", mean_case_length)
print("Mean case reward (training log):", mean_case_reward)


log_policy_results = {
    "case_ids": log_policy_case_ids,
    "rewards": log_policy_rewards,
    "granted": log_policy_granted,
    "lengths": log_policy_lengths,
    "avg_reward": np.mean(log_policy_rewards),
    "success_rate": 100 * np.mean(log_policy_granted),
}

# ============== Offline training ======================

num_episodes = args.num_episodes
num_episodes_test = len(env.test_case_ids)
learning_rate = 0.8
discount_factor = 0.9
stopping_criteria = 100 #max length of generated traces
epsilon_start = 0.1
epsilon_end = 0.01
epsilon_decay = (epsilon_start - epsilon_end) / num_episodes  
epsilon = epsilon_start
cumulative_reward = []
list_len_ep = []
list_granted = []
list_granted_percent = []
n_granted = 0
n_stopping_cases = 0
n_stopping_cases_long = 0
optimal_paths = []
recommended_traces = []
cases_with_impossible_actions = []
episode_runtimes = [] 
Q_max_changes = []
Q_max_changes_percent = []
convergence_threshold = 1e-2   # convergence threshold
stable_steps = 0
needed_stable = 500   # require 50 stable episodes in a row


print(f"Training the agent on {dataset} with {num_episodes} episodes")
print(f"State abstraction function: {state_abstraction}, parameter: {k}, State cols: {env.state_cols}")
#initialize Q-table with zeros
Q_table = np.zeros((env.n_states, env.n_actions))

if action_masking:
      for abs_state in range(env.n_states):
            valid_actions = env.valid_actions_abs.get(abs_state, [])
            impossible_actions = [a for a in range(env.n_actions) if a not in valid_actions]
            Q_table[abs_state, impossible_actions] = -np.inf

start_time = time.time()

#Q-learning implementation 
with tqdm(range(num_episodes), desc="Training Progress", unit="episode") as pbar:
   for episode in pbar:
      episode_start = time.time()
      done, granted = False, False
      reward_episode, len_epis = 0, 0
      Q_table_previous = Q_table
      max_change = 0
      max_change_percent = 0
      case_id = env.train_df[env.train_df['ID'] == np.random.choice(env.train_case_ids)]  
      current_state_unabs, current_state = env.reset(case_id) # get original state and abstract state

      while done == False:

            possible_actions = [action for action in range(env.n_actions) #according to the original environment model/MDP
                                if (env.state_unabs, action) in env.transition_proba and env.transition_proba[(env.state_unabs, action)].sum() > 0]
            
            if len(possible_actions) == 0 :
                  Q_table = Q_table_previous
                  n_stopping_cases +=1
                  break

            if len_epis >= stopping_criteria: 
                  Q_table = Q_table_previous
                  n_stopping_cases_long +=1
                  break

            # Choose action 
            if action_masking: # With action masking (constraining the action space with the actions allowed by the abstract MDP)
                  feasible_actions_amdp = np.where(Q_table[current_state] != -np.inf)[0] # feasible_actions_amdp (AMDP) are not necessarily equal to possible actions (MDP)
                  if len(feasible_actions_amdp)==0 :
                        n_stopping_cases +=1
                        break

                  if np.random.rand() < epsilon: # exploration
                        action = np.random.choice(feasible_actions_amdp) # Choice restricted to feasible actions according to the AMDP
                        
                  else: # Exploitation (select the action with the maximum value)
                        max_q = np.max(Q_table[current_state, feasible_actions_amdp])
                        max_value_indices = np.where(Q_table[current_state, feasible_actions_amdp] == max_q)[0]
                        action = feasible_actions_amdp[np.random.choice(max_value_indices)] # In case there are multiple feasible actions with the same maximum value
                   
            else: # Without action masking
                  if np.random.rand() < epsilon:  # Exploration
                        action = np.random.choice(range(env.n_actions))  
                  else:  # Exploitation (select the action with the maximum value)
                        max_value_indices = np.where(
                              Q_table[current_state, :] == np.max(Q_table[current_state, :])
                        )[0]
                        action = np.random.choice(max_value_indices) # In case there are multiple actions with the same maximum value
                    
                 
            next_state_unabs, next_state, reward, done, granted, _ = env.step(action)
        
            old_q = Q_table[current_state, action]
            #update Q-value using the Bellman Equation
            if done:  # next_state is terminal
                  Q_table[current_state, action] += learning_rate * (reward - Q_table[current_state, action])
            else:
                  Q_table[current_state, action] += learning_rate * (reward + discount_factor * np.max(Q_table[next_state]) - Q_table[current_state, action])

            delta = abs(Q_table[current_state, action] - old_q)
            if old_q != 0:
                delta_percent = abs(Q_table[current_state, action] - old_q) / abs(old_q) 
            else:
                delta_percent = 0.0  # avoid division by zero
            max_change = max(max_change, delta)
            max_change_percent = max(max_change_percent, delta_percent)

            current_state = next_state  
            current_state_unabs = next_state_unabs
            reward_episode += reward
            len_epis += 1
        
            
           
      if granted:
            n_granted +=1
      
      Q_max_changes.append(max_change)
      Q_max_changes_percent.append(max_change_percent)
      cases_with_impossible_actions.append(env.impossible_action_taken)
      cumulative_reward.append(reward_episode)
      list_len_ep.append(len_epis)
      list_granted.append(granted)
      list_granted_percent.append(100*(n_granted/(episode+1)))
      recommended_traces.append(env.current_trace)
      optimal_paths.append(env.current_path)

      episode_end = time.time()
      episode_runtimes.append(episode_end - episode_start)

      epsilon = max(epsilon_end, epsilon - epsilon_decay) # decay epsilon

      if max_change < convergence_threshold:
        stable_steps += 1
      else:
        stable_steps = 0
      if stable_steps == needed_stable:
        print(f"Converged after {episode} episodes.")
        

      # === SAVE CHECKPOINT EVERY 1000 EPISODES ===
      if (episode + 1) % 1000 == 0:
            checkpoint_results = {
                        "dataset": dataset,
                        "state_abstraction": args.state_abstraction,
                        "k": k,
                        "action_masking": action_masking,
                        "Q_table": Q_table,
                        "state_cols": env.state_cols,
                        "state_cols_simulation": env.state_cols_simulation,
                        "n_states": env.n_states,
                        "n_states_unabs": env.n_states_unabs,
                        "abs_state_space": env.all_states,
                        "original_state_space": env.all_states_unabs,
                        "all_state_unabs_index": env.all_state_unabs_index,
                        "all_state_index": env.all_state_index,
                        "unabs_to_abs_state": env.unabs_to_abs_state,
                        "optimal_path_frequencies": optimal_paths[-500:],
                        "recommended_traces": recommended_traces, 
                        "cumulative_reward": cumulative_reward,
                        "len_ep": list_len_ep,
                        "granted": list_granted,
                        "granted_percent": list_granted_percent,
                        "n_granted": n_granted,
                        "n_episode": num_episodes,
                        "cases_with_impossible_actions": cases_with_impossible_actions,
                        "n_stopping_cases": n_stopping_cases, #because of no possible actions
                        "n_stopping_cases_long": n_stopping_cases_long, #because of stopping criteria
                        "train_case_ids": env.train_case_ids,
                        "episode_runtimes": episode_runtimes,
                        "Q_max_changes": Q_max_changes,
                        "Q_max_changes_percent": Q_max_changes_percent,
                        "log_policy_results": log_policy_results}
             
            checkpoint_filename = os.path.join(results_dir, f"{dataset}_{args.state_abstraction}_{k}_{end_file}_training.pkl")
            with open(checkpoint_filename, 'wb') as f:
                  pickle.dump(checkpoint_results, f)
      
      if (episode + 1) % checkpoint_interval == 0: 
            test_env = copy.deepcopy(env)
            
            if dataset == 'SimBank':
                testing_results = online_testing_simbank(test_env, Q_table, num_episodes_test, args)
            else:
                testing_results = run_testing_phase(test_env, Q_table, num_episodes_test, knn_mapping, args)
            testing_filename = os.path.join(results_dir, f"{dataset}_{args.state_abstraction}_{k}_{end_file}_testing.pkl")

            with open(testing_filename, 'wb') as f:
                pickle.dump(testing_results, f)
            print(testing_filename)


      pbar.set_postfix(success=f"{100 * (n_granted / (episode + 1)):.2f}%", len_episode=f"{np.mean(list_len_ep):.1f}, impossible actions taken: {sum(cases_with_impossible_actions)}, stopping cases: {n_stopping_cases+n_stopping_cases_long}")
    
end_time = time.time()
runtime = end_time - start_time 

optimal_paths = optimal_paths[-1000:]

print("=====Training results====")
training_results = {"dataset": dataset,
                    "state_abstraction": args.state_abstraction,
                    "k": k,
                    "Q_table": Q_table,
                    "state_cols": env.state_cols,
                    "state_cols_simulation": env.state_cols_simulation,
                    "n_states": env.n_states,
                    "n_states_unabs": env.n_states_unabs,
                    "abs_state_space": env.all_states,
                    "original_state_space": env.all_states_unabs,
                    "all_state_unabs_index": env.all_state_unabs_index,
                    "all_state_index": env.all_state_index,
                    "unabs_to_abs_state": env.unabs_to_abs_state,
                    "optimal_path_frequencies": optimal_paths,
                    "recommended_traces": recommended_traces, 
                    "cumulative_reward": cumulative_reward,
                    "len_ep": list_len_ep,
                    "granted": list_granted,
                    "granted_percent": list_granted_percent,
                    "n_granted": n_granted,
                    "cases_with_impossible_actions": cases_with_impossible_actions,
                    "n_stopping_cases": n_stopping_cases, #because of no possible actions
                    "n_stopping_cases_long": n_stopping_cases_long, #because of stopping criteria
                    "train_case_ids": env.train_case_ids,
                    "n_episode": num_episodes,
                    "episode_runtimes": episode_runtimes,
                    "runtime": runtime,
                    "Q_max_changes": Q_max_changes,
                    "Q_max_changes_percent": Q_max_changes_percent,
                    "log_policy_results": log_policy_results}
print("Training runtime:", runtime)
print(f"Training success rate: {100 * (n_granted / (episode + 1)):.2f}%")


training_filename = os.path.join(results_dir, f"{dataset}_{args.state_abstraction}_{k}_{end_file}_training.pkl")

with open(training_filename, 'wb') as f:
    pickle.dump(training_results, f)

test_env = copy.deepcopy(env)
if dataset == 'SimBank':
    testing_results = online_testing_simbank(test_env, Q_table, num_episodes_test, args)
else:
    testing_results = run_testing_phase(test_env, Q_table, num_episodes_test, knn_mapping, args)

testing_filename = os.path.join(results_dir, f"{dataset}_{args.state_abstraction}_{k}_{end_file}_testing.pkl")


with open(testing_filename, 'wb') as f:
    pickle.dump(testing_results, f)
print(testing_filename)


