
from MDP_functions import *
import argparse
import math
from sklearn.preprocessing import StandardScaler
from testing import *
from utils import *
import copy
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import time
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
parser.add_argument('--state_abstraction', type=str, default='False', help="State abstraction method: 'False', 'structural', 'last_action', 'k_means', or 'k_means_features'")
parser.add_argument('--k', type=int, default=0, help="Number of clusters if using k-means abstraction")
parser.add_argument('--c', type=float, default=0.6, help="Discount factor")
parser.add_argument('--binary_outcome', type=str2bool, default=False, help="Reward based on binary outcome (True) or profitability (False)")
parser.add_argument('--pair_sample_ratio', type=float, default=0.1, help="Pair sample ratio")

args = parser.parse_args()
dataset = args.dataset
state_abstraction = args.state_abstraction
k = args.k
c = args.c

binary_outcome = args.binary_outcome
if state_abstraction in ['False-knn', "structural-knn", "action-set-knn", 'action-count-knn']:
    state_abstraction = state_abstraction.replace("-knn", "")
    knn_mapping = True
else:
    knn_mapping = False

if state_abstraction == 'False':
    state_abstraction = False
    k = None
elif state_abstraction in ['contextual', 'structural', 'action-count','action-set']:
    k = None
else:
    pass


def bisimulation_distance(env, pair_sample_ratio, c):

    def get_transitions_and_rewards(current_state, action, transition_proba):
        transition = transition_proba.get((current_state, action), None)
        if transition is None:
            return None, None, None  
        
        prob_vector = transition.toarray().flatten()
        next_states = np.arange(len(prob_vector))
        nonzero_mask_i = prob_vector > 0
        prob_vector = prob_vector[nonzero_mask_i]
        next_states = next_states[nonzero_mask_i]
    
        rewards = []
        for next_state_idx in next_states: #compute r_t+1 for each observed s_t+1

                next_state_dict = env.all_states_unabs.iloc[next_state_idx].to_dict()
                action_name = list(env.activity_index.keys())[list(env.activity_index.values()).index(action)]
                cost = env.costs_dic.get(action_name, 0)
                
                done, granted = env.config.is_terminal_successful_state(next_state_dict)
                if done: 
                    r = np.float32(
                        env.config.outcome_function(
                            granted,  
                            next_state_dict,
                            cost,
                            env.scaler,
                            env.numeric_cols,
                            env.state_cols_simulation
                        )
                    )
                else:
                    r = -cost
                
                rewards.append(r)

        rewards = np.array(rewards)
        
        return next_states, prob_vector, rewards

    encoded_state_array = np.array(env.all_states_unabs.values) #scaled state features
    # === Precompute transitions ===
    print("Precomputing transitions s_t, a_t, s_t+1, r_t+1...")
    precomputed_transitions = {}
    all_rewards = []

    for s in tqdm(range(env.n_states_unabs), desc="States"):
        for a in range(env.n_actions):
            result = get_transitions_and_rewards(s, a, env.transition_proba) 
            if not any(x is None for x in result):
                next_states, probs, rewards = result
                precomputed_transitions[(s, a)] = (next_states, probs, rewards) 
                all_rewards.extend(rewards)

    # === Standard scaling: (r - mean) / std ===
    all_rewards = np.array(all_rewards)
    '''mean_r = all_rewards.mean()
    std_r = all_rewards.std()
    all_rewards_scaled = (all_rewards - mean_r) / std_r
    scaled_penalty = (-env.penalty_illegal-mean_r)/std_r

    # Replace original rewards in precomputed_transitions
    idx = 0
    for key in precomputed_transitions:
        next_states, probs, rewards = precomputed_transitions[key]
        n = len(rewards)
        precomputed_transitions[key] = (next_states, probs, all_rewards_scaled[idx:idx+n])
        idx += n

    print("All rewards standard-scaled.")'''

    # === Group unabstracted states by abstracted state ===
    abs_to_unabs = {}
    for unabs_state, abs_state in env.unabs_to_abs_state.items():
        if abs_state is not None:
            abs_to_unabs.setdefault(abs_state, []).append(unabs_state)

    # ===== Computing bisimulation distance ============
    total_bisimilarity_distance = 0
    total_weight = 0  
    bisim_distances = dict()

    with tqdm(abs_to_unabs.items(), desc="Computing cluster distances", unit="cluster") as pbar:
            for abs_state, unabs_states in pbar:
                if len(unabs_states) <2: # no pairs in the abstract state to compute the bisimulation distance
                    continue
                indices = [env.all_state_unabs_index[s] for s in unabs_states if s in env.all_state_unabs_index]
                all_pairs = list(combinations(indices, 2))

                # sample a subset of pairs
                n_sample = min(len(all_pairs), max(1, int(len(all_pairs) * pair_sample_ratio)))
                sampled_pairs = random.sample(all_pairs, n_sample)
            
                bisimilarity_sum = 0
                state_size = len(indices)  #cluster size

                for idx_i, idx_j in sampled_pairs:
                    s_i = idx_i
                    s_j = idx_j

                    # Get all possible actions
                    possible_actions_i = [a for a in range(env.n_actions)
                                        if (s_i, a) in precomputed_transitions]
                    possible_actions_j = [a for a in range(env.n_actions)
                                        if (s_j, a) in precomputed_transitions]
                    all_possible_actions = set(possible_actions_i) | set(possible_actions_j) # union the two sets
                
                    max_action_dist = 0
                    for action in all_possible_actions: 
                            
                            result_i = precomputed_transitions.get((s_i, action), ([s_i], [1.0], np.array([-env.penalty_illegal]))) #if action not possible in s_i, receive penalty
                            result_j = precomputed_transitions.get((s_j, action), ([s_j], [1.0], np.array([-env.penalty_illegal])))

                            next_vecs_i, prob_i, r_i = result_i
                            next_vecs_j, prob_j, r_j = result_j

                            if isinstance(next_vecs_i[0], (int, np.integer)): #check if state idx is not None
                                next_vecs_i = encoded_state_array[next_vecs_i]
                            if isinstance(next_vecs_j[0], (int, np.integer)): #check if state idx is not None
                                next_vecs_j = encoded_state_array[next_vecs_j]

                            exp_r_i = float(np.dot(prob_i, r_i))
                            exp_r_j = float(np.dot(prob_j, r_j))
                            reward_diff = abs(exp_r_i - exp_r_j)
                            trans_dist = wasserstein_distance_nd(u_values=next_vecs_i, v_values=next_vecs_j, u_weights=prob_i, v_weights=prob_j)
                            action_dist = (1-c)*reward_diff + (c*trans_dist)
                            max_action_dist = max(max_action_dist, action_dist)
                            
                    bisimilarity_sum += max_action_dist
                    if abs_state not in bisim_distances:
                        bisim_distances[abs_state] = dict()
                                    
                    bisim_distances[abs_state][(idx_i, idx_j)] = max_action_dist

                #compute the average bisimilarity distance for this cluster
                #average distance for the cluster
                avg_bisimilarity_distance = bisimilarity_sum / len(sampled_pairs)

                #update total weighted bisimilarity distance
                total_bisimilarity_distance += avg_bisimilarity_distance * state_size #with size of cluster
                total_weight += state_size

                pbar.set_postfix(avg_bisimilarity_distance=f"{avg_bisimilarity_distance:.2f}")

    #compute the total weighted average bisimilarity distance
    weighted_avg_bisimilarity_distance = total_bisimilarity_distance / total_weight
    print(f"Weighted Bisimilarity Distance of {env.state_abstraction} with {env.n_states} blocks: {weighted_avg_bisimilarity_distance:.2f}")
    results = {
        'weighted_avg_bisimilarity_distance': weighted_avg_bisimilarity_distance,
        'bisim_distances': bisim_distances,
        'n_states': env.n_states, 
        'ratio': pair_sample_ratio
    }
    return results

script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.abspath(os.path.join(script_dir, '..', 'quality_measures_discount'))
os.makedirs(results_dir, exist_ok=True)

# ========== Create environment from event log ===========
env = Environment(dataset, k, state_abstraction, knn_mapping, binary_outcome)

results = {}

print("=====Post-training quality analysis====")
n_nodes, n_edges, abs_transitions_model = measure_simplicity(env)
results['n_nodes'] = n_nodes
results['n_edges'] = n_edges
results['state_abstraction'] = args.state_abstraction
results['k'] = k

filename = os.path.join(results_dir, f"{dataset}_{args.state_abstraction}_{k}_quality_measures.pkl")
with open(filename, 'wb') as f:
    pickle.dump(results, f)
print(filename)

pair_sample_ratio = args.pair_sample_ratio  # sample from of all possible pairs

start_time = time.time()
if env.state_abstraction == False: # does not make sense to compute bisim distance in unabstracted MDP because no state clusters
    bisimulation_dist_results = {
        'weighted_avg_bisimilarity_distance': 0,
        'bisim_distances': [],
        'n_states': env.n_states, 
        'ratio': pair_sample_ratio
    }
else:
    bisimulation_dist_results = bisimulation_distance(env, pair_sample_ratio, c)
end_time = time.time()

results['bisimulation_distance'] = bisimulation_dist_results
results['state_abstraction'] = args.state_abstraction
results['k'] = k
results['runtime_bisim'] = end_time - start_time
results['c'] = c
filename = os.path.join(results_dir, f"{dataset}_{args.state_abstraction}_{k}_bisimulation_distance_{pair_sample_ratio}_{c}.pkl")
with open(filename, 'wb') as f:
    pickle.dump(results, f)
print(filename)

