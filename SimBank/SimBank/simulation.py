import warnings
import pandas as pd
from copy import copy, deepcopy
import random
import simpy
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.utils.petri_utils import get_transition_by_name
from .extra_flow_conditions import ExtraFlowConditioner
from .activity_execution import ActivityExecutioner
from . import petri_net_generator
#from extra_flow_conditions import ExtraFlowConditioner
#from activity_execution import ActivityExecutioner
#import petri_net_generator
import numpy as np

class PresProcessGenerator():
    def __init__(self, dataset_params, offline_env, seed=82):
        #GENERAL params
        self.offline_env = offline_env
        self.log_cols = dataset_params["log_cols"]
        self.random_seed = seed
        self.simulation_start = dataset_params["simulation_start"] #datetime object
        #POLICY params
        self.policies_info = dataset_params["policies_info"] #conditions linked to bank policy
        #INTERVENTION params
        self.intervention_info = dataset_params["intervention_info"] #info linked to interventions activities
        self.intervention_info["flat_activities"] = [act for sublist in self.intervention_info["activities"] for act in sublist] #flattened list of intervention activities
        self.get_petri_net()

    warnings.filterwarnings('ignore')


    def find_activity_index(self, target_string): #finds whcih intervention group an activity belongs to
        for index, sublist in enumerate(self.intervention_info["activities"]):
            if target_string in sublist:
                return index
        return -1


    def get_petri_net(self):
        self.net = petri_net_generator.generate_petri_net()


    def simulation_of_events(self, net, initial_marking, n_cases, simulation_state=None, timeout=None):
        env = simpy.Environment()
        env.process(self.setup(env, net, initial_marking, n_cases, simulation_state, timeout))
        env.run()


    def setup(self, env, net, initial_marking, n_cases, simulation_state=None, timeout=None):
        for i in range(0, n_cases):
            # Set random seed for each case if normal simulation (not under each action)
            if self.normal_simulation:
                self.random_seed = self.random_seed + i
                self.random_obj.seed(self.random_seed)
                self.activity_executioner.random_obj = self.random_obj
                self.extra_flow_conditioner.random_obj = deepcopy(self.random_obj)
            # Set simulation state
            if not simulation_state:
                simulation_state = {"trace": [], "net": copy(net), "marking": copy(initial_marking), "parallel_executions": False, "parallel_timestamps": {"HQ": [self.simulation_start, self.simulation_start], "val": [self.simulation_start, self.simulation_start]}, "case_nr": i, "env": env}
            else:
                simulation_state["env"] = env
                simulation_state["net"] = copy(net)
                simulation_state["marking"] = copy(initial_marking)
                simulation_state["case_nr"] = i
            # Do timeout if necessary
            if timeout:
                self.do_timeout(env, timeout)
            # Simulate trace
            proc = env.process(self.simulate_trace(simulation_state))
            yield proc
            # Set simulation start and end
            if not self.generate_under_each_action and i == self.n_cases - 1:
                self.simulation_start, self.simulation_end = self.activity_executioner.set_simulation_end_and_start(simulation_start=self.simulation_start, last_event=self.log[-1])
            self.simulation_state = simulation_state #added by me

    def simulate_trace(self, simulation_state):
        case_num = simulation_state["case_nr"]
        env = simulation_state["env"]
        trace = deepcopy(simulation_state["trace"])
        marking = copy(simulation_state["marking"])
        net = copy(simulation_state["net"])
        parallel_executions = simulation_state["parallel_executions"]
        parallel_timestamps = simulation_state["parallel_timestamps"]

        if self.intervention_info["RCT"]:
            int_timing_list = [1000] * len(self.intervention_info["name"])
        int_enabled_list = [0] * len(self.intervention_info["name"])

        while True:
            # GET PREVIOUS EVENT
            prev_event = trace[-1] if len(trace) > 0 else None
            # BREAK IF NO TRANSITIONS ARE ENABLED (END OF TRACE)
            if (not semantics.enabled_transitions(net, marking)):
                for event in trace:
                    event["outcome"] = trace[-1]["outcome"]
                    self.log.append(event)
                    if self.generate_under_each_action:
                       self.int_points_available  = False
                break

            # GET THE ENABLED ACTIVITIES BASED ON (EXTRA) FLOW CONDITIONS (MOSTLY POLICIES)
            control_flow_enabled_trans = list(semantics.enabled_transitions(net, marking))
            control_flow_enabled_trans = sorted(control_flow_enabled_trans, key=lambda x: x.label)
            all_enabled_trans = self.extra_flow_conditioner.filter_enabled_trans(net, marking, control_flow_enabled_trans, trace, self.policies_info, self.intervention_info)
            
            # GENERATE UNDER EACH ACTION IF ENABLED
            if self.generate_under_each_action:
                all_enabled_trans_cf = self.extra_flow_conditioner.filter_enabled_trans(net, marking, control_flow_enabled_trans, trace, self.policies_info, self.intervention_info, ignore_intervention_policy=True)
                int_activities = [act for act in self.intervention_info["flat_activities"] if act != "do_nothing"]
                enabled_int_activities = [act for act in int_activities if get_transition_by_name(net, act) in all_enabled_trans_cf]
                # Is it an intervention point?
                if len(enabled_int_activities) > 0:
                    first_activity = enabled_int_activities[0]
                    self.current_int_index = self.find_activity_index(first_activity)
                    # Is there a choice for intervention?
                    choice_for_int_unavailable = (len(all_enabled_trans_cf) <= 1 and self.intervention_info["data_impact"][self.current_int_index] == "direct")
                    # If generate under each action is enabled, generate all intervention events
                    if not choice_for_int_unavailable and self.generate_under_each_action:
                        simulation_state = {"trace": deepcopy(trace), "net": copy(net), "marking": copy(marking), "parallel_executions": parallel_executions, "parallel_timestamps": parallel_timestamps, "case_nr": case_num, "env": env}
                        self.generate_all_int_events(self.current_int_index, simulation_state)
                        break
                    
            # CHECK FOR RCT IF ENABLED
            if self.intervention_info["RCT"]:
                int_activities = [act for act in self.intervention_info["flat_activities"] if act != "do_nothing"]
                enabled_int_activities = [act for act in int_activities if get_transition_by_name(net, act) in all_enabled_trans]
                if len(enabled_int_activities) > 0:
                    first_activity = enabled_int_activities[0]
                    self.current_int_index = self.find_activity_index(first_activity)
                    int_enabled_list[self.current_int_index] += 1
                    # Is there a choice for intervention?
                    choice_for_int_unavailable = (len(all_enabled_trans) <= 1 and self.intervention_info["data_impact"][self.current_int_index] == "direct")
                    if (int_enabled_list[self.current_int_index] - 1) != int_timing_list[self.current_int_index] and not choice_for_int_unavailable:
                        all_enabled_trans = [act for act in all_enabled_trans if act.label not in self.intervention_info["flat_activities"]]
                    elif (int_enabled_list[self.current_int_index] -1) == int_timing_list[self.current_int_index] and not choice_for_int_unavailable:
                        all_enabled_trans = [act for act in all_enabled_trans if act.label in self.intervention_info["flat_activities"]]
           
            self.random_obj.shuffle(all_enabled_trans)
            trans = all_enabled_trans[0]
            
            # EXECUTE THE ACTIVITY
            if trans.label is not None and 'ghost' not in trans.label:
                # Set basic attributes
                event = {}
                event["case_nr"] = case_num
                event["activity"] = trans.label

                # Set other attributes
                event = self.activity_executioner.set_event_variables(event, prev_event, intervention_info=self.intervention_info)
                # Set timestamp, elapsed time and timeout env
                event["timestamp"], parallel_executions, parallel_timestamps, timeout = self.activity_executioner.set_event_timestamp(event["activity"], prev_event, env, parallel_executions, parallel_timestamps, self.simulation_start)
                event["elapsed_time"] = ((event["timestamp"] - trace[0]["timestamp"]).total_seconds() / 86400) if len(trace) > 0 else 0
                yield env.timeout(timeout)

                if self.intervention_info["RCT"]:
                    for timing in int_timing_list:
                        if timing == 1000:
                            int_timing_list = self.sample_timing(event)

                trace.append(event)
            
            marking = semantics.execute(trans, net, marking)

    
    def generate_all_int_events(self, int_index, simulation_state):
        log_per_action_list = []
        self.simulation_state_list = []
        self.timeout_list = []
        log_current_state = []
        self.activity_executioner_list = []
        random_state = self.random_obj.getstate()
        
        for event in simulation_state["trace"]:
            event["outcome"] = simulation_state["trace"][-1]["outcome"]
            log_current_state.append(event)
        for action in self.intervention_info["actions"][int_index]:
            #Set each a new random state, to stay consistent (same changes of event variables under each action)
            random_obj = deepcopy(self.random_obj)
            activity_executioner = ActivityExecutioner(random_obj)
            activity_executioner.set_state(random_state)
            self.activity_executioner_list.append(activity_executioner)
            #Generate
            action_log = deepcopy(log_current_state)
            other_actions = [act for act in self.intervention_info["actions"][int_index] if act != action]
            int_event, action_simulation_state, timeout_to_be_done = self.generate_one_int_event(int_index, simulation_state, action, other_actions, activity_executioner, random_obj)
            #Append
            action_log.append(int_event)
            log_per_action_list.append(action_log)
            self.simulation_state_list.append(action_simulation_state)
            self.timeout_list.append(timeout_to_be_done)
        self.log_per_action_list = log_per_action_list


    def generate_one_int_event(self, int_index, simulation_state, action, other_actions, activity_executioner, random_obj):
        case_num = simulation_state["case_nr"]
        env = simulation_state["env"]
        trace = deepcopy(simulation_state["trace"])
        marking = copy(simulation_state["marking"])
        net = copy(simulation_state["net"])
        parallel_executions = deepcopy(simulation_state["parallel_executions"])
        parallel_timestamps = deepcopy(simulation_state["parallel_timestamps"])

        prev_event = trace[-1] if len(trace) > 0 else None

        control_flow_enabled_trans = list(semantics.enabled_transitions(net, marking))
        control_flow_enabled_trans = sorted(control_flow_enabled_trans, key=lambda x: x.label)
        all_enabled_trans = self.extra_flow_conditioner.filter_enabled_trans(net, marking, control_flow_enabled_trans, trace, self.policies_info, self.intervention_info)

        if self.intervention_info["data_impact"][int_index] == "direct":
            if action != "do_nothing":
                all_enabled_trans = self.extra_flow_conditioner.filter_enabled_trans(net, marking, control_flow_enabled_trans, trace, self.policies_info, self.intervention_info, action_to_be_taken=action, ignore_intervention_policy=True)
            else:
                all_enabled_trans = self.extra_flow_conditioner.filter_enabled_trans(net, marking, control_flow_enabled_trans, trace, self.policies_info, self.intervention_info, ignore_intervention_policy=True)
                transition_action_list = [get_transition_by_name(net, other_act) for other_act in other_actions]
                all_enabled_trans = [act for act in all_enabled_trans if act not in transition_action_list]

        random_obj.shuffle(all_enabled_trans)
        trans = all_enabled_trans[0]
        
        # EXECUTE THE ACTIVITY
        if trans.label is not None and 'ghost' not in trans.label:
            # Set basic attributes
            event = {}
            event["case_nr"] = case_num
            event["activity"] = trans.label

            # Set other attributes
            if self.intervention_info["data_impact"][int_index] == "indirect":
                event = activity_executioner.set_event_variables(event, prev_event, action, self.intervention_info)
            else:
                event = activity_executioner.set_event_variables(event, prev_event, intervention_info=self.intervention_info)

            # Set timestamp, elapsed time and timeout env
            event["timestamp"], new_parallel_executions, new_parallel_timestamps, timeout_to_be_done = activity_executioner.set_event_timestamp(event["activity"], prev_event, env, parallel_executions, parallel_timestamps, self.simulation_start)
            event["elapsed_time"] = ((event["timestamp"] - trace[0]["timestamp"]).total_seconds() / 86400) if len(trace) > 0 else 0
            # WE NEED TO REMEMBER THE TIMEOUT FOR THE REST OF THE CASE GENERATION
        trace.append(event)
        marking = semantics.execute(trans, net, marking)
        new_simulation_state = {"trace": deepcopy(trace), "net": copy(net), "marking": copy(marking), "parallel_executions": new_parallel_executions, "parallel_timestamps": new_parallel_timestamps, "case_nr": case_num, "env": env}
        return event, new_simulation_state, timeout_to_be_done
    

    def do_timeout(self, env, timeout):
        yield env.timeout(timeout)


    # INFERENCE IS ALWAYS ONE CASE
    def end_simulation_inference(self):
        return deepcopy(self.log)


    def continue_simulation_inference(self, action_index):
        simulation_state = self.simulation_state_list[action_index]
        timeout_to_be_done = self.timeout_list[action_index]
        self.activity_executioner = self.activity_executioner_list[action_index]
        self.random_obj = self.activity_executioner.random_obj
        self.log_per_action_list = []
        self.simulation_of_events(net=simulation_state["net"], initial_marking=simulation_state["marking"], n_cases=self.n_cases, simulation_state=simulation_state, timeout = timeout_to_be_done)
        return deepcopy(self.log_per_action_list)
    

    def start_simulation_inference(self, seed_to_add=0):
        self.normal_simulation = False
        self.random_seed = self.random_seed + seed_to_add
        self.random_obj = random.Random(self.random_seed)
        self.generate_under_each_action = True
        self.int_points_available = True
        self.n_cases = 1
        self.current_int_index = None
        self.activity_executioner = ActivityExecutioner(self.random_obj)
        self.extra_flow_conditioner = ExtraFlowConditioner(deepcopy(self.random_obj))
        self.log_per_action_list = []
        self.log = []
        self.simulation_of_events(self.net, self.net.initial_marking, n_cases=self.n_cases)
        return deepcopy(self.log_per_action_list)


    # FOR NORMAL YOU CAN SPECIFY THE N_CASES (create a full log)
    def run_simulation_normal(self, n_cases, seed_to_add=0):
        self.normal_simulation = True
        self.random_seed = self.random_seed + seed_to_add
        self.random_obj = random.Random(self.random_seed)
        self.random_obj_for_timing = random.Random(self.random_seed)
        self.generate_under_each_action = False
        self.int_points_available = True
        self.n_cases = n_cases
        self.current_int_index = None
        self.activity_executioner = ActivityExecutioner(self.random_obj)
        self.extra_flow_conditioner = ExtraFlowConditioner(deepcopy(self.random_obj))
        self.log = []
        self.simulation_of_events(self.net, self.net.initial_marking, n_cases=self.n_cases)
        self.log = pd.DataFrame(self.log)
        return deepcopy(self.log)
    

    def sample_with_weighted_probability(self, integer_set, decrease_rate):
        weights = [1/(i ** decrease_rate) for i in integer_set]
        return weights
    

    def sample_timing(self, event=None):
        timing_list = []
        for int_index in range(len(self.intervention_info["name"])):
            lower_bound = 0
            upper_bound = self.intervention_info["action_depth"][int_index] - 1
            if "do_nothing" in self.intervention_info["activities"][int_index]:
                upper_bound += 1
            
            timing = self.random_obj_for_timing.randint(lower_bound, upper_bound)
            if self.intervention_info["name"][int_index] == "time_contact_HQ":
                timing = timing*2
            timing_list.append(timing)
        return timing_list
    
    def reset_for_online(self, seed_to_add=0):
            """
            Initialize a simulation with a single starting event.
            Returns:
                initial_event: dict of the first simulated event
                simulation_state: dict containing environment, net, marking, etc. for stepping
            """
            self.normal_simulation = False
            self.random_seed = self.random_seed + seed_to_add
            self.random_obj = random.Random(self.random_seed)
            self.generate_under_each_action = True
            self.int_points_available = True
            self.n_cases = 1
            self.current_int_index = None

            # Initialize ActivityExecutioner and ExtraFlowConditioner
            self.activity_executioner = ActivityExecutioner(self.random_obj)
            self.extra_flow_conditioner = ExtraFlowConditioner(deepcopy(self.random_obj))

            # Initialize simulation state
            env = simpy.Environment()
            simulation_state = {
                "trace": [],
                "net": copy(self.net),
                "marking": copy(self.net.initial_marking),
                "parallel_executions": False,
                "parallel_timestamps": {"HQ": [self.simulation_start, self.simulation_start],
                                        "val": [self.simulation_start, self.simulation_start]},
                "case_nr": 0,
                "env": env,
                "ignore_policy_next_step": False  # new flag
            }

            # Get first enabled transition
            control_flow_enabled_trans = list(semantics.enabled_transitions(simulation_state["net"], simulation_state["marking"]))
            control_flow_enabled_trans = sorted(control_flow_enabled_trans, key=lambda x: x.label)
            all_enabled_trans = self.extra_flow_conditioner.filter_enabled_trans(
                simulation_state["net"], 
                simulation_state["marking"], 
                control_flow_enabled_trans, 
                simulation_state["trace"], 
                self.policies_info, 
                self.intervention_info
            )

            if len(all_enabled_trans) == 0:
                raise RuntimeError("No transitions enabled at the start of the simulation.")

            # Pick the first transition randomly
            self.random_obj.shuffle(all_enabled_trans)
            trans = all_enabled_trans[0]

            # Execute the first event
            event = {}
            event["case_nr"] = 0
            event["activity"] = trans.label
            prev_event = None
            event = self.activity_executioner.set_event_variables(event, prev_event, intervention_info=self.intervention_info)
            event["timestamp"], parallel_executions, parallel_timestamps, timeout = self.activity_executioner.set_event_timestamp(
                event["activity"], prev_event, env, simulation_state["parallel_executions"], simulation_state["parallel_timestamps"], self.simulation_start
            )
            event["elapsed_time"] = 0
            event['receive_acceptance'] = 0
            event['receive_refusal'] = 0
            simulation_state["trace"].append(event)

            # Update marking
            simulation_state["marking"] = semantics.execute(trans, simulation_state["net"], simulation_state["marking"])
            simulation_state["parallel_executions"] = parallel_executions
            simulation_state["parallel_timestamps"] = parallel_timestamps

            self.cum_costs = 0
            self.outcome = 0
            self.reward = 0.0
            self.current_trace = []
            self.current_path = []
            
            self.current_path.append((event))
            self.illegal_action_count = 0
            self.current_trace_illegal = False
            self.impossible_action_taken = False
            
            event_scaled = copy(event)
            # Fill NaNs with 0
            event_scaled = {k: (0 if pd.isna(v) else v) for k, v in event_scaled.items()}
            event_scaled['last_action'] = self.offline_env.activity_index[event_scaled['activity']]

            # Keep only the features used in scaler
            event_df = pd.DataFrame([event_scaled])
            event_df[self.offline_env.numeric_cols] = self.offline_env.scaler.transform(event_df[self.offline_env.numeric_cols])
            event_scaled = event_df.iloc[0].to_dict()

            return event, simulation_state, event_scaled
    
    def step_for_online(self, simulation_state, action):
        """
        Take one step in the online simulation where the agent can choose an action
        after every visible event.

        Args:
            simulation_state: dict (trace, net, marking, env, parallel_executions, parallel_timestamps, case_nr)
            action: str, label of chosen visible activity

        Returns:
            event, simulation_state, reward, granted, done
        """
        done = False
        granted = False
        reward = 0.0
        impossible_action = False
        # Ensure log exists (simulate_trace references it)
        if not hasattr(self, "log"):
            self.log = []

        # Defensive local copies (we'll write results back into simulation_state)
        case_num = simulation_state["case_nr"]
        env = simulation_state["env"]
        trace = deepcopy(simulation_state["trace"])
        marking = copy(simulation_state["marking"])
        net = copy(simulation_state["net"])
        parallel_executions = simulation_state["parallel_executions"]
        parallel_timestamps = simulation_state["parallel_timestamps"]
        prev_event = trace[-1] if len(trace) > 0 else None

        # Helper: auto-fire ghost/invisible transitions until a visible transition is available
        while True:
            if (not semantics.enabled_transitions(net, marking)):
                    for event in trace:
                        event["outcome"] = trace[-1]["outcome"]
                        self.log.append(event)
                        if self.generate_under_each_action:
                            self.int_points_available  = False
                    done = True
                    break
            
            control_flow_enabled = list(semantics.enabled_transitions(net, marking))
            control_flow_enabled = sorted(control_flow_enabled, key=lambda x: x.label)

            # Determine if we are at an intervention point
            int_activities = [act for act in self.intervention_info["flat_activities"] if act != "do_nothing"]
            enabled_int_activities = [
                act for act in int_activities
                if get_transition_by_name(net, act) in control_flow_enabled
            ]

            if enabled_int_activities:
                all_enabled = self.extra_flow_conditioner.filter_enabled_trans(
                    net, marking, control_flow_enabled, trace, self.policies_info, self.intervention_info,
                    ignore_intervention_policy=True  # <-- bypass default policy filtering
                )

            else: 
                all_enabled = self.extra_flow_conditioner.filter_enabled_trans(
                    net, marking, control_flow_enabled, trace, self.policies_info,
                    self.intervention_info
                )

            # visible transitions are those with a label and not containing 'ghost'
            visible = [t for t in all_enabled if (t.label is not None and 'ghost' not in str(t.label))]

            if visible :
                # we have visible transitions available; stop auto-firing ghosts
                break


            # Fire the first ghost transition (routing/invisible)
            ghost_trans = [t for t in all_enabled if (t.label is not None and 'ghost' in str(t.label))]

            t = ghost_trans[0]
        
            # execute ghost transition to advance marking
            marking = semantics.execute(t, net, marking)
        
        
        # Now `visible` contains the currently allowed visible transitions (based on marking & filters)
        visible_labels = [t.label for t in visible]
        #print(f"Visible{visible_labels}")
        if action in [0.07, 0.08, 0.09]:
            interest_rate = action
            action = 'calculate_offer'

        # If chosen action is not among visible labels -> invalid action
        if action not in visible_labels:
            granted = False
            done = True
            event = None
            event_scaled = None
            reward = -self.offline_env.penalty_illegal
            self.impossible_action_taken = True
            impossible_action = True
            self.current_trace = getattr(self, "current_trace", [])
            self.current_path = getattr(self, "current_path", [])
            self.current_trace.append(action)
            self.current_path.append((action, event))
            return event, simulation_state, event_scaled, reward, granted, done, impossible_action

        # Find transition object for the chosen action
        trans = get_transition_by_name(net, action)

        # Determine if action belongs to an intervention group
        int_index = self.find_activity_index(action)

        # If direct intervention, use generate_one_int_event to preserve intervention semantics
        if int_index >= 0 and self.intervention_info["data_impact"][int_index] == "direct":
            # prepare other_actions list
            other_actions = [a for a in self.intervention_info["actions"][int_index] if a != action]
            # call existing routine (preserves random state via passed activity_executioner/random_obj)
            event, new_sim_state, timeout_to_be_done = self.generate_one_int_event(
                int_index=int_index,
                simulation_state={"trace": deepcopy(trace), "net": copy(net), "marking": copy(marking),
                                "parallel_executions": parallel_executions, "parallel_timestamps": parallel_timestamps,
                                "case_nr": case_num, "env": env},
                action=action,
                other_actions=other_actions,
                activity_executioner=self.activity_executioner,
                random_obj=self.random_obj
            )
            # update simulation_state from returned new_sim_state
            simulation_state.update(new_sim_state)
            # ensure local refs reflect updates
            trace = deepcopy(simulation_state["trace"])
            net = simulation_state["net"]
            marking = simulation_state["marking"]
            parallel_executions = simulation_state["parallel_executions"]
            parallel_timestamps = simulation_state["parallel_timestamps"]
        else:
            # Non-intervention or indirect intervention: produce event via activity_executioner and execute transition
            event = {"case_nr": case_num, "activity": action}
            # For indirect interventions, set_event_variables takes (event, prev_event, action, intervention_info)
            if int_index >= 0 and self.intervention_info["data_impact"][int_index] == "indirect":
                #print('indirect')
                event = self.activity_executioner.set_event_variables(event, prev_event, interest_rate, self.intervention_info)
                #print(event)
            else:
                event = self.activity_executioner.set_event_variables(event, prev_event, intervention_info=self.intervention_info)

            # timestamp and parallel updates
            event["timestamp"], new_parallel_executions, new_parallel_timestamps, timeout = self.activity_executioner.set_event_timestamp(
                action, prev_event, env, parallel_executions, parallel_timestamps, self.simulation_start
            )
            event["elapsed_time"] = ((event["timestamp"] - trace[0]["timestamp"]).total_seconds() / 86400) if len(trace) > 0 else 0

            # append and execute transition
            trace.append(event)
            marking = semantics.execute(trans, net, marking)

            # write updated pieces back into simulation_state
            simulation_state["trace"] = deepcopy(trace)
            simulation_state["net"] = net
            simulation_state["marking"] = copy(marking)
            simulation_state["parallel_executions"] = new_parallel_executions
            simulation_state["parallel_timestamps"] = new_parallel_timestamps

        # Determine stochastic outcome
        event['receive_acceptance'] = 0
        event['receive_refusal'] = 0
        # Record trace & path info on the generator object
        self.current_trace = getattr(self, "current_trace", [])
        self.current_path = getattr(self, "current_path", [])
        self.current_trace.append(action)
        self.current_path.append((action, event))

        # Determine if stochastic outcome transition (after calculating offer): 
        if (semantics.enabled_transitions(net, marking)):
            control_flow_enabled = list(semantics.enabled_transitions(net, marking))
            control_flow_enabled = sorted(control_flow_enabled, key=lambda x: x.label)

            # Determine if we are at an intervention point
            int_activities = [act for act in self.intervention_info["flat_activities"] if act != "do_nothing"]
            enabled_int_activities = [
                act for act in int_activities
                if get_transition_by_name(net, act) in control_flow_enabled
            ]

            if enabled_int_activities:
                all_enabled = self.extra_flow_conditioner.filter_enabled_trans(
                    net, marking, control_flow_enabled, trace, self.policies_info, self.intervention_info,
                    ignore_intervention_policy=True  # <-- bypass default policy filtering
                )

            else: 
                all_enabled = self.extra_flow_conditioner.filter_enabled_trans(
                    net, marking, control_flow_enabled, trace, self.policies_info,
                    self.intervention_info
                )

            outcome_trans = [t for t in all_enabled if t.label in ["receive_acceptance", "receive_refusal"]]
            if outcome_trans:
                outcome_labels = [t.label for t in outcome_trans]
                t = outcome_trans[0]
                action = t.label
                print(f"Auto-fired stochastic outcome: {t.label}")
                # Record trace & path info on the generator object
                self.current_trace = getattr(self, "current_trace", [])
                self.current_path = getattr(self, "current_path", [])
                self.current_trace.append(action)
                self.current_path.append((action, event))
     
        if action in ['receive_acceptance', 'cancel_application', 'receive_refusal']:
            done = True     
            if action == 'receive_acceptance':
                event['receive_acceptance'] = 1
            elif action == 'receive_refusal':
                event['receive_refusal'] = 1
        granted = (action == "receive_acceptance")

        
        # Normalize event values, add last_action, scale numeric cols
        event_scaled = None
        if event is not None:
            event_scaled = copy(event)
            event_scaled = {k: (0 if pd.isna(v) else v) for k, v in event_scaled.items()}
            if not done:
                if event_scaled["activity"]!='calculate_offer':
                    event_scaled['last_action'] = self.offline_env.activity_index[event_scaled['activity']]
                else:
                    event_scaled['last_action'] = self.offline_env.activity_index[interest_rate]

            elif done:
                event_scaled['last_action'] = -1
            event_df = pd.DataFrame([event_scaled])
            event_df[self.offline_env.numeric_cols] = self.offline_env.scaler.transform(event_df[self.offline_env.numeric_cols])
            event_scaled = event_df.iloc[0].to_dict()

        # Compute reward
        if done:
            if not self.offline_env.binary_outcome:
                #print('computing non-binary reward')
                reward = np.float32(self.offline_env.config.outcome_function(
                    granted, event_scaled, event_scaled['cum_cost'],
                    self.offline_env.scaler, self.offline_env.numeric_cols, self.offline_env.state_cols_simulation))
            else:
                #print('computing binary reward')
                reward = np.float32(self.offline_env.config.binary_outcome_function(granted))
            event_scaled = None
        
        else:
            reward = 0.0

        # persist simulation_state and return
        simulation_state["trace"] = deepcopy(trace)
        simulation_state["net"] = copy(net)
        simulation_state["marking"] = copy(marking)
        simulation_state["parallel_executions"] = parallel_executions
        simulation_state["parallel_timestamps"] = parallel_timestamps

        return event, simulation_state, event_scaled, reward, granted, done, impossible_action
