# === Standard library ===
import sys
import os
import time
import math
import pickle
import random
from collections import Counter
from datetime import datetime, timedelta
from itertools import combinations

# === Data manipulation and visualization ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import dok_matrix
from collections import defaultdict

# === Scikit-learn ===
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing
from scipy.spatial.distance import cdist

# === Gym and environments ===
import gymnasium as gym
from gym import spaces

# === Custom utility ===
from scipy.stats import wasserstein_distance_nd

def seen_offline(env, state_unabs_idx, state_idx):
    seen = state_unabs_idx <= env.last_state_unabs_offline
    mapped_to_abs_state = seen or (state_idx <= env.last_state_offline)
    return seen, mapped_to_abs_state

def k_means(df, state_cols, k):

    #apply k-means clustering on the event attributes
    kmeanModel = KMeans(n_clusters=k, random_state=42).fit(df[state_cols])
    df['cluster'] = kmeanModel.labels_
    df.reset_index(drop=True, inplace=True)

    return df, kmeanModel


def flatten_and_convert(state):
        flat = []
        for v in state:
            if isinstance(v, (list, tuple, np.ndarray)):
                flat.extend(flatten_and_convert(v))  # recursive flatten
            elif isinstance(v, (np.str_, str)):
                flat.append(str(v))
            elif isinstance(v, (np.integer, int)):
                flat.append(int(v))
            elif isinstance(v, (np.floating, float)):
                flat.append(float(v))
            else:
                flat.append(v)
        return flat

def convert_to_tuple(path):
    return tuple(tuple(x) if isinstance(x, list) else x for x in path)


