"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

import torch
import pandas as pd
import numpy as np
#import numba

from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler

DATA = "wikipedia"
OUT_DF = './downstream_data/{}/ds_{}.csv'.format(DATA,DATA)


g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))


down_stream_time = list(np.quantile(g_df.ts, [0.80]))
print(down_stream_time)
d_data = g_df[g_df["ts"]>down_stream_time[0]]

d_data.iloc[:,1:].to_csv(OUT_DF)
