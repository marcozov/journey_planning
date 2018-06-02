# -*- coding: utf-8 -*-
""" Contains utility functions for testing purposes. """

import pandas as pd
import numpy as np
from scipy.stats import gamma

from .planner import Status

def toy_schedule():
    edges = pd.DataFrame(
        columns=[
            'trip_id', 
            'dep_node', 'arr_node', 
            'dep_day', 'arr_day', 
            'dep_time', 'arr_time', 
            'distribution', 
            'prob', 
            'label', 
            'prev_edge'
        ]
    )
    
    dist = gamma(4.659833415701387, -46.000000000100002, 25.816068623767663)
    dist.__str__ = lambda self: "gamma"
    dist.__repr__ = lambda self: "gamma"
    default = [dist, np.NaN, Status.Unvisited, np.NaN] # last 4 columns
    
    t = pd.Timedelta
    
    # create toy dataset
    edges.loc[edges.shape[0]] = ["1", "node1", "node2", 0, 0, t("10:05:00"), t("10:15:00"), *default]
    edges.loc[edges.shape[0]] = ["1", "node2", "node4", 0, 0, t("10:16:00"), t("10:25:00"), *default] 
    edges.loc[edges.shape[0]] = ["1", "node4", "node5", 0, 0, t("10:26:00"), t("10:40:00"), *default] 
    
    edges.loc[edges.shape[0]] = ["21", "node1", "node2", 0, 0, t("10:03:00"), t("10:13:00"), *default] 
    edges.loc[edges.shape[0]] = ["22", "node2", "node4", 0, 0, t("10:14:00"), t("10:23:00"), *default] 
    edges.loc[edges.shape[0]] = ["23", "node4", "node5", 0, 0, t("10:24:00"), t("10:38:00"), *default] 
    
    edges.loc[edges.shape[0]] = ["3", "node1", "node3", 0, 0, t("10:10:00"), t("10:15:00"), *default] 
    edges.loc[edges.shape[0]] = ["3", "node3", "node5", 0, 0, t("10:16:00"), t("10:41:00"), *default]
    
    edges.loc[edges.shape[0]] = ["4", "node3", "node4", 0, 0, t("10:19:00"), t("10:22:00"), *default]
    
    return edges


def toy_schedule_old(std=1e-10, mean=1e-10):
    edges = pd.DataFrame(
        columns=[
            "dep_name", "arr_name", 
            "dep_time", "arr_time", 
            "trip_id", 
            "prob", 
            "label", 
            "mean_delay", 
            "std_delay",
            "prev_edge"
        ]
    )
    
    # create toy dataset
    edges.loc[edges.shape[0]] = ["name1", "name2", 0.5, 2.0, -2, np.NaN, Status.Unvisited, mean, std, np.NaN]
    edges.loc[edges.shape[0]] = ["name1", "name2", 0.1, 1.5, 1, np.NaN, Status.Unvisited, mean, std, np.NaN]

    edges.loc[edges.shape[0]] = ["name2", "name4", 3.0, 4.0, 4, np.NaN, Status.Unvisited, mean, std, np.NaN]
    edges.loc[edges.shape[0]] = ["name2", "name4", 2.0, 3.0, 3, np.NaN, Status.Unvisited, mean, std, np.NaN]

    edges.loc[edges.shape[0]] = ["name4", "name5", 5.0, 7.0, 6, np.NaN, Status.Unvisited, mean, std, np.NaN]
    edges.loc[edges.shape[0]] = ["name4", "name5", 4.0, 6.0, 5, np.NaN, Status.Unvisited, mean, std, np.NaN]

    edges.loc[edges.shape[0]] = ["name1", "name3", 0.5, 1.0, 7, np.NaN, Status.Unvisited, mean, std, np.NaN]

    edges.loc[edges.shape[0]] = ["name3", "name5", 5.0, 8.0, 8, np.NaN, Status.Unvisited, mean, std, np.NaN]
    edges.loc[edges.shape[0]] = ["name3", "name5", 6.0, 9.0, 9, np.NaN, Status.Unvisited, mean, std, np.NaN]
    
    edges.loc[edges.shape[0]] = ["name3", "name4", 2.5, 2.5, 10, np.NaN, Status.Unvisited, mean, std, np.NaN]

    return edges

