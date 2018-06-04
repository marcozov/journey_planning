# -*- coding: utf-8 -*-
""" Contains utility functions for testing purposes. """

import pandas as pd
import numpy as np
from .planner import Status

def rand_edges(std=1e-10, mean=1e-10):
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
