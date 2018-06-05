# -*- coding: utf-8 -*-

""" Contains classes and functions used for the robust journey planning. """

from scipy.stats import gamma
import pandas as pd
import numpy as np

# TODO change "0000" to "walk"
class Status():
    """ 
    Enum specifying the status of an edge:
    - Unvisited: you have not considered this edge yet
    - Visited: you considered this edge but have not explored any options at the arrival node.
    - Expanded: you considered this edge and also explored the options at the arrival node.
    
    Notice that both visited and explored edges could be improved by finding anothe path which increase the overall probability.
    """
    Unvisited = "Unvisited"
    Visited = "Visited"
    Expanded = "Expanded"
    
    from scipy.stats import gamma    
    
class Planner():
    """ Class used to compute the journey plan. """
    prob_one = 0.99 # approximation of probability 1, used to prune the choices
    
    def __init__(self, edges, reverse=False):
        """ 
        Input:
        - edges: DataFrame indicating all the edges (edges) between nodes.
        - reverse: boolean indicating wether the graph has to be explored in a direct of reversed mode, i.e. if the exploration starts from the departure node (False) or from the arrival node (True).
        """
        self.all_edges = edges
        
        self.nowalking_edges_backup = edges[edges.trip_id != "0000"] # they will be filtered every time the algorithm starts
#         self.edges = self.nowalking_edges_backup
        
        self.walking_edges_backup = edges[edges.trip_id == "0000"] # keep separately the walking edges
        
        self.reverse = reverse
    
#     def clear(self):
#         """ Clear the structure so to restart the computation of the best path. """
#         self.nowalking_edges_backup.prob = np.NaN
#         self.nowalking_edges_backup.label = Status.Unvisited
#         self.nowalking_edges_backup.prev_edge = np.NaN
    
    def compute_plan(self, departure_node, arrival_node, time, treshold):
        """ Given the names of the departure and arrival nodes, a treshold and the time (which may refer either to the departure or arrival time depending on self.reverse), computes the edges above the treshold. """
        assert treshold>=0 and treshold<=1, "treshold should be in [0, 1]"
        
#         self.clear()
        # filter the edges (for efficiency)
        self.initialize_edges(time) # it takes ~40% of the original time with the filtering
        
#         print(self.edges.shape)
        
        # the nodes from which to start and end the algorithm depend on whether we have to perform the direct or reverse search
        if self.reverse:
            start_node = arrival_node 
            target_node = departure_node
        else:
            start_node = departure_node 
            target_node = arrival_node
            
        # list of Visited edges
        visited_ids = []

        # ------------ Initialization ------------ 
        # visit the edges from the starting node
        visited_ids = self.initialize_from(start_node, time)
        # ------------ Main iteration -------------
        while True:
            print(len(visited_ids))
#             print(self.edges.shape)
#             print(self.walking_edges.shape)
            if visited_ids.shape[0] == 0:
                print("There is no path with a probability higher than the given treshold.")
                return []
            
            # get node to be expanded
            edge_id = self.best_edge(visited_ids)
        
            # check termination: we reached the target node and with a edge which is the best one
            if self.done(edge_id, target_node):
                break

            # expand it and update the list of viewed edges
            visited_ids = visited_ids[visited_ids != edge_id] # remove the id we are going to expand
            newly_visited = self.expand_edge(edge_id, treshold=treshold) # expand with the given treshold
            visited_ids = np.append(visited_ids, newly_visited) # update the list of visited nodes
        
        last_edge_id = edge_id # identifies the very last mean we took
        
        # ------------ Collect the edge ids of the found path and return them -------------
        node_col = "arr_node" if self.reverse else "dep_node"
        edge = self.edges.loc[last_edge_id]

        path = [last_edge_id]
        while not pd.isnull(edge.prev_edge):
            edge = self.edges.loc[edge.prev_edge]
            path.append(edge.name)
            
        if not self.reverse:
            path.reverse()
            
        return path
        
    def initialize_from(self, node_name, time):
        """ 
        Initializes the search given a starting node and time.
        Fetches for all edges <node_name> -> A (where A is a reachable node) and return the ones, within the groups of edges with the same trip_id, that starts the first and set them as Visited with probability 1. Repeat for every possible A.
        Note: the meaning of 'starts first' has different meaning depending on the value of self.reverse: 
            - self.reverse = False: search for the first departing edges after the 'time'
            - self.reverse = True : search for the first arriving edges before the 'time'
        return the ids of all the visited edges.
        """
        # here we are applying the aforementioned hypothesis i)
        
        # extract the time
        day, time = time.days, time - pd.Timedelta(str(time.days) + " days")
        
        # to make it more efficient (but could lose some paths) you can remove "trip_id" from the groupby
        
        # select the ids of the edges
        if self.reverse:
            time_col = "arr_time"
            name_col = "arr_node"
            
            ids = (self.edges[(self.edges[name_col] == node_name) & (self.edges[time_col] <= time)]
                .groupby(["dep_node", "arr_node", "trip_id"])[time_col]
                .idxmax().values)
        else:
            time_col = "dep_time"
            name_col = "dep_node"
            ids = (self.edges[(self.edges[name_col] == node_name) & (self.edges[time_col] >= time)]
                .groupby(["dep_node", "arr_node", "trip_id"])[time_col]
                .idxmin().values)
                    
        # visit the selected edges with probability 1
        self.edges.loc[ids, ["prob", "label"]] = [1, Status.Visited]
        
#         self.prune_edges(node_name)
        return ids
        
    def expand_edge(self, edge_id, treshold):
        """ 
        Given a edge id, expand all the edges starting from the reached node whose probability is higher than the given treshold. 
        To prune we consider only the edges from B to A in ascending order of time and stop when there is at least a edge whose probability of taking it is >= Planner.prob_one.
        Note: the meaning of 'reach' has different meaning depending on the value of self.reverse: 
            - self.reverse = False: scan the edges departing after the time and from the station specified by the edge_id 
            - self.reverse = True : scan the edges arriving before the time and to the station specified by the edge_id 
        """
        
        # set the edge as expanded 
        self.edges.loc[edge_id, "label"] = Status.Expanded
        curr_edge = self.edges.loc[edge_id]
        
        # ------------- first visit the walking edges ---------
        ids_walk = self.visit_walking_edges(curr_edge)

        # ------------- then the other edges --------------
        ids = [] # keep a list of visited nodes
        # for each next possible node scan all the possible edges until the probability of 
        # taking the edge is above Planner.prob_one
        next_nodes = self.gen_cadidate_nodes(curr_edge) # generate a DataFrame of edges for each possible next node            
        for next_node in next_nodes:
            for idx, next_edge in next_node.iterrows():                    
                # else compute probability and prune
                prob_edge = self.prob_connection(curr_edge, next_edge)  # prob to take that edge
                prob_path = curr_edge.prob*prob_edge # prob to take the whole path AND that edge 
                if prob_path > treshold:
                    # if above the treshold then I could take (visit) this edge but:
                    # if it is the first time you visit the edge (i.e. label==Unvisited) then just visit this edge
                    # otherwise we also check if we can improve the probability (if so then we re-take this edge)
                    if next_edge.label == Status.Unvisited or (next_edge.label != Status.Unvisited and prob_path > next_edge.prob):
                        ids.append(idx)
                        self.edges.loc[idx, "label"] = Status.Visited
                        self.edges.loc[idx, "prob"] = prob_path

                if prob_edge > Planner.prob_one:
                    # we found at least one edge to the node whose probability is high enough
                    break

#         # finally prune the edges discarding all the ones the would let you come back to the current node
#         if self.reverse:
#             curr_node = curr_edge.dep_node
#         else:
#             curr_node = curr_edge.arr_node
#         self.prune_edges(curr_node)
        
        self.edges.loc[ids, "prev_edge"] = curr_edge.name # set the expanded edge as previous to the newly visited ones
        return ids + ids_walk
    
    def best_edge(self, ids):
        """ Given a list of ids corresponding to all the Visited edges (for efficiency) returns the id of the best edge (the one we have to expand)."""        
        
        candidates = self.all_edges.loc[ids]
        # the best node (the one to be expanded) is:
        if self.reverse:
            # the one with the highest departure time
            day = candidates.dep_day.min()
            return int(candidates.loc[candidates.dep_day == day, "dep_time"].idxmax())
        else:
            # the one with the lowest arrival time (relative to the lowest day)
            day = candidates.arr_day.min()
            return int(candidates.loc[candidates.arr_day == day, "arr_time"].idxmin())
    
    def done(self, best_edge_id, target_node):
        """ Given the id of the best edge and the name of the target node check if we reached it. """
        if self.reverse:
            return self.all_edges.loc[best_edge_id, "dep_node"] == target_node
        else:
            return self.all_edges.loc[best_edge_id, "arr_node"] == target_node
        
    def visit_walking_edges(self, curr_node):
        """ Given the current node search for close nodes reachable by foot, i.e. for walking edges. """
        if self.reverse:
            from_node_col = "arr_node"  
            to_node_col = "dep_node"
            
            to_time_col = "dep_time"
            to_day_col = "dep_time"
        else:
            from_node_col = "dep_node"  
            to_node_col = "arr_node"
            
            to_time_col = "arr_time"
            to_day_col = "arr_day"
        
        valid_walking_edges = self.walking_edges[
            (self.walking_edges.label == Status.Unvisited) &
            (self.walking_edges[from_node_col] == curr_node[to_node_col])
        ]

        for idx, edge in valid_walking_edges.iterrows():
            self.walking_edges.loc[idx, "label"] = Status.Visited
            self.walking_edges.loc[idx, "prob"] = curr_node.prob
            self.walking_edges.loc[idx, to_time_col] += curr_node[to_time_col]
            self.walking_edges.loc[idx, to_day_col] = curr_node[to_day_col]
        
#         # drop all the edges that arrive (or depart if self.reverse == True) to the current node
#         # so to avoid to come back to the node in which I am now
#         self.walking_edges = self.walking_edges[~(
#             (self.walking_edges.label == Status.Unvisited) & 
#             (self.walking_edges[to_node_col] == curr_node[to_node_col]))
#         ]
        
        return valid_walking_edges.index.tolist()

    def prob_connection(self, curr_edge, next_edge):
        """ Computes the probability of taking a connection given the current and next edges. Note that "current" and "next" do refer only to the fact the the algorithm is expanding curr_edge to next_edge, not to the fact that in the path you will take first the curr_edge and then the next_edge as it depends on self.reverse. """
        
        if curr_edge.trip_id == next_edge.trip_id:
            return 1
        
        if self.reverse:
            curr_edge, next_edge = next_edge, curr_edge
        # now curr_edge and next_edge happens in this order on the path!
        
        max_delay = next_edge.dep_time - curr_edge.arr_time
        
        assert max_delay >= pd.Timedelta("00:00:00"), "Error: negative max delay found in prob_connection"
        return curr_edge.distribution.cdf(max_delay.seconds) 
    
    def gen_cadidate_nodes(self, curr_edge):
        """ Returns a generator of DataFrames. For each possible next node return a DataFrame containing all the possible edges that bring there and also sort them from the best one to the worst one (in time sense). """
        if self.reverse:
            # define direction of the edges
            from_time_col = "arr_time"
            to_time_col = "dep_time"
            
            from_node_col = "arr_node"
            to_node_col = "dep_node"
        else:
            # define direction of the edges
            from_time_col = "dep_time"
            to_time_col = "arr_time"
            
            from_node_col = "dep_node"
            to_node_col = "arr_node"
            
            # list of the ids of the possible next edges
            time_constraint = self.edges.dep_time > curr_edge.arr_time        
        
        candidates = self.edges[
            (self.edges[from_node_col] == curr_edge[to_node_col]) &
            self.time_constraint(curr_edge)
        ]
        
        next_nodes = candidates[to_node_col].unique() # get the set of possible next nodes
#         print("-Curr node:", curr_edge[to_node_col], "Next nodes:", next_nodes)
        for next_node in next_nodes:
#             print("\tNext:",next_node)
            yield candidates[candidates[to_node_col] == next_node].sort_values(from_time_col, ascending=(not self.reverse))
    
    def time_constraint(self, curr_edge):
        """ Given the current edge return a list of edge ids of possible candidates such that the connection time is not longer than 24h. """
        if self.reverse:
            curr_edge, next_edge = self.edges, curr_edge
        else:
            curr_edge, next_edge = curr_edge, self.edges
        
        same_day = ((curr_edge.arr_day == next_edge.dep_day) & (curr_edge.arr_time < next_edge.dep_time))
        diff_day = ((curr_edge.arr_day == next_edge.dep_day-1) & (curr_edge.arr_time > next_edge.dep_time))
        return same_day | diff_day
    
    def initialize_edges(self, time):
        """ This is called at the very start of the algrithm. Given the time (which may refer to teh departure or arrival time depending on self.reverse), filters the edges to keep only the ones within 24h. """
        # TODO tread sundays - mondays
        
        # ----- initialize the walking edges just by copying them ------
        self.walking_edges = self.walking_edges_backup
        self.walking_edges.dep_time = -self.walking_edges.arr_time
        
        # ----- initialize the other edges by filtering them with the time ------
        # extract the day and time
        day, time = time.days, time - pd.Timedelta(str(time.days) + " days")
        
        e = self.nowalking_edges_backup
        if self.reverse:
            # same day
            idx = ((e.dep_day == day) & (e.dep_time < time))            
            # and previous day 
            idx = idx | ((e.dep_day == day-1) & (e.dep_time > time))
        else:         
            # same day
            idx = ((e.arr_day == day) & (time < e.arr_time))
            # and next day
            idx = idx | ((e.arr_day == day+1) & (time > e.arr_time))
            
        self.edges = self.nowalking_edges_backup[idx]
        
#   prune  def prune_edges(self, curr_node):
#         """ Given the current edges discard from self.edges all the unseen ones that depart or arrive to the current node. """
#         self.edges = self.edges[~(
#             (self.edges.label == Status.Unvisited) &
#             ((self.edges.dep_node == curr_node) |
#              (self.edges.arr_node == curr_node)
#             )
#         )
#         ]