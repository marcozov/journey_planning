# -*- coding: utf-8 -*-

""" Contains classes and functions used for the robust journey planning. """

from scipy.stats import gamma
import pandas as pd
import numpy as np

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

    
def compute_gamma_prob(mean, std, max_val):
    """ Given mean and std of the distribution, creates a gamma distribution and computes the cdf of the passed max_val """
    
    alpha = (mean/std)**2
    scale = std**2 / mean
    gam = gamma(a=alpha, scale=scale)

    prob = gam.cdf(max_val)
    return prob
    
    
class Planner():
    """ Class used to compute the journey plan. """
    prob_one = 0.99 # approximation of probability 1, used to prune the choices
    
    def __init__(self, edges, reverse=False):
        """ 
        Input:
        - edges: DataFrame indicating all the edges (edges) between nodes.
        - reverse: boolean indicating wether the graph has to be explored in a direct of reversed mode, i.e. if the exploration starts from the departure node (False) or from the arrival node (True).
        """
        
        self.edges = edges
        self.reverse = reverse
    
    def clear(self):
        """ Clear the structure so to restart the computation of the best path. """
        self.edges.prob = np.NaN
        self.edges.label = Status.Unvisited
        self.edges.prev_edge = np.NaN
    
    def compute_plan(self, departure_node, arrival_node, time, treshold):
        """ Given the names of the departure and arrival nodes, a treshold and the time (which may refer either to the departure or arrival time depending on self.reverse), computes the edges above the treshold. """
        assert treshold>=0 and treshold<=1, "treshold should be in [0, 1]"
        
        self.clear() # TODO clearn other fields?
        
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
            if visited_ids.shape[0] == 0:
                print("There is no path with a probability higher than the given treshold.")
                return []
            
            # get node to be expanded
            edge_id = self.best_node(visited_ids)
        
            # check termination: we reached the target node and with a edge which is the best one
            if self.done(edge_id, target_node):
                break

            # expand it
            visited_ids = visited_ids[visited_ids != edge_id] # remove the id we are going to expand
            newly_visited = self.expand_edge(edge_id, treshold=treshold) # expand
            visited_ids = np.append(visited_ids, newly_visited) # update the list of visited nodes
        
        last_edge_id = edge_id
        
        # ------------ Collect the edges id of the found path and return them -------------
        node_col = "arr_name" if self.reverse else "dep_name"
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
        Fetches for all edges <node_name> -> A (where A is a reachable node) the one that starts the first and and set it as Visited with probability 1. 
        Note: the meaning of 'starts first' has different meaning depending on the value of self.reverse: 
            - self.reverse = False: search for the first departing edges after the 'time'
            - self.reverse = True : search for the first arriving edges before the 'time'
        return the ids of the visited edges.
        """
        # here we are applying the aforementioned hypothesis i)
        
        # select the ids of the edges
        if self.reverse:
            time_col = "arr_time"
            name_col = "arr_name"
            
            ids = (self.edges[(self.edges[name_col] == node_name) & (self.edges[time_col] <= time)]
                .groupby(["dep_name", "arr_name"])[time_col]
                .idxmax().values)
        else:
            time_col = "dep_time"
            name_col = "dep_name"
            ids = (self.edges[(self.edges[name_col] == node_name) & (self.edges[time_col] >= time)]
                .groupby(["dep_name", "arr_name"])[time_col]
                .idxmin().values)
                    
        # visit the selected edges with probability 1
        self.edges.loc[ids, ["prob", "label"]] = [1, Status.Visited]
        return ids
        
    def expand_edge(self, edge_id, treshold=0.8):
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
        
        if self.reverse:
            # define direction of the edges
            from_time_col = "arr_time"
            to_time_col = "dep_time"
            
            from_node_col = "arr_name"
            to_node_col = "dep_name"
            
            # define the constraint between the time in the 'from_time_col' column and the current time
            time_constraint = lambda arr_time, curr_time: arr_time < curr_time
            # specify how to order the edges when scanning them to compute the probabilities
            ascending = False
            
            # given a next possible edge compute the probability of taking it            
            def compute_prob(next_edge):
                max_delay = curr_time - next_edge.arr_time
                std, mean = next_edge.std_delay, next_edge.mean_delay

                return compute_gamma_prob(mean, std, max_delay) 
        else:            
            # define direction of the edges
            from_time_col = "dep_time"
            to_time_col = "arr_time"
            
            from_node_col = "dep_name"
            to_node_col = "arr_name"
            
            # define the constraint between the time in the 'from_time_col' column and the current time
            time_constraint = lambda dep_time, curr_time: dep_time > curr_time
            # specify how to order the edges when scanning them to compute the probabilities
            ascending = True
            
            # given a next possible edge compute the probability of taking it
            def compute_prob(next_edge):
                max_delay = next_edge.dep_time - curr_time
                std, mean = curr_edge.std_delay, curr_edge.mean_delay
                
                return compute_gamma_prob(mean, std, max_delay) 
            
        # get current node and time
        curr_node = curr_edge[to_node_col]
        curr_time = curr_edge[to_time_col]
            
        # TODO: check day (select only current and next days)
        # select all the possible candidate edges
        # get also the aready visited edges, we will try to improve their probability
        candidates = self.edges[
            (self.edges[from_node_col] == curr_node) & 
            (time_constraint(self.edges[from_time_col], curr_time))
        ]

        # for each next possible node scan all the possible edges until the probability of taking the edge is above Planner.prob_one
        nodes = candidates[to_node_col].unique() # get the set of possible next nodes
        ids = []
        for next_node in nodes:
            possible_edges = candidates[candidates[to_node_col] == next_node].sort_values(from_time_col, ascending=ascending)
            for idx, edge in possible_edges.iterrows():
                # TODO: check if the column trip_id remains the same (if so just propagate curr_edge.prob)
                prob_edge = compute_prob(edge)  # prob to take that edge
                prob_path = curr_edge.prob*prob_edge # prob to take the whole path AND that edge 
                if prob_path > treshold:
                    # if above the treshold then I could take (visit) this edge
                    
                    # if it is the first time you visit the edge (label==Unvisited) then just visit this edge
                    # otherwise we also check if we can improve the probability (if so then we re-take this edge)
                    if edge.label == Status.Unvisited or (edge.label != Status.Unvisited and prob_path > edge.prob):
                        ids.append(idx)
                        self.edges.loc[idx, "label"] = Status.Visited
                        self.edges.loc[idx, "prob"] = prob_path
                            
                if prob_edge > Planner.prob_one:
                    # we found at least one edge to the node whose probability is high enough
                    break
        
        self.edges.loc[ids, "prev_edge"] = curr_edge.name # set the expanded edge as previous to the newly visited ones
        return ids
    
    def best_node(self, ids):
        """ Given a list of ids corresponding to al the Visited edges (for efficiency) returns the id of the best edge (the one we have to expand)."""        
        # the best node (the one to be expanded) is:
        if self.reverse:
            # the one with the highest departure time
            return int(self.edges.loc[ids, "dep_time"].idxmax())
        else:
            # the one with the lowest arrival time
            return int(self.edges.loc[ids, "arr_time"].idxmin())
    
    def done(self, best_edge_id, target_node):
        """ Given the id of the best edge and the name of the target node check if we reached it. """
        if self.reverse:
            return self.edges.loc[best_edge_id, "dep_name"] == target_node
        else:
#             print(self.edges.loc[best_edge_id, "arr_name"], target_node)
            return self.edges.loc[best_edge_id, "arr_name"] == target_node