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
        self.all_edges = edges # they will be filtered every time the algorithm starts
        
#         self.nowalking_edges_backup = edges[edges.trip_id != "0000"] # they will be filtered every time the algorithm starts
#         self.edges = self.nowalking_edges_backup
#         self.walking_edges_backup = edges[edges.trip_id == "0000"] # keep separately the walking edges
        
        self.reverse = reverse
        
        self.init_edge_direction()
#         self.sort_edges()
    
#     def clear(self):
#         """ Clear the structure so to restart the computation of the best path. """
#         self.nowalking_edges_backup.prob = np.NaN
#         self.nowalking_edges_backup.label = Status.Unvisited
#         self.nowalking_edges_backup.prev_edge = np.NaN
    
    def compute_plan(self, departure_node, arrival_node, time, treshold):
        """ Given the names of the departure and arrival nodes (e.g. I want to go from 'Zürich HB' to 'Stettbach'), a treshold (in [0, 1]) and the time (which may refer either to the departure or arrival time depending on self.reverse), the bets path (which take the lowest time) whose probability is above the give treshold. """
        
        # TODO could "arrival_node" be a time? so to indicate WHEN to stop instead of WHERE
        
        assert treshold>=0 and treshold<=1, "treshold should be in [0, 1]"
        
#         self.clear()
                
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
#         visited_ids = self.init_edges(start_node, time)
        transp_ids, walk_ids = self.init_edges(start_node, time)

        # ------------ Main iteration -------------
        while True:
#             print(len(visited_ids))
#             print(self.edges.shape)
#             print(self.walking_edges.shape)
            if len(transp_ids) == 0 and len(walk_ids) == 0:
                print("There is no path with a probability higher than the given treshold.")
                return []
            
            # pop node to be expanded
            edge_id = self.pop_best_edge(transp_ids, walk_ids)
        
            # check termination: we reached the target node and with a edge which is the best one
            if self.done(edge_id, target_node):
                break

            # expand the best edge and update the list of viewed edges
            transp_ids_new, walk_ids_new = self.expand_edge(edge_id, treshold=treshold) # expand with the given treshold
            # update the lists of visited edges
            transp_ids += transp_ids_new
            walk_ids += walk_ids_new
        
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

    def init_edge_direction(self):
        """ Initialize the direction of the edge. """
        
        # set the direction of the edges depending on self.reverse (indicate from which solumn to which column they point)
        if self.reverse:
            self.from_node = "arr_node"  
            self.to_node = "dep_node"
            
            self.from_time = "arr_time"
            self.to_time = "dep_time"
            
            self.from_day = "arr_day"
            self.to_day = "dep_day"
        else:
            self.from_node = "dep_node"  
            self.to_node = "arr_node"
            
            self.from_time = "dep_time"
            self.to_time = "arr_time"
            
            self.from_day = "dep_day"
            self.to_day = "arr_day"
        
    def init_edges(self, node_name, time):
        """ 
        Initializes the search given a starting node and time, i.e. initializes the first edges to visit.
        Fetches for all edges <node_name> -> A (where A is a reachable node) and return the ones, within the groups of edges with the same trip_id, that starts the first and set them as Visited with probability 1. Repeat for every possible A.
        Note: the meaning of 'starts first' has different meaning depending on the value of self.reverse: 
            - self.reverse = False: search for the first departing edges after the 'time'
            - self.reverse = True : search for the first arriving edges before the 'time'
        return two list of visited edges ids, the first one regardign transport edges and the second one regarding walk edges.
        """            
        
        # set the direction of the edges depending on self.reverse (indicate from which column to which column they point)
        self.init_edge_direction()
            
        # for efficiency we first select only a substet of edges and then sort them (first teh "best" ones)
        self.select_subset_edges(time) # select only a valid subset of edges
        
        # select only the edges which start from our node
        valid_edges = pd.Index(np.where(self.edges[self.from_node] == node_name)[0])
        
        # ------------------- Initialize transport edges --------------------
        # here we are applying the aforementioned hypothesis i)
        transp_edges = self.edges.loc[valid_edges.intersection(self.transport_ids)]

        # TODO: to make it more efficient (but could lose some paths) you can remove "path_id" from the groupby
        # select the ids of the edges (note: the edges are sorted!) for EACH different path
        transp_ids = (
            transp_edges
               .drop_duplicates(["dep_node", "arr_node", "path_id"], keep='first') # keep first => keep best
               .index)
        
        # visit the selected edges with probability 1
        self.edges.loc[transp_ids, ["prob", "label"]] = [1, Status.Visited]
        
        # -------------------Initialize walk edges --------------------------
        # extract the time
        day, time = time.days, time - pd.Timedelta(days=time.days)
        
        # when summing the time also check to stay within a day
        def sum_time(x):
            new_time = (time - x.walking_time) if self.reverse else (time + x.walking_time)
            
            new_day = day + new_time.days
            
            new_time = new_time - pd.Timedelta(days=new_time.days)
            return [new_day, new_time, x.walking_time]
        
        walk_edges_ids = valid_edges.intersection(self.walk_ids)
        walk_edges = self.edges.loc[walk_edges_ids]
        # visit the selected edges with probability 1 and also update the arrival time
        self.edges.loc[walk_edges_ids, ["prob", "label", self.from_day, self.from_time]] = [1, Status.Visited, day, time]
        self.edges.loc[walk_edges_ids, [self.to_day, self.to_time]] = walk_edges[[self.to_day, self.to_time, "walking_time"]].apply(sum_time, axis=1)[[self.to_day, self.to_time]]
        
        # prune: remove all the other edges with the same path_id that pass by this starting node
        path_to_drop = self.edges.loc[transp_ids].path_id.unique()
        self.edges = self.edges[~(
            self.edges.path_id.isin(path_to_drop) &
            ((self.edges.dep_node == node_name) | (self.edges.arr_node == node_name)) & 
            (self.edges.label == Status.Unvisited))
        ]
        
        return transp_ids.tolist(), walk_edges_ids.tolist()
#         return np.append(transp_ids.values, walk_edges.index.values) # TODO: keep separated (for efficiency)
        
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
        walk_edges = self.edges.loc[self.walk_ids]
        walk_edges = walk_edges[
            (walk_edges[self.from_node] == curr_edge[self.to_node]) & 
            (walk_edges[self.to_node] != curr_edge[self.from_node])
        ]

        ids_walk = []
        for idx, walk_edge in walk_edges.iterrows():
            # for each walking edge you can take, take it only if you never took it or 
            # if you have now a higher probability
            if pd.isnull(walk_edge.prob) or walk_edge.prob < curr_edge.prob:
                ids_walk.append(idx)
        self.visit_walk_edges(curr_edge, ids_walk)

        # ------------- then the other edges --------------
        ids_transp = [] # keep a list of visited nodes
        # for each next possible node scan all the possible edges until the probability of 
        # taking the edge is above Planner.prob_one
        next_nodes = self.gen_cadidate_transports(curr_edge) # generate a DataFrame of edges for each possible next node
        for next_node in next_nodes: 
            # for each edge that bring me to the same next node
            for path_id, next_path in next_node.groupby("path_id"):
                # for each edge that shares the same path
                for idx, next_edge in next_path.iterrows():         
                    # take the edges on this path until the probability gets high enough (usually just 1 or 2)
                    prob_edge = self.prob_connection(curr_edge, next_edge)  # prob to take that edge
                    prob_path = curr_edge.prob*prob_edge # prob to take the whole path AND that edge 
                    if prob_path > treshold:
                        # if above the treshold then I could take (visit) this edge but:
                        # if it is the first time you visit the edge (i.e. label==Unvisited) then just visit this edge
                        # otherwise we also check if we can improve the probability (if so then we re-take this edge)
                        if next_edge.label == Status.Unvisited or (next_edge.label != Status.Unvisited and prob_path > next_edge.prob):
                            ids_transp.append(idx)
                            self.edges.loc[idx, ["label", "prob", "prev_edge"]] = [Status.Visited, prob_path, curr_edge.name]

                    if prob_edge > Planner.prob_one:
                        # we found at least one edge to the node whose probability is high enough
                        break

#         print(self.edges.shape)
        # finally prune the transport edges if the probability of the current edge was ~1
        if curr_edge.prob >= Planner.prob_one:
            self.prune_edges(curr_edge, ids_transp)
        return ids_transp, ids_walk
    
    def pop_best_edge(self, transp_ids, walk_ids):
        """ Given a list of ids corresponding to all the Visited edges (for efficiency) returns the id of the best edge (the one we have to expand)."""        
        
        best_transp_id = min(transp_ids) # we sorted them, the lower the better
        
        # take walk edges and the best transport edge
        candidates = self.edges.loc[np.append(walk_ids, best_transp_id)] 
        if self.reverse:
            # the one with the highest departure time
            day = candidates.dep_day.max()
            best_id = int(candidates.loc[candidates.dep_day == day, "dep_time"].idxmax())
        else:
            # the one with the lowest arrival time
            day = candidates.arr_day.min()
            best_id = int(candidates.loc[candidates.arr_day == day, "arr_time"].idxmin())
            
        if self.edges.loc[best_id].trip_id == "0000":
            walk_ids.remove(best_id)
        else:
            transp_ids.remove(best_id)
        return best_id
    
    def done(self, best_edge_id, target_node):
        """ Given the id of the best edge and the name of the target node check if we reached it. """
        if self.reverse:
            return self.edges.loc[best_edge_id, "dep_node"] == target_node
        else:
            return self.edges.loc[best_edge_id, "arr_node"] == target_node
        
    def visit_walk_edges(self, curr_edge, walk_edge_ids):
        """ Initializes the walk edges correponding to the given indices as visited. """
        
        day, time = curr_edge[self.to_day], curr_edge[self.to_time]
        
        def sum_time(x):
            new_time = (time - x.walking_time) if self.reverse else (time + x.walking_time)
            
            new_day = day + new_time.days
            
            new_time = new_time - pd.Timedelta(days=new_time.days)
            return [new_day, new_time, x.walking_time]
        
        # set the common parameters
        self.edges.loc[walk_edge_ids, [self.from_day, self.from_time, "label", "prob", "prev_edge", "distribution"]] = [
            curr_edge[self.to_day],
            curr_edge[self.to_time],
            Status.Visited, 
            curr_edge.prob,
            curr_edge.name,
            curr_edge.distribution
        ]        
        
        # compute the times
        self.edges.loc[walk_edge_ids, [self.to_day, self.to_time]] = self.edges.loc[walk_edge_ids, [self.to_day, self.to_time, "walking_time"]].apply(sum_time, axis=1)[[self.to_day, self.to_time]]
        
    def prob_connection(self, curr_edge, next_edge):
        """ Computes the probability of taking a connection given the current and next edges. Note that "current" and "next" do refer only to the fact the the algorithm is expanding curr_edge to next_edge, not to the fact that in the path you will take first the curr_edge and then the next_edge as it depends on self.reverse. """
        
        if self.reverse:
            curr_edge, next_edge = next_edge, curr_edge
        # now curr_edge and next_edge happens in this order on the path!
        if curr_edge.trip_id == next_edge.trip_id or pd.isnull(curr_edge.distribution):
            return 1
        
        dep_time = next_edge.dep_time + pd.Timedelta(days=next_edge.dep_day)
        arr_time = curr_edge.arr_time + pd.Timedelta(days=curr_edge.arr_day)
        max_delay = dep_time - arr_time
        
        assert max_delay >= pd.Timedelta("00:00:00"), "Error: negative max delay found in prob_connection"
        return curr_edge.distribution.cdf(max_delay.seconds) 
    
    def gen_cadidate_transports(self, curr_edge):
        """ Returns a generator of DataFrames. For each possible next node return a DataFrame containing all the possible transport edges that bring there sorted from the best one to the worst one (in time sense). """
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
        
        # pick candidate edges that:
        # - are transports
        # - start from where I just arrived
        # - don't bring be back 
        # - and whose time is valid
        candidates = self.edges.loc[self.transport_ids]
        candidates = candidates[
            (candidates[from_node_col] == curr_edge[to_node_col]) &
            (candidates[to_node_col] != curr_edge[from_node_col]) & 
            self.time_constraint(curr_edge, candidates)
        ]
        
        # for each next node yied the dataframe of edges that bring there
        for next_node, next_node_edges in candidates.groupby(to_node_col):
            yield next_node_edges
    
    def time_constraint(self, curr_edge, next_edges):
        """ Given the current edge return a list of edge ids of possible candidates such that the connection time is not longer than 24h. """
        if self.reverse:
            curr_edge, next_edges = next_edges, curr_edge
        
        same_day = ((curr_edge.arr_day == next_edges.dep_day) & (curr_edge.arr_time <= next_edges.dep_time))
        diff_day = ((curr_edge.arr_day == next_edges.dep_day-1) & (curr_edge.arr_time >= next_edges.dep_time))
        return same_day | diff_day
    
    def sort_edges(self):
        """ Sort the (filtered) edges by day and by time and depending on the value of self.reverse => best edges first. """            
        self.edges.sort_values([self.to_day, self.to_time], inplace=True, ascending=not self.reverse)
        self.edges.reset_index(drop=True, inplace=True)
        
        # extract the indices of the transport and walk edges for faster accessing
        self.transport_ids = self.edges.trip_id != "0000"
        self.walk_ids = np.where(~self.transport_ids)[0]
        self.transport_ids = np.where(self.transport_ids)[0]        
        
    def select_subset_edges(self, time):
        """ This is called at the very start of the algrithm. Given the time (which may refer to the departure or arrival time depending on self.reverse), filters the edges to keep only the ones within 24h. It then sort the edges and reset the index so that the lower the index the best the edge (e.g. the lower the arrival time when self.reverse=False): However, the walking edge are not sorted as they do not have any arrival/departure time."""
        
        # extract the day and time (of the query)
        day, time = time.days, time - pd.Timedelta(days=time.days)
        
        e = self.all_edges
        if self.reverse:
            next_day = day-1
            
            # same day
            idx = ((e[self.from_day] == day) & (e[self.from_time] < time))  # lower bound  
            # and previous day 
            idx = idx | ((e[self.from_day] == np.mod(next_day,7)) & 
                         (e[self.to_day] == np.mod(next_day,7)) &
                         (e[self.to_time] > time)) # upper bound
        else:       
            next_day = day+1
            
            # same day
            idx = ((e[self.from_day] == day) & (time < e[self.from_time])) # lower bound
            # and next day
            idx = idx | ((e[self.from_day] == np.mod(next_day,7)) & 
                         (e[self.to_day] == np.mod(next_day,7)) &
                         (time > e[self.to_time])) # upper bound
                    
        self.edges = self.all_edges.loc[idx | (self.all_edges.trip_id=="0000")].copy() # select all the walk edges too
            
        self.edges.loc[self.edges[self.from_day] == np.mod(next_day,7), self.from_day] = next_day
        self.edges.loc[self.edges[self.to_day]   == np.mod(next_day,7), self.to_day] = next_day
        
        self.sort_edges() # sort the edges every time (if self.reverse got changed => change order of the edges too)
        
    def prune_edges(self, curr_edge, ids_transp):
        """ Drop all the outgoung unvisited edges with the same path id as the passed transport edges: it does not make sense to keep a next means of transport when you already took a previous one with high probability. """
        path_ids_to_drop = self.edges.loc[ids_transp, "path_id"].unique()
        
        # criteria:
        # - unvisited edges
        # - which start from the wher current edge arrived
        # - does not bring you back to the previous node (this because you did not took an edge in that direction => a later path may need to travel that way )
        # - with same path_id as already visited transport edges (?)
        self.edges = self.edges[~(
            (self.edges.label == Status.Unvisited) &
            (self.edges[self.from_node] == curr_edge[self.to_node]) & 
            (self.edges[self.to_node] != curr_edge[self.from_node])
#             self.edges.path_id.isin(path_ids_to_drop)
        )
        ]