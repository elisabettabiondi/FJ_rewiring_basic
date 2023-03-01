This code is a simulator for the opinion dynamics of the Friedkin-Johnsen (FJ) model with rewiring. 

Given a network of 20 nodes, at each iteration, the simulation acts:
- with probability p_rew, a rewiring: if two nodes connected that are in disagreement the edge is cut and replaced with one that connects two agreeing nodes
- with probability 1-p_rew, the FJ model

The agreement/disagreement is controlled by a parameter ("threshold"): i and j are in disagreement if and only if |x_i-x_j| > threshold.

The network is loaded by the adjacency matrix. In this example, there are two networks:
- an Erdosh Renhi graph of 20 nodes with p= 0.3 ("random_graph_20_0.3.txt")
- a stochastic block model of 20 nodes with two blocks (A,B) of 10 nodes each with p_A=0.4, q_B=0.6 and p_AB=0.06 ("random_block_20_0.4_0.6_0.06.txt")

The outputs of the file are:
- finalGraph_model: a plot of the final graph with nodes coloured based on the final opinions
- final_opinion: the final opinions of nodes
- final_graph: the weighted matrix of the final graph
- polarizationRes: a list of the coefficients of the graph 'bimodality_coefficient', 'homogeneity', 'nai', 'bimodality_coefficient_clean'
