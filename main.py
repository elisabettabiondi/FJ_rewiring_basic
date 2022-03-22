import networkx as nx
#import random
import numpy as np
#from scipy.stats import skewnorm
import random
import matplotlib.pyplot as plt

#import PyCX.pycxsimulator
# Make sure networkx, numpy, scipy, and matplotlib is installed. PyCX.pycxsimulator is not working on my laptop, so ignoring it.

G = nx.fast_gnp_random_graph(n=20, p=0.3)

#W = np.matrix([[.220,.120,.360,.300],[.147,251,.344,.294],[0,0,1,0],[0.09,0.178,0.446,0.286]])
#G = nx.from_numpy_matrix(W)



# This is going to be our network of interactions: A random network with 20 nodes and density = 0.3 (we can change this), meaning that the probability that an edge exists between any given pair of nodes = 30%
# I have not implemented the "emotional closeness" yet.
def initialize(susc):
    for i in G.nodes:
        G.nodes[i]['opinion'] = random.random()  # Opinion of an agent is a uniform random number between 0 to 1
        G.nodes[i]['initial_opinion'] = G.nodes[i]['opinion']  # Initial opinion
        G.nodes[i]['susceptibility'] = 0.9 #random.random()   Susceptibility of an agent is a uniform random number between 0 to 1
    # G.nodes[0]['susceptibility'] = 0.78
    # G.nodes[1]['susceptibility'] = 0.785
    # G.nodes[2]['susceptibility'] = 0.
    # G.nodes[3]['susceptibility'] = 0.714
    # G.nodes[0]['opinion'] = 25
    # G.nodes[1]['opinion'] = 25
    # G.nodes[2]['opinion'] = 75
    # G.nodes[3]['opinion'] = 85
    # for i in G.nodes:
    #     #G.nodes[i]['opinion'] = random.random()  # Opinion of an agent is a uniform random number between 0 to 1
    #     G.nodes[i]['initial_opinion'] = G.nodes[i]['opinion']  # Initial opinion
    #     #G.nodes[i]['susceptibility'] = 0.5 #random.random()   Susceptibility of an agent is a uniform random number between 0 to 1


def calculate_w():
    global G

    W = nx.to_numpy_array(G)
    row_sums = W.sum(axis=1)
    W = W / row_sums[:, np.newaxis]

    return W


W = calculate_w()


def observe():
    plt.cla()  # Clears the current plot
    nx.draw(G, cmap=plt.cm.Spectral, vmin=0, vmax=1,
            node_color=[G.nodes[i]['state'] for i in G.nodes],
            edge_cmap=plt.cm.binary, edge_vmin=0, edge_vmax=1,
            # edge_color = [G.edges[i, j]['weight'] for i, j in G.edges],
            pos=G.pos)
    plt.show()


def rewire(threshold):
    global G
    global W

    for i in G.nodes:
        for j in G.neighbors(i):
            if abs(G.nodes[j]['opinion']-G.nodes[j]['opinion']) > threshold:
                G.remove_edge(i, j)
                #print("removed link (",i,",", j,")")
                list = [k for k in G.neighbors(i) if abs(G.nodes[k]['opinion']-G.nodes[i]['opinion']) <= threshold]
                j = random.choice(list)
                G.add_edge(i, j)
                #print("added link (",i,",",i, j,")")
    W = calculate_w()


def update_opinions_synch():
    global G
    G1 = G #use G1 for building the new opinions
    for i in G1.nodes:

        neigh_average_i = 0
        for j in G.neighbors(i):
            neigh_average_i += G.nodes[j]['opinion'] / G.degree[i]  # It is the susceptibility component of the opinion update. In this case we only have a network with a

        # Now using our full update equation, we will calculate the new opinion of node i
        susc_i = G.nodes[i]['susceptibility']
        initial_opi_i = G.nodes[i]['initial_opinion']
        G1.nodes[i]['opinion'] = (1 - susc_i) * initial_opi_i + susc_i * neigh_average_i  # neigh_average was computed just above
    G = G1


        # PyCX.pycxsimulator.GUI().start(func=[initialize, observe, update])


# The above line of code throws error in my laptop, please uncomment it and check if it runs on yours!

def update_opinions_asynch():
    global G
    global W

    neigh = []
    nodes = list(G.nodes)
    while neigh == [] and nodes != []:
        i = random.choice(nodes)
        neigh = list(G.neighbors(i))
        nodes.remove(i)
    #print("selected node ", i)
    if not nodes:
        print("No more edges")
        return(-1)
    deg_i = G.degree(i)
    susc_i = G.nodes[i]['susceptibility']
    if deg_i == 1:
        h_i = 0
    else:
        h_i = (deg_i - (1-susc_i))/deg_i
    j = random.sample(list(G.neighbors(i)), 1)[0]
    if i == j and deg_i != 1:
        gamma_ij = (deg_i*(1-h_i)+h_i-(1-susc_i * W[i,i]))/h_i
    elif i!=j and deg_i != 1:
        gamma_ij = susc_i * W[i,j]/h_i
    elif i == j and deg_i == 1:
        gamma_ij = 1
    else:
        gamma_ij = 0

    initial_opi_i = G.nodes[i]['initial_opinion']
    G.nodes[i]['opinion'] = h_i * ((1 - gamma_ij) * G.nodes[i]['opinion'] + gamma_ij * G.nodes[j]['opinion']) + (1-h_i) * initial_opi_i  # calculated as the gossip-based algo
    return 0


        # PyCX.pycxsimulator.GUI().start(func=[initialize, observe, update])


# The above line of code throws error in my laptop, please uncomment it and check if it runs on yours!

def pdf(op):
    hist, bins = np.histogram(op, bins=200, normed=True)
    bin_centers = (bins[1:] + bins[:-1]) * 0.5
    plt.plot(bin_centers, hist)


def run_sumulation(total_timestep,p_rew,threshold, model):
    #opinions = []
    average_opinion = []
    sum_op_av = []
    sum_op = np.zeros(len(G.nodes()))
    time = []
    t = 0
    actions = ["rew", "upd"]
    #p_rew = 0 #define the rewiring probability
    dist_actions = [p_rew, 1-p_rew]
    n = G.number_of_nodes()

    while t < total_timestep:


        act = random.choices(actions, dist_actions)
        #print(act)

        if model == "synch":
            if act == ["upd"]:
                r = update_opinions_synch()
            else:
                # print("rewiring")
                rewire(threshold)
            if r == -1:
                return -1, -1
            sum_op_av.append([G.nodes[i]['opinion'] for i in
                      G.nodes()])
        else:
            if act == ["upd"]:
                r = update_opinions_asynch()
            else:
                # print("rewiring")
                rewire(threshold)
            if r == -1:
                return -1, -1

            sum_op = [sum_op[i] + G.nodes[i]['opinion'] for i in
                      G.nodes()]
            # pdf([sum_op[i]/(t+1) for i in G.nodes()])
            sum_op_av.append([sum_op[i] / (t + 1) for i in G.nodes()])




        avg_op = sum([G.nodes[i]['opinion'] for i in
                      G.nodes()]) / n  # Calculating average opinion of all agents after each update
        average_opinion.append(avg_op)
        time.append(t)
        t += 1

    return average_opinion, time, sum_op_av


def plot_avgOp_vs_time(time, sum_op_av,time_steps, model):
    #initialize()
    #average_opinion, time, sum_op_av = run_sumulation(2000) # Get values over 50 timesteps
    plt.figure(figsize=(100, 5))
    #plt.plot(time, average_opinion, "ro-", markersize = 4)
    plt.plot(time, sum_op_av,  markersize=4)
    plt.xlabel("Timestamp", fontsize=18, color = "black")
    if model== "asynch":
        plt.ylabel(u'z\u0305', fontsize=18, color = "black")
    else:
        plt.ylabel("Opinions", fontsize=18, color="black")
    plt.xticks(np.arange(0,time_steps,step=time_steps/20),fontsize=18, color = "black")
    plt.yticks(fontsize=18, color = "black")
    plt.show()




def main():
    susc = 0.5 * np.ones(len(G.nodes)) #vector of susceptibility values
    threshold =  0.1 #admittable disagreement for not rewiring
    p_rew = 0.5 #probability of rewiring
    model = "synch" #synch or asynch model
    time_steps = 100
    initialize(susc)
    average_opinion, time, sum_op_av = run_sumulation(time_steps,p_rew,threshold,model)#(total_timestep,p_rew,threshold, model)
    plot_avgOp_vs_time(time, sum_op_av,time_steps,model)  # This will output a plot showing how the average opinion of team members is changing over time.

if __name__ == "__main__":
    main()