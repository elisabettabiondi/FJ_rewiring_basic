import networkx as nx
#import random
import numpy as np
#from scipy.stats import skewnorm
import random
import matplotlib.pyplot as plt
from itertools import count
import pickle
from scipy.stats import skew, kurtosis
import math
import csv




#import PyCX.pycxsimulator
# Make sure networkx, numpy, scipy, and matplotlib is installed. PyCX.pycxsimulator is not working on my laptop, so ignoring it.





#G = nx.fast_gnp_random_graph(n=20, p=0.3)
#nodes = G.nodes()


#W = np.matrix([[.220,.120,.360,.300],[.147,251,.344,.294],[0,0,1,0],[0.09,0.178,0.446,0.286]])
#G = nx.from_numpy_matrix(W)



# This is going to be our network of interactions: A random network with 20 nodes and density = 0.3 (we can change this), meaning that the probability that an edge exists between any given pair of nodes = 30%
# I have not implemented the "emotional closeness" yet.

def calculate_w():
    global G
    global W
    W = nx.to_numpy_array(G)
    row_sums = W.sum(axis=1)
    for r in range(0,len(row_sums)-1):
        if row_sums[r]!=0:
            W[r] = W[r]/row_sums[r]

    return W




def assign_weights():
    global G
    global W

    for i, j, w in G.edges(data=True):
        w['weight'] = W[i, j]


def initialize(susc_val):
    global G
    global initialSetting
    global prefix
    #l=[]
    count=0

    #iniop = np.load('initial_opinions.npy', allow_pickle=True)
    #iniop = np.loadtxt('/Users/elisabetta/Documents/CEU-IIT/work/polarizing_vector_susc' + str(susc_val) +'.txt')
    if initialSetting == "random":
        prefix ="initRand"
        iniop = np.loadtxt('/Users/elisabetta/Documents/CEU-IIT/work/initial_opinion_random.txt')
    else:
        prefix="initRandBlock"
        iniop = np.loadtxt('/Users/elisabetta/Documents/CEU-IIT/work/initial_opinion_randomBlocks.txt')
    for i in G.nodes:

        #val = 2 * random.random()-1
        #l.append(val)
        G.nodes[i]['opinion'] = iniop[count]#val # Opinion of an agent is a uniform random number between -1 to 1

        G.nodes[i]['initial_opinion'] = G.nodes[i]['opinion']  # Initial opinion
        G.nodes[i]['susceptibility'] = 0.9 #random.random()   Susceptibility of an agent is a uniform random number between 0 to 1
        count = count+1
    assign_weights()
    #np.save('initial_opinions.npy', l, allow_pickle=True)

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









def observe():
    plt.cla()  # Clears the current plot
    nx.draw(G, cmap=plt.cm.Spectral, vmin=0, vmax=1,
            node_color=[G.nodes[i]['state'] for i in G.nodes],
            edge_cmap=plt.cm.binary, edge_vmin=-1, edge_vmax=1,
            # edge_color = [G.edges[i, j]['weight'] for i, j in G.edges],
            pos=G.pos)
    plt.show()

rewire0 = -1
def rewire(threshold, model,t):
    global G
    global W
    global rewire0

    # for i in G.nodes:
    #     for j in G.neighbors(i):
    #         if abs(G.nodes[j]['opinion']-G.nodes[j]['opinion']) > threshold:
    #             G.remove_edge(i, j)
    #             #print("removed link (",i,",", j,")")
    #             list = [k for k in G.neighbors(i) if abs(G.nodes[k]['opinion']-G.nodes[i]['opinion']) <= threshold]
    #             j = random.choice(list)
    #             G.add_edge(i, j)
    #             #print("added link (",i,",",i, j,")")

    number_edges = G.number_of_edges()
    if model == "Asynch":
        disagreeList = [(i, j) for (i, j) in G.edges() if abs(G.nodes[i]['opinion'] - G.nodes[j]['opinion']) > threshold]
        if disagreeList!=[]:
            while disagreeList!=[]:
                (i,j)= random.choice(disagreeList)
                #G.remove_edge(i, j)
                agreeList = [k for k in G.nodes() if k!= i and (i,k) not in G.edges and abs(G.nodes[k]['opinion']-G.nodes[i]['opinion']) <= threshold]
                if agreeList !=[]:
                    G.remove_edge(i, j)
                    if G.number_of_edges() != number_edges-1:
                        print("some problems here")
                    j = random.choice(agreeList)
                    if (i,j) in G.edges():
                        print("selected an already existing edge")
                    G.add_edge(i, j)
                    W = calculate_w()
                    break
                else:
                    disagreeList.remove((i,j))
            #G = nx.from_numpy_matrix(W)
            #G.nodes = nodes
        else:
            if rewire0 == -1:
                print("Time " + str(t) + " WARNING: agreement, not more rewiring")
                rewire0 = 1
            update_opinions_asynch()
    else:
        for i in G.nodes:
            disagreeList = [j for j in G.neighbors(i) if abs(G.nodes[i]['opinion'] - G.nodes[j]['opinion']) > threshold]
            if disagreeList!=[]:
                while disagreeList!=[]:
                    j= random.choice(disagreeList)
                    #G.remove_edge(i, j)
                    #print("removed link (",i,",", j,")")
                    agreeList = [k for k in G.nodes() if  k!= i and (i,k) not in G.edges and abs(G.nodes[k]['opinion']-G.nodes[i]['opinion']) <= threshold]
                    if agreeList != []:
                        G.remove_edge(i, j)
                        j = random.choice(agreeList)
                        G.add_edge(i, j)
                        W = calculate_w()
                        break
                    else:
                        disagreeList.remove((i, j))
    if number_edges != G.number_of_edges():
        print("Problem here")



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


def run_sumulation(total_timestep,p_rew,threshold, model, susc_val):
    #opinions = []
    n = G.number_of_nodes()
    iniop = [G.nodes[i]['initial_opinion'] for i in G.nodes()]
    average_opinion = [sum(iniop) / n]
    sum_op_av = [iniop]
    sum_op = iniop
    time = [0]
    t = 1
    actions = ["rew", "upd"]
    #p_rew = 0 #define the rewiring probability
    dist_actions = [p_rew, 1-p_rew]
    polarization_output = open('initRand_polarizationRes_model' + model + '_thr' + str(threshold) + '_probRew' +  str(p_rew) + '_susc' + str(susc_val) + '_2.csv', "w", encoding="utf8")
    pol_writer = csv.writer(polarization_output)
    header = ['bimodality_coefficient','homogeneity','nai','bimodality_coefficient_clean']
    pol_writer.writerow(header)

    while t < total_timestep:


        act = random.choices(actions, dist_actions)
        #print(act)

        if model == "Synch":
            if act == ["upd"]:
                r = update_opinions_synch()
                if r == -1:
                    return -1, -1
            else:
                # print("rewiring")
                rewire(threshold,model,t)

            op = [G.nodes[i]['opinion'] for i in
                      G.nodes()]
            sum_op_av.append(op)
            row = [bimodality_coefficient(op), homogeneity(t,op), nai(t,op),bimodality_coefficient(remove_outliers(op))]
        else:
            if act == ["upd"]:
                r = update_opinions_asynch()
                if r == -1:
                    return -1, -1
            else:
                # print("rewiring")
                rewire(threshold, model,t)


            sum_op = [sum_op[i] + G.nodes[i]['opinion'] for i in
                      G.nodes()]
            # pdf([sum_op[i]/(t+1) for i in G.nodes()])
            op = [sum_op[i] / (t + 1) for i in G.nodes()]
            sum_op_av.append(op)
            row = [bimodality_coefficient(op), homogeneity(t, op), nai(t, op),bimodality_coefficient(remove_outliers(op))]


        pol_writer.writerow(row)

        avg_op = sum([G.nodes[i]['opinion'] for i in
                      G.nodes()]) / n  # Calculating average opinion of all agents after each update
        average_opinion.append(avg_op)
        time.append(t)
        t += 1

    polarization_output.close()
    return average_opinion, time, sum_op_av


def remove_outliers(vec):

    q75, q25 = np.percentile(vec, [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)


    vec = [x for x in vec if x>=min and x<=max]
    return(vec)



def bimodality_coefficient(opinionVector):
    global G

    #opinionVector= np.zeros(G.number_of_nodes())
    #for i in G.nodes:
    #    opinionVector[i] = G.nodes[i]['opinion']
    m3 = skew(opinionVector, axis=0, bias=False)
    m4 = kurtosis(opinionVector, axis=0, bias=False)
    n = len(opinionVector)
    b = (m3 ** 2 + 1) / (m4 + 3 * ((n - 1) ** 2 / ((n - 2) * (n - 3))))
    return b

def homogeneity(t,opinionVector):
    global G
    global W

    A = np.where(W>0, 1, 0)
    n=G.number_of_nodes()
    #opinionVector = np.zeros(n)
    #for i in G.nodes:
    #    opinionVector[i] = G.nodes[i]['opinion']
    mean = sum(opinionVector)/n
    if t == 0:
        hom0 = 2/3
    else:
        if mean <= 1/3:
            slope = 1.5 * mean
            hom0 = 2/15*(5+2*slope** 2)
        else:
            slope =2/9 * (1/(1-mean)**2)
            hom0 = 1-( 2* math.sqrt(2)/ (15* math.sqrt(slope) ))
    h =0
    for i in range(n) :
        Asum = np.array(A).sum(axis=1)
        if Asum[i] != 0:
            h = h+(1-1/Asum[i] * sum(abs(np.array(opinionVector) - np.array([opinionVector[i]]*n))*A[:][i])/2)
        else:
            h=h+1
    h = h/n
    return ((h-hom0)/(1-hom0))


def nai(t,opinionVector):
    global G
    global W


    n=G.number_of_nodes()
    #opinionVector = np.zeros(n)
    #for i in G.nodes:
    #    opinionVector[i] = G.nodes[i]['opinion']
    mean = sum(opinionVector)/n
    if t == 0:
        nai0 = 2/3
    else:
        if mean <= 1/3:
            slope = 1.5 * mean
            nai0 = 5/6+2/9*slope** 2
        else:
            slope =2/9 * (1/(1-mean)**2)
            nai0 = 1-( 1/ 36* slope )
    h =0
    for i in range(n) :
        h = h+ (1 - sum( (np.array(opinionVector) - np.array([opinionVector[i]]*n))**2  * W[:][i])/4)
    h = h/n
    return ((h-nai0)/(1-nai0))


def plot_avgOp_vs_time(time, sum_op_av,time_steps, model,threshold,p_rew,susc_val):
    global graph
    global prefix

    #initialize()
    #average_opinion, time, sum_op_av = run_sumulation(2000) # Get values over 50 timesteps
    plt.figure(figsize=(10, 5))
    #plt.plot(time, average_opinion, "ro-", markersize = 4)
    plt.plot(time, sum_op_av,  markersize=4)
    plt.xlabel("Timestamp", fontsize=18, color = "black")
    if model== "Asynch":
        plt.ylabel(u'z\u0305', fontsize=18, color = "black")
    else:
        plt.ylabel("Opinions",  fontsize=18,color="black")
    if model == "Synch":
        plt.xticks(np.arange(0,time_steps,step=round(time_steps/20)), color = "black")
    else:
        plt.xticks(np.arange(0, time_steps, step=round(time_steps / 10)), color="black")
    plt.yticks(fontsize=18,color = "black")
    plt.ylim([np.min(sum_op_av)-0.04,np.max(sum_op_av)+0.04])
    if graph == "Block":
        plt.savefig(
            './results block model/'+ prefix +'_opinions_model' + model + '_thr' + str(threshold) + '_probRew' + str(p_rew) + '_susc' + str(
                susc_val) + '_2.png')
    else:
        plt.savefig('initRand_opinions_model' + model + '_thr' + str(threshold) + '_probRew' +  str(p_rew) + '_susc' + str(susc_val) + '_2.png')
    #plt.show()




def main():
    global G
    global W
    global nodes
    global graph
    global prefix
    global initialSetting

    graph = "nonBlock"

    # HERE WE HAVE TO IMPORT THE INITIAL GRAPH:

    if graph == "Block":
        A = np.loadtxt('/Users/elisabetta/Documents/CEU-IIT/work/random_block_20_0.4_0.6_0.06.txt')
    else:
        A = np.loadtxt('/Users/elisabetta/Documents/CEU-IIT/work/random_graph_20_0.3.txt')

    G = nx.from_numpy_matrix(A)
    W = calculate_w()

    initialSetting = "random"#"random" or "randomBlock"

    susc_val = 0.5
    susc = susc_val * np.ones(len(G.nodes)) #vector of susceptibility values
    threshold =  0 #admittable disagreement for not rewiring
    p_rew = 0 #probability of rewiring
    model = "Asynch" #synch or asynch model
    time_steps = 10000 #set 10000 for Asy and 100 for Syn
    initialize(susc_val)

    #I  have commented the following lines because the initial graph is the same for every simulation
    # f = plt.figure()
    # # get unique groups
    #
    # #G = nx.from_numpy_matrix(W)
    # #G.nodes() = nodes
    # widths = nx.get_edge_attributes(G, 'weight')
    # # nx.draw(G, ax=f.add_subplot(111), with_labels = True)
    # pos = nx.spring_layout(G)
    # colors = [G.nodes[n]['opinion'] for n in G.nodes()]
    # ec = nx.draw_networkx_edges(G, pos, alpha=0.6, width=list(widths.values()))
    # nc = nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=colors,
    #                             label=G.nodes(), node_size=100, cmap=plt.cm.jet, vmin =-1, vmax =1)

    lab = {}
    for i in G.nodes():
        lab[i] = i

    # nx.draw_networkx_labels(G,pos ,lab, font_color='w', font_size=8, font_family='Verdana')
    # cbar = plt.colorbar(nc)
    # cbar.set_label('Opinions')
    # if graph == "Block":
    #   f.savefig('./results block model/' + prefix +'_initialGraph_model' + model + '_thr' + str(threshold) + 'probRew_' +  str(p_rew) + 'susc' + str(susc_val) + '.png', bbox_inches='tight')
    # else:
    #   f.savefig('initRand_initialGraph_model' + model + '_thr' + str(threshold) + 'probRew_' +  str(p_rew) + 'susc' + str(susc_val) + '.png', bbox_inches='tight')
    # # #plt.show()


    average_opinion, time, sum_op_av = run_sumulation(time_steps,p_rew,threshold,model, susc_val)#(total_timestep,p_rew,threshold, model)
    plot_avgOp_vs_time(time, sum_op_av,time_steps,model,threshold,p_rew,susc_val)  # This will output a plot showing how the average opinion of team members is changing over time.

    f = plt.figure()
    # get unique groups
    assign_weights()
    widths = nx.get_edge_attributes(G, 'weight')
    #nx.draw(G, ax=f.add_subplot(111), with_labels = True)
    pos = nx.spring_layout(G)
    colors = [G.nodes[n]['opinion'] for n in G.nodes()]
    ec = nx.draw_networkx_edges(G, pos, alpha=0.6,width=list(widths.values()))
    nc = nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=colors,
                                label=G.nodes(), node_size=100, cmap=plt.cm.jet,vmin=-1,vmax=1)
    cbar = plt.colorbar(nc)
    cbar.set_label('Opinions')

    nx.draw_networkx_labels(G, pos, lab, font_color='w', font_size=8, font_family='Verdana')
    if graph == "Block":
        f.savefig(
            './results block model/'+prefix+'_finalGraph_model' + model + '_thr' + str(threshold) + '_probRew' +  str(p_rew) + '_susc' + str(susc_val) + '_2.png', bbox_inches='tight')
    else:
        f.savefig('initRand_finalGraph_model' + model + '_thr' + str(threshold) + '_probRew' +  str(p_rew) + '_susc' + str(susc_val) + '_2.png', bbox_inches='tight')
    #plt.show()
    #f1=nx.draw(G)
    #f1.show()
    #plt.savefig('ne.png')
    if graph == "Block":
        np.savetxt(
            './results block model/'+prefix+'_final_opinion_model' + model + '_thr' + str(threshold) + '_probRew' +  str(p_rew) + '_susc' + str(susc_val) + '_2.txt', sum_op_av[-1])
        np.savetxt(
            './results block model/'+prefix+'_weighted_matrix_model' + model + '_thr' + str(threshold) + '_probRew' +  str(p_rew) + '_susc' + str(susc_val) + '_2.txt', W)
    else:
        np.savetxt('initRand_final_opinion_model' + model + '_thr' + str(threshold) + '_probRew' +  str(p_rew) + '_susc' + str(susc_val) + '_2.txt', sum_op_av[-1])
        np.savetxt('initRand_weighted_matrix_model' + model + '_thr' + str(threshold) + '_probRew' +  str(p_rew) + '_susc' + str(susc_val) + '_2.txt', W)

if __name__ == "__main__":
    main()