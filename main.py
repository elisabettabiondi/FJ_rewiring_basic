import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import math
import csv



def calculate_w():
    global G
    global W
    W = nx.to_numpy_array(G) # get the adjacency matrix
    row_sums = W.sum(axis=1)
    for r in range(0,len(row_sums)-1):
        if row_sums[r]!=0:
            W[r] = W[r]/row_sums[r] # generate the normalized adjacency matrix, i.e. the weighted adjacency matrix

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

    count=0


    if initialSetting == "random":
        prefix ="initRand"
        iniop = np.loadtxt('initial_opinion_random.txt')
    else:
        prefix="initRandBlock"
        iniop = np.loadtxt('initial_opinion_randomBlocks.txt')
    for i in G.nodes:

        G.nodes[i]['opinion'] = iniop[count]# Opinion of an agent is a uniform random number between -1 to 1

        G.nodes[i]['initial_opinion'] = G.nodes[i]['opinion']  # Initial opinion
        G.nodes[i]['susceptibility'] = susc_val
        count = count+1
    assign_weights()



rewire0 = -1

def rewire(threshold, model,t):
    global G
    global W
    global rewire0

    number_edges = G.number_of_edges()

    if model == "Asynch":
        # generates the list of disagreement edges
        disagreeList = [(i, j) for (i, j) in G.edges() if abs(G.nodes[i]['opinion'] - G.nodes[j]['opinion']) > threshold]

        if disagreeList!=[]:
            while disagreeList!=[]:
                (i,j)= random.choice(disagreeList) #chooses a disagreement edge
                #generates the list of agreement edges
                agreeList = [k for k in G.nodes() if k!= i and (i,k) not in G.edges and abs(G.nodes[k]['opinion']-G.nodes[i]['opinion']) <= threshold]

                if agreeList !=[]: #if there is an agreement edge then replace the edge
                    G.remove_edge(i, j)
                    #if G.number_of_edges() != number_edges-1:
                    #    print("some problems here")
                    j = random.choice(agreeList)
                    G.add_edge(i, j)
                    W = calculate_w()
                    break
                else:
                    disagreeList.remove((i,j))

        else: #no disagreement in the graph
            if rewire0 == -1:
                print("Time " + str(t) + " WARNING: agreement, not more rewiring")
                rewire0 = 1
            update_opinions_asynch()
    else: #in the synchronous case, the rewiring is done one node at a time
        for i in G.nodes:
            disagreeList = [j for j in G.neighbors(i) if abs(G.nodes[i]['opinion'] - G.nodes[j]['opinion']) > threshold]
            if disagreeList!=[]:
                while disagreeList!=[]:
                    j= random.choice(disagreeList)

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
            neigh_average_i += G.nodes[j]['opinion'] / G.degree[i]

        # Now using our full update equation, we will calculate the new opinion of node i
        susc_i = G.nodes[i]['susceptibility']
        initial_opi_i = G.nodes[i]['initial_opinion']
        G1.nodes[i]['opinion'] = (1 - susc_i) * initial_opi_i + susc_i * neigh_average_i  # neigh_average was computed just above
    G = G1






def update_opinions_asynch():
    global G
    global W

    neigh = []
    nodes = list(G.nodes)
    while neigh == [] and nodes != []:
        i = random.choice(nodes)
        neigh = list(G.neighbors(i))
        nodes.remove(i)

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




def run_simulation(total_timestep,p_rew,threshold, model, susc_val):

    n = G.number_of_nodes()
    iniop = [G.nodes[i]['initial_opinion'] for i in G.nodes()]
    average_opinion = [sum(iniop) / n]
    sum_op_av = [iniop]
    sum_op = iniop
    time = [0]
    t = 1
    # two actions are possible: rewire the edge ("rew") or update the opinion based in the FJ model ("upd")
    actions = ["rew", "upd"]

    #the following code opens the output file
    dist_actions = [p_rew, 1-p_rew]
    polarization_output = open('initRand_polarizationRes_model' + model + '_thr' + str(threshold) + '_probRew' +  str(p_rew) + '_susc' + str(susc_val) + '_2.csv', "w", encoding="utf8")
    pol_writer = csv.writer(polarization_output)
    header = ['bimodality_coefficient','homogeneity','nai','bimodality_coefficient_clean']
    pol_writer.writerow(header)

    while t < total_timestep:


        act = random.choices(actions, dist_actions) #this chooses which action is selected
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


def remove_outliers(vec): #this function removes the outliers of the opinions

    q75, q25 = np.percentile(vec, [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)


    vec = [x for x in vec if x>=min and x<=max]
    return(vec)



def bimodality_coefficient(opinionVector): #this function calculates the bimodality coefficient of the opinions
    global G

    #opinionVector= np.zeros(G.number_of_nodes())
    #for i in G.nodes:
    #    opinionVector[i] = G.nodes[i]['opinion']
    m3 = skew(opinionVector, axis=0, bias=False)
    m4 = kurtosis(opinionVector, axis=0, bias=False)
    n = len(opinionVector)
    b = (m3 ** 2 + 1) / (m4 + 3 * ((n - 1) ** 2 / ((n - 2) * (n - 3))))
    return b

def homogeneity(t,opinionVector): #this function calculates the hommogeneity coefficient of the opinions
    global G
    global W

    A = np.where(W>0, 1, 0)
    n=G.number_of_nodes()

    mean = sum(opinionVector)/n

    #hom0 is the homogeneity coefficient of a random graph
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
    return ((h-hom0)/(1-hom0)) #the final hom coeff is computed normalizing by the homogeneity coeff obtained with a random graph


def nai(t,opinionVector): #this function calculates the network agreement index  of the opinions
    global G
    global W


    n=G.number_of_nodes()

    mean = sum(opinionVector)/n

    #nai0 is the homogeneity coefficient of a random graph
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
    return ((h-nai0)/(1-nai0)) #the final nai coeff is computed normalizing by the nai coeff obtained with a random graph


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



    # We import the initial graph: the file contains the adjacency matrix of the graph.
    # Two types of graphs with 20 nodes are considered: an Erdosh-Renhi random graph with p=0.3 and a stochastic block model with two blocks (A,B) and probabilities p_A=0.4, q_B=0.6 and p_AB= 0.06

    graph = "nonBlock" #this is a label to identify if the network considered is SBM or not

    if graph == "Block":
        A = np.loadtxt('random_block_20_0.4_0.6_0.06.txt')
    else:
        A = np.loadtxt('random_graph_20_0.3.txt')

    G = nx.from_numpy_matrix(A) # create the graph from the adjacency matrix
    W = calculate_w() # generate the weighted adjacency matrix

    # Two initial settings are considered: in "random" opinions are uniformly randomly selected in [-1,1]; in "randomBlock" they are randomly selected in [-0.5,-0.1] if nodes are in block A and in [0.1,0.5] if nodes are in block B
    initialSetting = "random" #"random" or "randomBlock"

    susc_val = 0.9 #susceptibility value for all nodes

    threshold =  0 # this value establishes if nodes are in disagreement or not
    p_rew = 0 #probability of rewiring

    #Two possible rewiring process can be used: Synchronous ("Synch") or Asynchronous ("Asynch")
    model = "Asynch" # "synch" or "asynch" model

    time_steps = 10000 #set 10000 for Asy and 100 for Syn
    initialize(susc_val)



    lab = {}
    for i in G.nodes():
        lab[i] = i

    #run simulation
    average_opinion, time, sum_op_av = run_simulation(time_steps,p_rew,threshold,model, susc_val)#(total_timestep,p_rew,threshold, model)


    plot_avgOp_vs_time(time, sum_op_av,time_steps,model,threshold,p_rew,susc_val)  # This will output a plot showing how the average opinion of team members is changing over time.

    # Teh following generates the graph plot with nodes colored based on the opinions
    f = plt.figure()
    assign_weights()
    pos = nx.spring_layout(G)
    colors = [G.nodes[n]['opinion'] for n in G.nodes()]
    nc = nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=colors,
                                label=G.nodes(), node_size=100, cmap=plt.cm.jet,vmin=-1,vmax=1)
    cbar = plt.colorbar(nc)
    cbar.set_label('Opinions')

    nx.draw_networkx_labels(G, pos, lab, font_color='w', font_size=8, font_family='Verdana')

    # In the following the outputs are saved into txt files
    if graph == "Block":
        f.savefig(
            './results block model/'+prefix+'_finalGraph_model' + model + '_thr' + str(threshold) + '_probRew' +  str(p_rew) + '_susc' + str(susc_val) + '_2.png', bbox_inches='tight')
    else:
        f.savefig('initRand_finalGraph_model' + model + '_thr' + str(threshold) + '_probRew' +  str(p_rew) + '_susc' + str(susc_val) + '_2.png', bbox_inches='tight')
    plt.show()


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