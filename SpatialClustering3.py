class Cluster:
    o1 = 0.0
    o2 = 0.0
    n = 0
    m = 0
        
    def __init__(self, o1, o2, n, m, k, input_data):
        self.o1 = o1
        self.o2 = o2
        self.n = n
        self.m = m
        self.k = k
        self.input_data = input_data
        self.membership = []
        self.border = [] #1: a border case 1 and 0: not a border case 
        self.nc = [] # nc[k]: number of cases in cluster k
        self.centers = [[]] # centers
        self.total_distance = 0 #distance
        self.mbd = 0 #mean boundary distance
        self.distance = 0 #distance for eliminating cluster solutions
        self.avg_attributes = [] #mean for j = d+1 ... d+m
        self.tp = 0 #total penalty for clustering solution

    def assign_cluster(self):
        for i in range(self.n):
            self.membership.append(random.randint(0,self.k-1))
        return self.membership



    def assign_cluster_construction(self,neighbor_array,neighbor_count):

        #-1 not assigned to a cluster yet.

        
    
        self.membership=numpy.ones(self.n,dtype=int) * -1 

        #print(self.membership)
        random_i = [i for i in range(self.n)]
        shuffle(random_i)
        
        #start with k random cluster points
        for i in range(self.k):
            j=random_i[i]
            self.membership[j]=i

        find_a_point=1
        random_i = [i for i in range(neighbor_count)]
        
        while(find_a_point==1):
            find_a_point=0 #exit from the loop when we cannot make any more assigments
            shuffle(random_i)
            for i in random_i:
                a = neighbor_array[1][i]
                #print(a)
                b = neighbor_array[2][i]
                #print(b)
                #a is assigned but not b or vice versa. 
                if self.membership[a] > -1 and self.membership[b] == -1:
                    self.membership[b]=self.membership[a]
                    find_a_point=1
                    break
                
                if self.membership[b] > -1 and self.membership[a] == -1:
                    self.membership[a]=self.membership[b]
                    find_a_point=1
                    break
                
        return self.membership

    def identify_border(self,neighbor_array,neighbor_count):
        self.border=numpy.zeros(self.n)
        for i in range(neighbor_count):
            a = neighbor_array[1][i]
            b = neighbor_array[2][i]
            if self.membership[a] != self.membership[b]:
                self.border[a]=1;
                self.border[b]=1;
        return 0
		
		

##def calc_avg_attributes(n,d1_dm,m_values):
##    avg_attributes = []
##    for j in range(d1_dm):
##        add = 0
##        for i in range(n):
##            add += float(m_values[i][j])
##        add = add / n
##        avg_attributes.append(add)
##    return avg_attributes

def calc_avg_attributes(n, k, d1_dm, m_values, membership):

    sum_distances = numpy.zeros((k,d1_dm), dtype=float)
    n_membership=numpy.zeros((k,d1_dm),dtype=int)
    avg_attributes=numpy.zeros((k,d1_dm), dtype=float)
    
    for i in range(n):
        km = membership[i]
        for c in range(d1_dm):
            sum_distances[km][c] += float(m_values[i][c])
            n_membership[km][c]+=1

    for c in range(k):
        for i in range(d1_dm):
            if(n_membership[c][i] > 0 ):    
                avg_attributes[c][i]=sum_distances[c][i]/n_membership[c][i]

    return avg_attributes

def calc_sum_attributes(n, k, d1_dm, m_values, membership):

    sum_attributes=numpy.zeros((k,d1_dm), dtype=float)
    
    for i in range(n):
        km = membership[i]
        for c in range(d1_dm):
            sum_attributes[km][c] += float(m_values[i][c])

    #print(sum_attributes) 

    return sum_attributes

def calc_total_penalty(k,pj,a_array,sum_attributes,avg_sum_in):
    tp = 0

    for c in range(k):
        tp += (pj*max(0,(a_array-(min(sum_attributes[c][0], sum_attributes[c][1])/max(sum_attributes[c][0], sum_attributes[c][1])))))
    
    for c in range(k):
        tp += (pj*max(0,((numpy.abs(sum_attributes[c][0] - avg_sum_in)/avg_sum_in)-0.1)))

    return tp

#def calc_total_penalty(k, d1_dm, pj,a_array,sum_attributes):
    #tp = 0
    
    #for c in range(k):
        #for i in range(d1_dm):
            #tp += (pj[i]*max(0,(a_array[i]-sum_attributes[c][i])))
    #return tp


def calc_centers(n, k, membership, d_values):
    # notes
    # i case index
    # j variable index
    # q cluster index
    # c coordinates index

    center_array = numpy.zeros((k, 2))
    nc = numpy.zeros(k)

    sumc = numpy.zeros((k, 2))
        
    # each cluster has zero member
    for q in range(k):
        nc[q] = 0

    for i in range(n):
        #get cluster membership of ith case
        kcluster = membership[i]
        nc[kcluster] += 1 
        for c in range(2):
            sumc[kcluster][c] +=   float(d_values[i][c])

    #calculate the cluster centers       
    for q in range(k):
        for c in range(2):
            center_array[q][c] = sumc[q][c] / nc[q]

    return center_array


def calc_distances(n, k, membership, d_values, center_array):
    sum_distances = numpy.zeros(k)
    total_distance = 0
    
    for i in range(n):
        km = membership[i]
        for c in range(2):
            sum_distances[km] += (center_array[km][c] - float(d_values[i][c]))**2

    for q in range(k):
        total_distance += sum_distances[q]

    return total_distance

def calc_mbd_2(n, neighbor_count, membership, d_m, neighbor_array):
    mbd = 0
    for i in range(neighbor_count):
        a = neighbor_array[1][i]
        b = neighbor_array[2][i]
        if membership[a] != membership[b]:
            for c in range(4):
                mbd += (float(d_m[a][c]) - float(d_m[b][c]))**2
    mbd = mbd / n
    return mbd

def plot_data(n, k, membership, coordinates, x_centers, y_centers):
    colors = ['r', 'b', 'g', 'y', 'c', 'm', 'maroon', 'darkorange', 'olive', 'darkslategrey', 
    'navy', 'indigo', 'fuchsia', 'sienna', 'yellow', 'lime', 'skyblue','blueviolet', 'peru', 'plum']
    fig, sub = plt.subplots()
    for q in range(k):
        points = numpy.array([coordinates[i] for i in range(n) if membership[i] == q])
        sub.scatter(points[:, 0], points[:, 1], s = 5, c = colors[q])
    sub.scatter(x_centers, y_centers, marker = '*', s = 100, c = 'black')            
    #plt.show()

def test_dominance(cluster1, cluster2):
    if (cluster1.total_distance < cluster2.total_distance and cluster1.mbd < cluster2.mbd):
        return 1
    elif cluster1.total_distance >= cluster2.total_distance and cluster1.mbd >= cluster2.mbd:
        return 2
    else:
        return 0

def eliminate_cluster(Clusters, npop):
    Clusters.sort(key=Cluster.total_distance)
    Max_E = Clusters[npop].total_distance
    Max_MBD = Clusters[0].mbd
    Clusters[0].distance = 2
    Clusters[npop].distance = 2
    lcd = 0
    for i in range(1, npop):
        Clusters[i].distance = (abs(Clusters[i-1].total_distance - Clusters[i+1].total_distance) / Max_E) + (abs(Clusters[i-1].mbd - Clusters[i+1].mbd) / Max_MBD)
        if (Clusters[i].distance < Clusters[lcd].distance):
            lcd = i
    del Clusters[lcd]
    


def main():
# Imports data and parses it accordingly

    random.seed(2985)
    input_data = pd.read_csv('201806sum_all_data.csv')
    neighbor_matrix=numpy.loadtxt("adjacency.csv", dtype='i', delimiter=',')
    #input_data = pd.read_csv('new_data.csv')
    #neighbor_matrix=numpy.loadtxt("neighbors.csv", dtype='i', delimiter=',')
    nm_n = neighbor_matrix.shape[0]
    nm_m = neighbor_matrix.shape[1]
  
    neighbor_array = [[]]
    neighbor_x = []
    neighbor_y = []
    
    for i in range(nm_n):
        for j in range(nm_m):
            if neighbor_matrix[i][j] == 1:
                neighbor_x.append(i)
                neighbor_y.append(j)
               

    neighbor_array.append(neighbor_x)
    neighbor_array.append(neighbor_y)
    neighbor_count = len(neighbor_x)

   

    n = input_data.shape[0]
    print(n)
    m = input_data.shape[1]
    k = 15
    npop = 10
    
    k_array = numpy.zeros(k)
    for q in range(k):
        k_array[q] = q
        
    x_values = input_data.iloc[:,0]
    y_values = input_data.iloc[:,1]
    d1 = input_data.iloc[:,0]
    d2 = input_data.iloc[:,1]
    m1 = input_data.iloc[:,2]
    m2 = input_data.iloc[:,3]
    coordinates = numpy.array(list(zip(x_values, y_values)))
    d_values = numpy.array(list(zip(d1, d2)))
    m_values = numpy.array(list(zip(m1, m2)))
    d1_dm = m_values.shape[1]
    d_m = numpy.array(list(zip(d1, d2, m1, m2)))
    pj = 100
    a_array = 0.95

    sum_in = 0
    for i in range(n):
        sum_in += m_values[i][0]
    avg_sum_in = sum_in / k

    print(avg_sum_in)
    #print(q)
    # we need to discuss this part of the code. I am not sure why you have something called coordinates. 

    # Creates first cluster instance
    Clusters = []
    for p in range(npop):
        cluster = Cluster(0, 0, n, m, k, input_data)
        # current_cluster =Cluster(0, 0, n, m, k, input_data)
        # new_cluster = Cluster(0, 0, n, m, k, input_data)
        # best_cluster = Cluster(0, 0, n, m, k, input_data)
        # cluster.assign_cluster()

        cluster.assign_cluster_construction(neighbor_array,neighbor_count)

        
        centers = calc_centers(n, k, cluster.membership, d_values)
        cluster.centers=centers
        sum_attributes = calc_sum_attributes(n,k,d1_dm,m_values,cluster.membership)
        tp = calc_total_penalty(k,pj,a_array,sum_attributes,avg_sum_in)
        cluster.tp = tp
        distance = calc_distances(n, k, cluster.membership, d_values, centers)
        cluster.total_distance = distance + tp
        mbd = calc_mbd_2(n, neighbor_count, cluster.membership, d_m, neighbor_array)
        cluster.mbd = mbd + tp
        cluster.identify_border(neighbor_array,neighbor_count)
        Clusters.append(cluster)
        #print(cluster)

    # New Optimization Loop
    improvement = 1
    random_n = [i for i in range(n)]
    random_i = [i for i in range(n)]
    shuffle(random_n)
    current_npop = npop


    n_iterations=0
    n_max = 50000
    for i in range(current_npop):
        print(str(Clusters[i].total_distance) + " " + str(Clusters[i].mbd) + " " + str(Clusters[i].tp))
        x_centers = []
        y_centers = []
        for q in range(k):
            x_centers.append(Clusters[i].centers[q][0])
            y_centers.append(Clusters[i].centers[q][1])
        plot_data(n, k, Clusters[i].membership, coordinates, x_centers, y_centers)
    plt.show()
    print ("-------------------------------DIVIDER---------------------------------------")

    
    while(n_iterations < n_max):
        improvement = 0

        pp = random.randint(0, current_npop-1)
        new_cluster =deepcopy(Clusters[pp])
        n_iterations=n_iterations+1
        
        

        #disabling this method
        if(1==0):
            #Increase new cluster membership
            ii = random.randint(0,n-1)
            new_cluster.membership[ii] = new_cluster.membership[ii] + 1
            if new_cluster.membership[ii] >= k:
                new_cluster.membership[ii] = 0

        #find a border point and set its cluster membership to the same one.      

        border_point=0              

        while(border_point==0):
            #Increase new cluster membership
            new_membership=-1
            shuffle(random_i)
            #identify data point to update. ii must be a border point. Break in first random data point. 
            for i in random_i:
                if (new_cluster.border[i] == 1):
                    ii=i
                    break
            #identify data point to update. ii must be a border point     
            for j in random_i:
                if (neighbor_matrix[ii][j]==1 and new_cluster.membership[ii] != new_cluster.membership[j]):
                    #ii is a border point
                    border_point=1
                    new_membership=new_cluster.membership[j]
                    break
            
        if new_membership == -1:
            sys.exit()
        
        new_cluster.membership[ii] = new_membership
        
            

        #Recalculate new_cluster
        centers = calc_centers(n, k, new_cluster.membership, d_values)
        new_cluster.centers=centers
        sum_attributes = calc_sum_attributes(n,k,d1_dm,m_values,new_cluster.membership)
        tp = calc_total_penalty(k,pj,a_array,sum_attributes,avg_sum_in)
        new_cluster.tp = tp
        distance = calc_distances(n, k, new_cluster.membership, d_values, centers)
        new_cluster.total_distance = distance + tp
        mbd = calc_mbd_2(n, neighbor_count, new_cluster.membership, d_m, neighbor_array)
        new_cluster.mbd = mbd + tp

        #Dominance Test
        dominates = 0
        current_npop = len(Clusters)
        i = 0
        while i < current_npop:
            if (test_dominance(new_cluster, Clusters[i]) == 2 and dominates == 0):
                dominates = 2
                break
            if (test_dominance(new_cluster, Clusters[i]) == 1 and dominates == 0):
                new_cluster.identify_border(neighbor_array,neighbor_count)
                Clusters[i] = deepcopy(new_cluster)
                dominates = 1
                improvement = 1
            elif (test_dominance(new_cluster, Clusters[i]) == 1 and dominates == 1):
                del Clusters[i]
                improvement = 1
                current_npop = len(Clusters)
            i += 1
          

        current_npop = len(Clusters)
        if (dominates == 0 and current_npop < npop):
            new_cluster.identify_border(neighbor_array,neighbor_count)
            Clusters.append(deepcopy(new_cluster))
            current_npop = len(Clusters)
            
        if (dominates == 0 and current_npop >= npop):
            #p=random.randint(1,current_npop)
            #del Clusters[p]
            #eliminate_cluster(Clusters, npop)
            Clusters.sort(key=lambda Cluster: Cluster.total_distance)
            Max_E = Clusters[current_npop-1].total_distance
            Max_MBD = Clusters[0].mbd
            Clusters[0].distance = 2
            Clusters[current_npop-1].distance = 2
            lcd = 0
            for i in range(1, current_npop-1):
                Clusters[i].distance = (abs(Clusters[i-1].total_distance - Clusters[i+1].total_distance) / Max_E) + (abs(Clusters[i-1].mbd - Clusters[i+1].mbd) / Max_MBD)
                if (Clusters[i].distance < Clusters[lcd].distance):
                    lcd = i
            del Clusters[lcd]
            current_npop = len(Clusters)

        if(n_iterations == (n_max/2)):
            for i in range(current_npop):
                print(str(Clusters[i].total_distance) + " " + str(Clusters[i].mbd) + " "+ str(Clusters[i].tp))
                x_centers = []
                y_centers = []
                for q in range(k):
                    x_centers.append(Clusters[i].centers[q][0])
                    y_centers.append(Clusters[i].centers[q][1])
                plot_data(n, k, Clusters[i].membership, coordinates, x_centers, y_centers)
            plt.show()
            print ("-------------------------------DIVIDER---------------------------------------")

    #end of the while    

    for i in range(current_npop):
        print(str(Clusters[i].total_distance) + " " + str(Clusters[i].mbd) + " " + str(Clusters[i].tp))
        x_centers = []
        y_centers = []
        for q in range(k):
            x_centers.append(Clusters[i].centers[q][0])
            y_centers.append(Clusters[i].centers[q][1])
        plot_data(n, k, Clusters[i].membership, coordinates, x_centers, y_centers)
    print ("-------------------------------DIVIDER---------------------------------------")
    plt.show()

                    
    #for q in range(k):
       # x_centers.append(current_cluster.centers[q][0])
       # y_centers.append(current_cluster.centers[q][1])
    #plot_data(n, k, current_cluster.membership, coordinates, x_centers, y_centers)



# Main
import numpy  
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt
import random
from random import shuffle
import itertools
q=[1,1]
main()
    
