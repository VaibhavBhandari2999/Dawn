import math
import random
import numpy as np
from timeit import default_timer
import random
class SimAnneal(object):
    def __init__(self, coords, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        '''
        Initializing Everything
        '''
        self.coords = coords
        self.N = len(coords)
        self.T = math.sqrt(self.N) if T == -1 else T
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 0.00000001 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        self.dist_matrix = self.to_dist_matrix(coords)
        self.nodes = [i for i in range(self.N)]
        
        #Calling the Greedy's Algo function
        self.cur_solution = self.initial_solution()
        
        self.best_solution = list(self.cur_solution)
        self.cur_fitness = self.fitness(self.cur_solution)
        self.initial_fitness = self.cur_fitness
        self.best_fitness = self.cur_fitness
        self.fitness_list = [self.cur_fitness]

    def initial_solution(self):
        """
        Using greedy's algorithm to get an initial solution
        """
        cur_node = random.choice(self.nodes)
        solution = [cur_node]

        free_list = list(self.nodes)
        free_list.remove(cur_node)

        while free_list:
            closest_dist = min([self.dist_matrix[cur_node][j] for j in free_list])
            cur_node = self.dist_matrix[cur_node].index(closest_dist)
            if(cur_node in free_list):
                free_list.remove(cur_node)
                solution.append(cur_node)
            else:
                continue
        print("The greedy algorithm solution is::")
        print(solution)
        
        print("\n")
        return solution

    def dist(self, coord1, coord2):
        '''
        For finding Distance
        '''
        return round(math.sqrt(math.pow(coord1[1] - coord2[1], 2) + math.pow(coord1[2] - coord2[2], 2)), 4)

    def to_dist_matrix(self, coords):
        '''
        For finding the distance matrix for all the coordinates
        '''
        n = len(coords)
        mat = [[self.dist(coords[i], coords[j]) for i in range(n)] for j in range(n)]
        return mat

    def fitness(self, sol):
        """ Objective value of a solution """
        return round(sum([self.dist_matrix[sol[i - 1]][sol[i]] for i in range(1, self.N)]) +
                     self.dist_matrix[sol[0]][sol[self.N - 1]], 4)

    def p_accept(self, candidate_fitness):
        '''
        Function for accepting of candidate solutions which includes temperature
        '''
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    def accept(self, candidate):
        """
        Definetly accept if candidate is better than current. Use p_accept function if candidate is worse than current
        """
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness = candidate_fitness
            self.cur_solution = candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness = candidate_fitness
                self.best_solution = candidate

        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness = candidate_fitness
                self.cur_solution = candidate

    def anneal(self):
        '''
        Main Simulated Annealing Algo
        '''
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            candidate[i:(i + l)] = reversed(candidate[i:(i + l)])
            self.accept(candidate)
            self.T *= self.alpha
            self.iteration += 1

            self.fitness_list.append(self.cur_fitness)

    def outputs(self):
        '''
        For all the Outputs
        '''
        print("The simulated annealing solution::")
        print(self.best_solution)
        print("\n")
        print("Greedy Algorithm distance:",self.initial_fitness)
        print('Simulated Annnealing distance obtained: ', self.best_fitness)
        print('Improvement over greedy heuristic: ',round((self.initial_fitness - self.best_fitness),4))
        print("\n")
        x=[]
        q=[]
        for node in self.best_solution:
           x.append(self.coords[node-1])
           q.append(self.coords[node-1])
        for row in x:
            del row[0]
        y=[tuple(l) for l in x]
        for i in range(len(y)):
            for j in range(i+1,len(y)):
                if(y[i]==y[j]):
                    print("Same element found, Input is wrong")

        z=run_2opt(y,self.best_fitness)
        #Prints out all the nodes
        '''
        for node in z:
           print(node)
        print("\n")
        '''
        m=[list(l) for l in z]
        l=[]
        x1=self.coords
        print()
        for i in range(len(m)):
            for j in range(len(x1)):
                if(m[i][0]==x1[j][0] and m[i][1]==x1[j][1]):
                    l.append(j+1)
                    break
        print("Best path is:")
        print(l)
        return self.best_solution


def route_distance(route):
    '''
    To give distance between two sets of coordinates ot two cities
    '''
    dist = 0
    prev = route[-1]
    for node in route:
        dist += ((prev[0]-node[0])**2+(prev[1]-node[1])**2)**0.5
        prev = node
        
    return dist

def swap_2opt(route, i, k):
    '''
    Helps in swapping the edges joining coordinates
    '''
    assert i >= 0 and i < (len(route) - 1)
    assert k > i and k < len(route)
    new_route = route[0:i]
    new_route.extend(reversed(route[i:k + 1]))
    new_route.extend(route[k+1:])
    assert len(new_route) == len(route)
    return new_route

def run_2opt(route,a):
    '''
    Main 2opt algo
    '''
    improvement = True
    best_route = route
    best_distance = route_distance(route)
    while improvement: 
        improvement = False
        for i in range(len(best_route) - 1):
            for k in range(i+1, len(best_route)):
                new_route = swap_2opt(best_route, i, k)
                new_distance = route_distance(new_route)
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_route = new_route
                    improvement = True
                    break # if improvement found, return to the top of the wphile loop
            if improvement:
                break
    
    assert len(best_route) == len(route)
    if(a>route_distance(best_route)):
        print("2-opt improved the solution.")
        print("New Distance : " + str(route_distance(best_route)))
        return best_route
    else:
        print("2-opt didn't improve the solution.")
        return route

coords = []
a=[]
with open('wi29.tsp','r') as f:
    i = 0
    for line in f.readlines():
        line = [float(x.replace('\n','')) for x in line.split(' ')]
        #print(line)
        coords.append([])
        for j in range(0,3):
            coords[i].append(line[j])
        i += 1

#print(coords)        
if __name__ == '__main__':
    sa = SimAnneal(coords, stopping_iter = 5000000000000)
    start=default_timer()
    sa.anneal()
    sa.outputs()
    end=default_timer()
    print("\nTime to find optimal path : %.2f seconds" % (end-start))
