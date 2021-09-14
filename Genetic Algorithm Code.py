# **************************************** BINARY CODED GENETIC ALGORITHM *******************************************
# *******************************************************************************************************************
# NAME: Suraj Kiran Shah                       ROLL NO.: 204103334                      DEPT.: Mechanical Engineering
# *******************************************************************************************************************
import numpy as np
import random as rd
import math as math
import matplotlib.pyplot as plot

# USER INPUT *********************************************************************************************************
N = 100  # Population Size
accuracy_X1 = 0.000001  # Desire accuracy level of x1
accuracy_X2 = 0.000001  # Desire accuracy level of x2
Pc = 0.9  # Crossover Probability
Pm = 0.3  # Mutation Probability
Max_Generation = 100

# Geometric Constraints
X1max = 0.5
X1min = 0
X2max = 0.5
X2min = 0


# FUNCTIONS **********************************************************************************************************
# Objective Function
def fitness(x1, x2):
    return 1 / (1 + (x1 + x2 - 2 * x1 * x1 - x2 * x2 + x1 * x2))


# Single pt and Multi pt Crossover Function
def single_pt_crossover(A, B, x):
    A_new = np.append(A[:x], B[x:])
    B_new = np.append(B[:x], A[x:])
    return A_new, B_new


def multi_pt_crossover(A, B, X):
    for i in X:
        A, B = single_pt_crossover(A, B, i)
    return A, B


# Finding length of string *******************************************************************************************
g = (X1max - X1min) / accuracy_X1
l1 = math.ceil(math.log(g, 2))
h = (X2max - X2min) / accuracy_X2
l2 = math.ceil(math.log(h, 2))
L = l1 + l2

# Initializing a population of solution at random (Generation=0) *****************************************************
G = []
for i in range(N):
    temp = []
    for j in range(L):
        temp.append(rd.randint(0, 1))
    G.append(temp)

# *************************************************** MAIN PROGRAM ****************************************************
# Initializing for plotting
Avg_Fitness_List = []
Max_Fitness_List = []
Min_Fitness_List = []
Optimal_Solution_List = []
Generation_List = []
# Initializing for while loop
Generation = 0

while Generation < Max_Generation:
    # REPRODUCTION ****************************************************************************************************
    # Finding Real Value of variables from coded string
    decoded = []
    real = []

    for i in range(N):
        temp = []
        for j in range(l1):
            temp.append(G[i][j])
        a = int("".join(str(x) for x in temp), 2)
        temp = []
        for j in range(l2):
            temp.append(G[i][j + l1])
        b = int("".join(str(x) for x in temp), 2)
        list1 = [a, b]
        decoded.append(list1)

    for i in range(N):
        a = X1min + (X1max - X1min) * (decoded[i][0]) / (pow(2, l1) - 1)
        b = X2min + (X2max - X2min) * (decoded[i][1]) / (pow(2, l2) - 1)
        list1 = [a, b]
        real.append(list1)

    # Calculating fitness value of generation
    Fitness = [0] * N
    for i in range(N):
        Fitness[i] = fitness(real[i][0], real[i][1])

    # Finding maximum, minimum and average fitness value of generation
    Sum_Fitness = sum(Fitness)
    Avg_Fitness = Sum_Fitness / N
    Avg_Fitness_List.append(Avg_Fitness)

    Max_Fitness = max(Fitness)
    Max_Fitness_List.append(Max_Fitness)

    Min_Fitness = min(Fitness)
    Min_Fitness_List.append(Min_Fitness)

    # Finding optimal solution of generation
    j = Fitness.index(min(Fitness))
    Optimal_Solution = real[j]
    Optimal_Solution_List.append(Optimal_Solution)

    # Roulette Wheel: Selection of Mating Pool
    Probability = [0] * N

    for i in range(N):
        Probability[i] = Fitness[i] / Sum_Fitness

    Cum_Probability = [sum(Probability[0:x:1]) for x in range(0, N + 1)]

    Selection = []
    Mating_Pool = []

    for i in range(N):
        r = rd.uniform(0, 1)
        for j in range(N):
            if r > Cum_Probability[j] and r <= Cum_Probability[j+1]:
                temp = j
                break
        Selection.append(temp)

    for i in range(N):
        temp = []
        temp = G[Selection[i]]
        Mating_Pool.append(temp)

    # CROSSOVER *******************************************************************************************************
    # Randomly selecting mating pairs for crossover
    Mating_Pairs = []
    Solution_Number = list(range(0, N))
    for i in range(int(N / 2)):
        a = rd.choice(Solution_Number)
        Solution_Number.remove(a)
        b = rd.choice(Solution_Number)
        Solution_Number.remove(b)
        temp = [a, b]
        Mating_Pairs.append(temp)

    # Crossover Process
    Children_Solution = []
    Crossover_Site = list(range(1, L))

    for i in range(int(N / 2)):
        r = rd.uniform(0, 1)
        if r <= Pc:
            a = rd.sample(Crossover_Site, 2)
            a.sort()
            M1 = Mating_Pool[Mating_Pairs[i][0]]
            M2 = Mating_Pool[Mating_Pairs[i][1]]
            A, B = multi_pt_crossover(M1, M2, a)
            Mating_Pool[Mating_Pairs[i][0]] = A.tolist()
            Mating_Pool[Mating_Pairs[i][1]] = B.tolist()

    Children_Solution = Mating_Pool

    # MUTATION ********************************************************************************************************
    for i in range(N):
        for j in range(L):
            r = rd.uniform(0, 1)
            if r <= Pm:
                if Children_Solution[i][j] == 0:
                    Children_Solution[i][j] = 1
                else:
                    Children_Solution[i][j] = 0

    # SURVIVOR OF THE FITTEST *****************************************************************************************
    Clubbed_G = G + Children_Solution

    # Finding Real Value of variables from coded string
    decoded = []
    real1 = []

    for i in range(2 * N):
        temp = []
        for j in range(l1):
            temp.append(Clubbed_G[i][j])
        a = int("".join(str(x) for x in temp), 2)
        temp = []
        for j in range(l2):
            temp.append(Clubbed_G[i][j + l1])
        b = int("".join(str(x) for x in temp), 2)
        list1 = [a, b]
        decoded.append(list1)

    for i in range(2 * N):
        a = X1min + (X1max - X1min) * (decoded[i][0]) / (pow(2, l1) - 1)
        b = X2min + (X2max - X2min) * (decoded[i][1]) / (pow(2, l2) - 1)
        list1 = [a, b]
        real1.append(list1)

    # Calculating fitness value of Clubbed Generation
    Fitness_Clubbed_G = [0] * (2 * N)
    for i in range(2 * N):
        Fitness_Clubbed_G[i] = fitness(real1[i][0], real1[i][1])

    # Making generation with best fitness value
    G_New = [[0] * L] * N
    for i in range(N):
        j = Fitness_Clubbed_G.index(max(Fitness_Clubbed_G))
        G_New[i] = Clubbed_G[j]
        Fitness_Clubbed_G[j] = 0

    G = G_New  # Assigning new generated population to G for next iteration

    Generation += 1
    Generation_List.append(Generation)
    print(str(Generation) + "\t" + str(Avg_Fitness))

# ********************************************** END OF MAIN PROGRAM *************************************************

# PLOTTING OF GRAPHS *************************************************************************************************
# Plotting of Fitness Value Vs Generation
plot.cla()
plot.plot(Generation_List, Max_Fitness_List, linewidth=2, label='Maximum Fitness')
plot.plot(Generation_List, Avg_Fitness_List, linewidth=2, label='Average Fitness')
plot.plot(Generation_List, Min_Fitness_List, linewidth=2, label='Minimum Fitness')

plot.title('Fitness value v/s Generation')
plot.xlabel('Generation')
plot.ylabel('Fitness Value')
plot.legend(loc='right')
plot.tight_layout()
plot.show()

# Plotting of last generation solution
real = np.asarray(real)
for i in range(N):
    plot.scatter(real[i][0], real[i][1], color='blue')
plot.title('X1 X2 values for last generation')
plot.xlabel('Value of X1')
plot.ylabel('Value of X2')
plot.show()

# Plotting of Optimal Solution Vs Generation
Optimal_Solution_List = np.asarray(Optimal_Solution_List)
plot.plot(Generation_List, Optimal_Solution_List[:, 0], linewidth=2, label='X1')
plot.plot(Generation_List, Optimal_Solution_List[:, 1], linewidth=2, label='X2')
plot.scatter(Generation_List, Optimal_Solution_List[:, 0], marker="o", linewidth=0)
plot.scatter(Generation_List, Optimal_Solution_List[:, 1], marker="o", linewidth=0)
plot.title('Optimal Solution v/s Generation')
plot.xlabel('Generation')
plot.ylabel('Optimal Solution')
plot.legend(loc='right')
plot.tight_layout()
plot.show()

# ********************************************************************************************************************