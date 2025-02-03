import numpy as np
import wfdb
import heapq
from collections import defaultdict
from deap import base, creator, tools, algorithms
import random

# Load ECG Data from MIT-BIH Arrhythmia Database
def load_ecg_data(record_name='100', path='./mitdb/'):
    record = wfdb.rdrecord(path + record_name)
    signals = record.p_signal[:, 0]  # Extract first channel
    return signals

# Log-2 Sub-Band Encoding (Simplified Example)
def l2sb_encode(signal, param1=2, param2=4):
    compressed = np.floor(np.log2(np.abs(signal) + 1) / param1) * param2
    return compressed

# Huffman Compression
def build_huffman_tree(data):
    frequency = defaultdict(int)
    for value in data:
        frequency[value] += 1
    
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    huff_dict = dict(heap[0][1:])
    return huff_dict

def huffman_encode(data, huff_dict):
    return ''.join(huff_dict[value] for value in data)

# Genetic Algorithm for Parameter Optimization
def fitness_function(individual):
    param1, param2 = individual
    encoded = l2sb_encode(ecg_data, param1, param2)
    compressed_size = len(set(encoded))  # Rough compression estimate
    return compressed_size,

# Set up DEAP for Genetic Algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 1, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=1, up=10, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def optimize_l2sb_parameters(ecg_data, generations=10, population_size=20):
    population = toolbox.population(n=population_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, 
                        stats=None, halloffame=None, verbose=True)
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual

if __name__ == "__main__":
    # Load ECG data
    ecg_data = load_ecg_data()
    
    # Apply L2SB Encoding (Fixed Parameters)
    encoded_signal = l2sb_encode(ecg_data)
    
    # Apply Huffman Compression
    huff_tree = build_huffman_tree(encoded_signal)
    huff_encoded_signal = huffman_encode(encoded_signal, huff_tree)
    
    # Optimize L2SB Parameters
    best_params = optimize_l2sb_parameters(ecg_data)
    print("Optimized L2SB Parameters:", best_params)
