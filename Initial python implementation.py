import numpy as np
import wfdb
import heapq
import matplotlib.pyplot as plt
from collections import defaultdict
from deap import base, creator, tools, algorithms
import random
import os

# Load ECG Data from MIT-BIH Arrhythmia Database
def load_ecg_data(record_name='100', path='./mit-bih-arrhythmia-database-1.0.0/'):
    file_path = os.path.join(path, record_name + '.hea')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ECG data file not found: {file_path}. Check if the dataset is in the correct folder.")
    
    record = wfdb.rdrecord(os.path.join(path, record_name))
    signals = record.p_signal[:, 0]  # Extract first channel
    return signals

# Load multiple ECG records
def load_multiple_ecg_records(record_names, path='./mit-bih-arrhythmia-database-1.0.0/'):
    all_signals = {}
    for record_name in record_names:
        try:
            all_signals[record_name] = load_ecg_data(record_name, path)
        except FileNotFoundError as e:
            print(e)
    return all_signals

# Improved Log-2 Sub-Band Encoding (Adaptive Quantization)
def l2sb_encode(signal, param1=2, param2=4, sub_band_count=4):
    sub_band_size = len(signal) // sub_band_count
    compressed = np.zeros_like(signal)
    
    for i in range(sub_band_count):
        start_idx = i * sub_band_size
        end_idx = (i + 1) * sub_band_size if i != sub_band_count - 1 else len(signal)
        sub_band = signal[start_idx:end_idx]
        
        # Apply adaptive quantization to each sub-band
        sub_band_max = np.max(np.abs(sub_band))
        scale_factor = np.log2(sub_band_max + 1) / param1 if sub_band_max > 0 else 1
        
        compressed[start_idx:end_idx] = np.floor(np.log2(np.abs(sub_band) + 1) / scale_factor) * param2
    
    return compressed

# Reconstruction function to evaluate signal fidelity
def l2sb_decode(compressed_signal, param1=2, param2=4):
    reconstructed = (2 ** (compressed_signal / param2 * param1)) - 1
    return reconstructed

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

# Compression Performance Metrics
def compression_ratio(original, compressed):
    return len(set(compressed)) / len(set(original))

def mean_squared_error(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def visualize_compression_comparison(original, compressed, reconstructed):
    plt.figure(figsize=(12, 6))
    plt.plot(original, label='Original ECG Signal', color='blue', alpha=0.7)
    plt.plot(compressed, label='L2SB Compressed Signal', color='red', linestyle='dashed', alpha=0.7)
    plt.plot(reconstructed, label='Reconstructed Signal', color='green', linestyle='dotted', alpha=0.7)
    plt.legend()
    plt.show()

# Genetic Algorithm for Parameter Optimization
def fitness_function(individual):
    param1, param2 = individual
    encoded = l2sb_encode(ecg_data, param1, param2)
    reconstructed = l2sb_decode(encoded, param1, param2)
    return compression_ratio(ecg_data, encoded) + 0.3 * mean_squared_error(ecg_data, reconstructed),

# Set up DEAP for Genetic Algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 1, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=3, up=7, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def optimize_l2sb_parameters(ecg_data, generations=10, population_size=20):
    population = toolbox.population(n=population_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, 
                        stats=None, halloffame=None, verbose=True)
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual

if __name__ == "__main__":
    # Load multiple ECG records
    records_to_test = ['100', '101', '102', '234']
    ecg_data_sets = load_multiple_ecg_records(records_to_test)
    
    for record_name, ecg_data in ecg_data_sets.items():
        print(f"Processing Record: {record_name}")
        
        # Apply L2SB Encoding
        encoded_signal = l2sb_encode(ecg_data)
        
        # Optimize L2SB Parameters
        best_params = optimize_l2sb_parameters(ecg_data)
        print(f"Optimized L2SB Parameters for {record_name}: {best_params}")
        
        # Reconstruct signal
        reconstructed_signal = l2sb_decode(encoded_signal, *best_params)
        
        # Visualize
        visualize_compression_comparison(ecg_data, encoded_signal, reconstructed_signal)
        
        # Print Compression Ratio & MSE
        print(f"Compression Ratio for {record_name}: {compression_ratio(ecg_data, encoded_signal)}")
        print(f"Mean Squared Error for {record_name}: {mean_squared_error(ecg_data, reconstructed_signal)}")
