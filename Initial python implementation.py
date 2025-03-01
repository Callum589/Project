import os
import sys
import numpy as np
import wfdb
import math
import heapq
import random
import time
import tracemalloc
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from collections import Counter, namedtuple
from scipy.stats import norm
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

# -------------------------------
# Helper: Measure Performance
# -------------------------------
def measure_performance(func, *args, **kwargs):
    """
    Measure the run time and peak memory usage of a function call.
    
    Returns:
      (result, run_time_in_seconds, peak_memory_in_bytes)
    """
    start_time = time.perf_counter()
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.perf_counter()
    return result, (end_time - start_time), peak

# -------------------------------
# Helper: Print Record Metrics in Table Format
# -------------------------------
def print_record_metrics(record_name, results):
    """
    Print the metrics for a given record in a formatted table.
    
    Parameters:
      record_name (str): The record identifier.
      results (dict): Dictionary with method names as keys and metrics dictionary as value.
                      Expected keys: 'CR', 'PRD', 'WDD', 'time', 'mem'.
    """
    print(f"\nMetrics for record {record_name}:")
    header = "{:<15s} {:>8s} {:>10s} {:>10s} {:>10s} {:>12s}".format(
        "Method", "CR", "PRD (%)", "WDD (%)", "Time (s)", "Memory (MB)")
    print(header)
    print("-" * len(header))
    for method, metrics in results.items():
        mem_mb = metrics['mem'] / (1024*1024)
        print("{:<15s} {:>8.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>12.2f}".format(
            method, metrics['CR'], metrics['PRD'], metrics['WDD'], metrics['time'], mem_mb))
    print()

# -------------------------------
# 1. Data Loading
# -------------------------------
def load_ecg_data(record_name='100', path='./mit-bih-arrhythmia-database-1.0.0/'):
    """
    Load the first channel of an ECG record using WFDB.
    
    Parameters:
      record_name (str): Name of the ECG record.
      path (str): Path to the MIT-BIH Arrhythmia Database.
      
    Returns:
      numpy.ndarray: Array containing the ECG signal samples.
    """
    file_path = os.path.join(path, record_name + '.hea')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ECG data file not found: {file_path}. Check if the dataset is in the correct folder.")
    record = wfdb.rdrecord(os.path.join(path, record_name))
    signals = record.p_signal[:, 0]  # Extract first channel
    return signals

# -------------------------------
# 2. L2SB Compression / Decompression
# -------------------------------
def l2sb_compress(signal, thresholds, quant_steps):
    """
    Compress the signal using a simple Log-2 Sub-Band approach.
    
    Parameters:
      signal (numpy.ndarray): 1D array of ECG samples.
      thresholds (list): List of two increasing values [t1, t2].
      quant_steps (list): List of quantisation step sizes for each sub-band.
      
    Returns:
      list: A list of tuples (subband, q_value, sign) for each sample.
    """
    encoded = []
    for x in signal:
        sign = 1 if x >= 0 else -1
        abs_x = abs(x)
        if abs_x < thresholds[0]:
            subband = 0
            lower_bound = 0.0
        elif abs_x < thresholds[1]:
            subband = 1
            lower_bound = thresholds[0]
        else:
            subband = 2
            lower_bound = thresholds[1]
        Q = quant_steps[subband]
        q_value = int(round((abs_x - lower_bound) / Q))
        encoded.append((subband, q_value, sign))
    return encoded

def l2sb_decompress(encoded, thresholds, quant_steps):
    """
    Decompress the encoded data produced by l2sb_compress.
    
    Parameters:
      encoded (list): List of tuples (subband, q_value, sign).
      thresholds (list): List of two threshold values [t1, t2].
      quant_steps (list): List of quantisation step sizes for each sub-band.
      
    Returns:
      numpy.ndarray: Reconstructed ECG signal.
    """
    rec = []
    for (subband, q_value, sign) in encoded:
        if subband == 0:
            lower_bound = 0.0
        elif subband == 1:
            lower_bound = thresholds[0]
        else:
            lower_bound = thresholds[1]
        Q = quant_steps[subband]
        recon_val = lower_bound + q_value * Q
        rec.append(sign * recon_val)
    return np.array(rec)

# -------------------------------
# 3. Metrics: PRD, Compression Ratio, and WDD
# -------------------------------
def compute_prd(original, reconstructed):
    """
    Compute the Percentage Root-Mean-Square Difference (PRD) between the original and reconstructed signals.
    
    PRD = sqrt(sum((x - x_rec)^2)/sum(x^2)) * 100
    
    Parameters:
      original (numpy.ndarray): Original ECG signal.
      reconstructed (numpy.ndarray): Reconstructed ECG signal.
      
    Returns:
      float: PRD value in percentage.
    """
    error = original - reconstructed
    prd = np.sqrt(np.sum(error**2) / np.sum(original**2)) * 100
    return prd

def compute_compression_ratio(encoded, num_samples, bit_depth=16):
    """
    Estimate the compression ratio (CR) given the encoded signal.
    
    Assumes the original signal uses 'bit_depth' bits per sample.
    For each encoded sample:
      - 2 bits are used for the subband index.
      - A variable number of bits are used for the quantized value.
    
    CR = (total bits in original signal) / (total bits in compressed signal)
    
    Parameters:
      encoded (list): List of encoded tuples.
      num_samples (int): Number of samples in the original signal.
      bit_depth (int): Bit-depth of the original signal samples.
      
    Returns:
      float: Compression ratio.
    """
    total_bits = 0
    for (subband, q_value, sign) in encoded:
        bits_subband = 2
        bits_q = 1 if q_value == 0 else math.ceil(math.log2(q_value + 1))
        total_bits += bits_subband + bits_q
    original_bits = num_samples * bit_depth
    cr = original_bits / total_bits if total_bits > 0 else 0
    return cr

def compute_wdd(original, reconstructed, threshold_factor=0.5):
    """
    Compute the Weighted Diagnostic Distortion (WDD) between the original and reconstructed signals.
    
    A simple approach: if abs(x) > threshold_factor * max(abs(original)), assign weight=2, else weight=1.
    Then,
      WDD = sqrt(sum(w * (x - x_rec)^2) / sum(w * x^2)) * 100
    
    Parameters:
      original (numpy.ndarray): Original ECG signal.
      reconstructed (numpy.ndarray): Reconstructed ECG signal.
      threshold_factor (float): Fraction of max amplitude to decide high weight.
      
    Returns:
      float: WDD value in percentage.
    """
    orig = np.array(original)
    rec = np.array(reconstructed)
    max_val = np.max(np.abs(orig))
    weights = np.where(np.abs(orig) > threshold_factor * max_val, 2.0, 1.0)
    num = np.sum(weights * (orig - rec)**2)
    den = np.sum(weights * (orig**2))
    wdd = np.sqrt(num / den) * 100
    return wdd

# -------------------------------
# 4. Huffman Compression (Lossless)
# -------------------------------
Node = namedtuple("Node", ["freq", "symbol", "left", "right"])

def huffman_tree(symbols_freq):
    """
    Build a Huffman tree given symbol frequencies.
    
    Parameters:
      symbols_freq (dict): Dictionary mapping symbols to frequencies.
      
    Returns:
      Node: Root node of the Huffman tree.
    """
    heap = []
    for sym, freq in symbols_freq.items():
        heapq.heappush(heap, (freq, Node(freq, sym, None, None)))
    while len(heap) > 1:
        freq1, node1 = heapq.heappop(heap)
        freq2, node2 = heapq.heappop(heap)
        merged = Node(freq1 + freq2, None, node1, node2)
        heapq.heappush(heap, (merged.freq, merged))
    return heap[0][1]

def huffman_code_lengths(root, prefix=""):
    """
    Traverse the Huffman tree to determine code lengths for each symbol.
    
    Parameters:
      root (Node): Root of the Huffman tree.
      prefix (str): Current code prefix.
      
    Returns:
      dict: Mapping from symbol to its Huffman code length.
    """
    lengths = {}
    if root.symbol is not None:
        lengths[root.symbol] = len(prefix) if prefix != "" else 1
    else:
        lengths.update(huffman_code_lengths(root.left, prefix + "0"))
        lengths.update(huffman_code_lengths(root.right, prefix + "1"))
    return lengths

def huffman_compression_bits(signal):
    """
    Estimate the average number of bits per symbol when encoding the quantized signal using Huffman coding.
    
    Parameters:
      signal (numpy.ndarray): The original ECG signal (quantized to integers).
      
    Returns:
      float: Average bits per symbol.
    """
    quantized = np.round(signal).astype(int)
    freq_counter = Counter(quantized)
    total = sum(freq_counter.values())
    prob = {sym: freq/total for sym, freq in freq_counter.items()}
    root = huffman_tree(freq_counter)
    code_lengths = huffman_code_lengths(root)
    avg_bits = sum(prob[sym] * code_lengths[sym] for sym in prob)
    return avg_bits

def compute_huffman_cr(signal, bit_depth=16):
    """
    Compute the compression ratio using Huffman coding.
    
    Parameters:
      signal (numpy.ndarray): Original ECG signal.
      bit_depth (int): Bit depth per sample.
      
    Returns:
      float: Compression ratio.
    """
    avg_bits = huffman_compression_bits(signal)
    cr = bit_depth / avg_bits
    return cr

# -------------------------------
# 5. Genetic Algorithm for Adaptive L2SB (Improved)
# -------------------------------
def evaluate_individual(ind, signal, quant_steps, prd_limit):
    """
    Evaluate a candidate individual [t1, t2] for L2SB thresholds.
    
    Instead of discarding individuals with PRD above the limit, an adaptive penalty is applied:
    If PRD > prd_limit, fitness = CR * (prd_limit / PRD); else, fitness = CR.
    
    Parameters:
      ind (list): Candidate thresholds [t1, t2].
      signal (numpy.ndarray): Original ECG signal.
      quant_steps (list): Quantisation step sizes for each sub-band.
      prd_limit (float): Maximum acceptable PRD.
      
    Returns:
      float: Fitness value (compression ratio adjusted for PRD).
    """
    t1, t2 = ind
    if t2 <= t1:
        return 0.0
    thresholds = [t1, t2]
    encoded = l2sb_compress(signal, thresholds, quant_steps)
    rec_signal = l2sb_decompress(encoded, thresholds, quant_steps)
    prd = compute_prd(signal, rec_signal)
    cr = compute_compression_ratio(encoded, len(signal))
    if prd > prd_limit:
        fitness = cr * (prd_limit / prd)
    else:
        fitness = cr
    return fitness

def ga_optimize_l2sb(signal, quant_steps, prd_limit=5.0, pop_size=20, generations=30):
    """
    Use a genetic algorithm to optimize the L2SB thresholds for maximum compression ratio
    while maintaining PRD within the acceptable limit.
    
    Improvements included:
      - Parallelized fitness evaluation using joblib.
      - Adaptive PRD penalty.
      - Roulette wheel selection.
      - Gaussian noise mutation.
      - Surrogate ML model filtering.
    
    Parameters:
      signal (numpy.ndarray): Original ECG signal.
      quant_steps (list): Quantisation steps for each sub-band.
      prd_limit (float): Maximum acceptable PRD.
      pop_size (int): Population size for GA.
      generations (int): Number of generations (reduce for quick tests).
      
    Returns:
      dict: Dictionary with best thresholds, CR, and PRD.
    """
    t1_min, t1_max = 0.05, 0.3
    t2_min, t2_max = 0.3, 1.0

    # Initialize population
    population = [[random.uniform(t1_min, t1_max), random.uniform(max(t2_min, random.uniform(t1_min, t1_max) + 0.01), t2_max)]
                  for _ in range(pop_size)]
    
    best_ind = None
    best_fit = -1

    for gen in range(generations):
        # Display a loading bar on the same line
        bar_length = 20
        progress = int((gen+1) / generations * bar_length)
        bar = "[" + "#" * progress + "-" * (bar_length - progress) + "]"
        sys.stdout.write(f"\rGA Generation: {bar} {gen+1}/{generations}")
        sys.stdout.flush()
        
        # Parallel fitness evaluation
        fitnesses = Parallel(n_jobs=-1)(
            delayed(evaluate_individual)(ind, signal, quant_steps, prd_limit) for ind in population
        )
        
        # Update best individual
        for ind, fit in zip(population, fitnesses):
            if fit > best_fit:
                best_fit = fit
                best_ind = ind.copy()
        
        # Train surrogate model on current population
        X = np.array(population)
        y = np.array(fitnesses)
        surrogate = LinearRegression().fit(X, y)
        median_fitness = np.median(y)
        
        # Roulette wheel selection based on actual fitness
        total_fit = sum(fitnesses)
        if total_fit == 0:
            probabilities = [1.0 / pop_size] * pop_size
        else:
            probabilities = [f / total_fit for f in fitnesses]
        new_population = [population[np.random.choice(range(pop_size), p=probabilities)].copy() for _ in range(pop_size)]
        
        # Crossover (arithmetic)
        for i in range(0, pop_size, 2):
            if i+1 < pop_size and random.random() < 0.7:
                parent1 = new_population[i]
                parent2 = new_population[i+1]
                alpha = random.random()
                child1 = [alpha * parent1[0] + (1 - alpha) * parent2[0],
                          alpha * parent1[1] + (1 - alpha) * parent2[1]]
                child2 = [(1 - alpha) * parent1[0] + alpha * parent2[0],
                          (1 - alpha) * parent1[1] + alpha * parent2[1]]
                new_population[i] = child1
                new_population[i+1] = child2
        
        # Mutation with Gaussian noise
        for i in range(pop_size):
            if random.random() < 0.3:
                mutation = np.random.normal(0, 0.01, size=2)
                new_population[i][0] += mutation[0]
                new_population[i][1] += mutation[1]
                new_population[i][0] = min(max(new_population[i][0], t1_min), t1_max)
                new_population[i][1] = min(max(new_population[i][1], t2_min), t2_max)
                if new_population[i][1] <= new_population[i][0]:
                    new_population[i][1] = new_population[i][0] + 0.01
        
        # Surrogate filtering: Penalize individuals with predicted fitness below median
        predicted = surrogate.predict(np.array(new_population))
        for i in range(pop_size):
            if predicted[i] < median_fitness:
                predicted[i] = 0.0
        total_pred = np.sum(predicted)
        if total_pred == 0:
            probabilities = [1.0 / pop_size] * pop_size
        else:
            probabilities = predicted / total_pred
        new_population = [new_population[np.random.choice(range(pop_size), p=probabilities)].copy() for _ in range(pop_size)]
        
        population = new_population
    sys.stdout.write("\n")  # finish loading bar

    best_thresholds = best_ind
    best_encoded = l2sb_compress(signal, best_thresholds, quant_steps)
    best_rec = l2sb_decompress(best_encoded, best_thresholds, quant_steps)
    best_prd = compute_prd(signal, best_rec)
    best_cr = compute_compression_ratio(best_encoded, len(signal))
    return {"thresholds": best_thresholds, "CR": best_cr, "PRD": best_prd}

# -------------------------------
# 6. Deep Learning Autoencoder
# -------------------------------
def segment_signal(signal, frame_length):
    """
    Segment the ECG signal into non-overlapping frames.
    
    Parameters:
      signal (numpy.ndarray): The ECG signal.
      frame_length (int): Number of samples per segment.
      
    Returns:
      numpy.ndarray: 2D array where each row is a signal segment.
    """
    num_frames = len(signal) // frame_length
    segments = np.array(np.split(signal[:num_frames * frame_length], num_frames))
    return segments

def build_autoencoder(input_dim, latent_dim):
    """
    Build a simple dense autoencoder model for ECG compression.
    
    Parameters:
      input_dim (int): Dimensionality of the input (frame length).
      latent_dim (int): Dimension of the latent (compressed) representation.
      
    Returns:
      tensorflow.keras.models.Model: Compiled autoencoder model.
    """
    inp = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(inp)
    encoded = Dense(64, activation='relu')(encoded)
    latent = Dense(latent_dim, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(latent)
    decoded = Dense(128, activation='relu')(decoded)
    out = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(inputs=inp, outputs=out)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(segments, latent_dim, epochs=50, batch_size=32):
    """
    Train the autoencoder on segmented ECG data.
    
    Note: For a quick test, set 'epochs' to a small number (e.g., 5-10).
          For better reconstruction, use 50+ epochs.
    
    Parameters:
      segments (numpy.ndarray): 2D array of ECG signal segments.
      latent_dim (int): Dimension of the latent representation.
      epochs (int): Number of training epochs (reduce for quick tests).
      batch_size (int): Training batch size.
      
    Returns:
      tensorflow.keras.models.Model: Trained autoencoder model.
    """
    input_dim = segments.shape[1]
    autoencoder = build_autoencoder(input_dim, latent_dim)
    autoencoder.fit(segments, segments, epochs=epochs, batch_size=batch_size, verbose=0)
    return autoencoder

def evaluate_autoencoder(autoencoder, segments, latent_dim):
    """
    Evaluate the autoencoder by computing the average PRD and estimating the compression ratio.
    
    Parameters:
      autoencoder (tensorflow.keras.models.Model): Trained autoencoder model.
      segments (numpy.ndarray): Array of ECG signal segments.
      latent_dim (int): Known latent dimension used in the autoencoder.
      
    Returns:
      tuple: (average PRD, compression ratio) for the autoencoder.
    """
    reconstructed = autoencoder.predict(segments)
    prd_list = [compute_prd(orig, rec) for orig, rec in zip(segments, reconstructed)]
    avg_prd = np.mean(prd_list)
    input_dim = segments.shape[1]
    cr = input_dim / latent_dim
    return avg_prd, cr

# -------------------------------
# 7. Evaluate a Single Record
# -------------------------------
def evaluate_record(record_name, path, run_original, run_ga, run_auto, run_huff,
                    ga_generations, auto_epochs, pop_size, quant_steps, frame_length=256, prd_limit=5.0):
    """
    Evaluate compression on a single ECG record for any subset of:
      - Original L2SB
      - GA-Optimized L2SB
      - Autoencoder
      - Huffman
    
    Parameters:
      record_name (str): The MIT-BIH record number (e.g., '100').
      path (str): Path to the dataset.
      run_original (bool): Whether to run Original L2SB.
      run_ga (bool): Whether to run GA L2SB.
      run_auto (bool): Whether to run Autoencoder.
      run_huff (bool): Whether to run Huffman.
      ga_generations (int): Number of GA generations.
      auto_epochs (int): Number of autoencoder epochs.
      pop_size (int): Population size for GA.
      quant_steps (list): Quantisation steps for L2SB.
      frame_length (int): Segment size for autoencoder.
      prd_limit (float): Maximum PRD for GA optimization.
    
    Returns:
      dict: Contains the metrics (CR, PRD, WDD, time, mem) for each method that was run.
    """
    print(f"Record {record_name}: Loading data...")
    signal = load_ecg_data(record_name=record_name, path=path)
    results = {}
    
    # Original L2SB
    if run_original:
        print(f"Record {record_name}: Running Original L2SB compression...")
        def method_original():
            fixed_thresholds = [0.1, 0.4]
            encoded = l2sb_compress(signal, fixed_thresholds, quant_steps)
            rec = l2sb_decompress(encoded, fixed_thresholds, quant_steps)
            return {
                'CR': compute_compression_ratio(encoded, len(signal)),
                'PRD': compute_prd(signal, rec),
                'WDD': compute_wdd(signal, rec)
            }
        res, t, mem = measure_performance(method_original)
        print(f"Record {record_name}: Original L2SB done in {t:.2f} s, mem {mem/(1024*1024):.2f} MB")
        results['Original L2SB'] = {**res, 'time': t, 'mem': mem}
    
    # GA L2SB
    if run_ga:
        print(f"Record {record_name}: Running GA L2SB compression (generations={ga_generations}, pop_size={pop_size})...")
        def method_ga():
            ga_res = ga_optimize_l2sb(signal, quant_steps, prd_limit=prd_limit, pop_size=pop_size, generations=ga_generations)
            thresholds = ga_res['thresholds']
            encoded = l2sb_compress(signal, thresholds, quant_steps)
            rec = l2sb_decompress(encoded, thresholds, quant_steps)
            return {
                'CR': compute_compression_ratio(encoded, len(signal)),
                'PRD': compute_prd(signal, rec),
                'WDD': compute_wdd(signal, rec)
            }
        res, t, mem = measure_performance(method_ga)
        print(f"Record {record_name}: GA L2SB done in {t:.2f} s, mem {mem/(1024*1024):.2f} MB")
        results['GA L2SB'] = {**res, 'time': t, 'mem': mem}
    
    # Autoencoder
    if run_auto:
        print(f"Record {record_name}: Running Autoencoder compression (epochs={auto_epochs})...")
        def method_auto():
            segments = segment_signal(signal, frame_length)
            latent_dim = 32
            autoenc = train_autoencoder(segments, latent_dim, epochs=auto_epochs, batch_size=32)
            rec_prd, cr = evaluate_autoencoder(autoenc, segments, latent_dim)
            wdd_list = []
            recon_segments = autoenc.predict(segments)
            for orig, rec in zip(segments, recon_segments):
                wdd_list.append(compute_wdd(orig, rec))
            avg_wdd = np.mean(wdd_list)
            return {'CR': cr, 'PRD': rec_prd, 'WDD': avg_wdd}
        res, t, mem = measure_performance(method_auto)
        print(f"Record {record_name}: Autoencoder done in {t:.2f} s, mem {mem/(1024*1024):.2f} MB")
        results['Autoencoder'] = {**res, 'time': t, 'mem': mem}
    
    # Huffman
    if run_huff:
        print(f"Record {record_name}: Running Huffman compression...")
        def method_huff():
            cr = compute_huffman_cr(signal, bit_depth=16)
            return {'CR': cr, 'PRD': 0.0, 'WDD': 0.0}
        res, t, mem = measure_performance(method_huff)
        print(f"Record {record_name}: Huffman done in {t:.2f} s, mem {mem/(1024*1024):.2f} MB")
        results['Huffman'] = {**res, 'time': t, 'mem': mem}
    
    # Only print metrics here (once) in main, not inside this function.
    return results

# -------------------------------
# 8. Plotting Helper: Plot a Metric for Each Method
# -------------------------------
def plot_metric(overall_results, metric_key, title, ylabel, convert_func=None):
    """
    Plot the distribution of a given metric across methods using three subplots:
      Boxplot, Violin Plot, and Histogram + Gaussian Fit.
    
    Parameters:
      overall_results (dict): Dictionary with keys as method names and values as lists of metric values.
      metric_key (str): Key of the metric to plot (e.g., 'CR', 'PRD', 'WDD', 'time', 'mem').
      title (str): Title for the plots.
      ylabel (str): Y-axis label.
      convert_func (function, optional): Function to convert raw metric values (e.g., convert memory from bytes to MB).
    """
    methods = list(overall_results.keys())
    data = []
    for method in methods:
        values = np.array(overall_results[method][metric_key])
        if convert_func is not None:
            values = convert_func(values)
        data.append(values)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Boxplot
    axes[0].boxplot(data, labels=methods, patch_artist=True)
    axes[0].set_title(f"{title} - Boxplot")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(alpha=0.3)
    
    # Violin plot
    axes[1].violinplot(data, showmeans=True, showextrema=True)
    axes[1].set_title(f"{title} - Violin Plot")
    axes[1].set_xticks(np.arange(1, len(methods)+1))
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel(ylabel)
    axes[1].grid(alpha=0.3)
    
    # Histogram + Gaussian Fit (overlay for each method)
    for i, method in enumerate(methods):
        mu, sigma = norm.fit(data[i])
        bins = np.linspace(np.min(data[i]), np.max(data[i]), 15)
        n, bins, patches = axes[2].hist(data[i], bins=bins, alpha=0.3, density=True, label=method)
        y_fit = norm.pdf(bins, mu, sigma)
        axes[2].plot(bins, y_fit, '--', linewidth=2)
    axes[2].set_title(f"{title} - Histogram + Gaussian Fit")
    axes[2].set_xlabel(ylabel)
    axes[2].legend(methods)
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# -------------------------------
# 9. Main: User Prompt, Evaluate Records, and Plot Results
# -------------------------------
def main():
    # User Prompts
    print("Select which compression methods to run:")
    run_original = input("Run Original L2SB? (y/n): ").strip().lower().startswith('y')
    run_ga = input("Run GA L2SB? (y/n): ").strip().lower().startswith('y')
    pop_size = 20  # default
    if run_ga:
        pop_size = int(input("Enter population size for GA (e.g., 10 for quick test, 20+ for normal): "))
        ga_generations = int(input("Enter number of GA generations (e.g., 5 for quick test, 30 for normal): "))
    else:
        ga_generations = 30
    run_auto = input("Run Autoencoder? (y/n): ").strip().lower().startswith('y')
    if run_auto:
        auto_epochs = int(input("Enter number of autoencoder epochs (e.g., 5 for quick test, 50 for normal): "))
    else:
        auto_epochs = 50
    run_huff = input("Run Huffman? (y/n): ").strip().lower().startswith('y')
    
    if not any([run_original, run_ga, run_auto, run_huff]):
        print("No methods selected. Exiting.")
        return
    
    # Define record IDs according to specified ranges:
    # 100-109, 111-119, 121-124, 200-203, 205, 207-210, 212-215, 217, 219-223, 228, 230-234
    record_ids = list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125)) \
                 + list(range(200, 204)) + [205] + list(range(207, 211)) + list(range(212, 216)) \
                 + [217] + list(range(219, 224)) + [228] + list(range(230, 235))
    path = './mit-bih-arrhythmia-database-1.0.0/'
    quant_steps = [0.01, 0.05, 0.1]
    frame_length = 256
    prd_limit = 5.0
    
    overall_results = {}
    for method in ['Original L2SB', 'GA L2SB', 'Autoencoder', 'Huffman']:
        overall_results[method] = {'CR': [], 'PRD': [], 'WDD': [], 'time': [], 'mem': []}
    
    valid_records = []
    total_records = len(record_ids)
    for idx, rid in enumerate(record_ids, start=1):
        rec_str = str(rid)
        print(f"\nProcessing record {rec_str} ({idx}/{total_records})...")
        try:
            res = evaluate_record(rec_str, path, run_original, run_ga, run_auto, run_huff,
                                  ga_generations, auto_epochs, pop_size, quant_steps, frame_length, prd_limit)
            for method in res:
                overall_results[method]['CR'].append(res[method]['CR'])
                overall_results[method]['PRD'].append(res[method]['PRD'])
                overall_results[method]['WDD'].append(res[method]['WDD'])
                overall_results[method]['time'].append(res[method]['time'])
                overall_results[method]['mem'].append(res[method]['mem'])
            valid_records.append(rid)
            print(f"Record {rec_str} processed successfully.")
            print_record_metrics(rec_str, res)
        except FileNotFoundError:
            print(f"Record {rec_str} not found. Skipping...")
        except Exception as e:
            print(f"Error processing record {rec_str}: {e}")
    
    if not valid_records:
        print("No valid records processed. Exiting.")
        return
    
    # Print Summary Statistics
    print("\n=== Overall Metrics Statistics ===")
    for method in overall_results:
        print(f"{method}:")
        for metric in ['CR', 'PRD', 'WDD', 'time', 'mem']:
            arr = np.array(overall_results[method][metric])
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            var_val = np.var(arr)
            if metric == 'mem':  # convert memory from bytes to MB
                mean_val /= (1024*1024)
                std_val /= (1024*1024)
                var_val /= ((1024*1024)**2)
            print(f"  {metric} -> Mean: {mean_val:.2f}, Std: {std_val:.2f}, Var: {var_val:.2f}")
    
    # Plotting for Each Metric
    mem_convert = lambda x: x / (1024*1024)
    plot_metric(overall_results, 'CR', "Compression Ratio (CR) Across Methods", "CR")
    plot_metric(overall_results, 'PRD', "Percentage Root-Mean-Square Difference (PRD) Across Methods", "PRD (%)")
    plot_metric(overall_results, 'WDD', "Weighted Diagnostic Distortion (WDD) Across Methods", "WDD (%)")
    plot_metric(overall_results, 'time', "Run Time Across Methods", "Time (seconds)")
    plot_metric(overall_results, 'mem', "Memory Usage Across Methods", "Memory (MB)", convert_func=mem_convert)

if __name__ == "__main__":
    main()
