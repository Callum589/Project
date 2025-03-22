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
        "Method", "CR", "PRD (%)", "WDD (%)", "Time (s)", "Memory (MB)"
    )
    print(header)
    print("-" * len(header))
    for method, metrics in results.items():
        mem_mb = metrics['mem'] / (1024*1024)
        print("{:<15s} {:>8.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>12.2f}".format(
            method, metrics['CR'], metrics['PRD'], metrics['WDD'], metrics['time'], mem_mb))
    print()

# -------------------------------
# Plot Metric Function: Boxplot, Violin, Histogram + Gaussian Fit
# -------------------------------
def plot_metric(overall_results, metric_key, title, ylabel, convert_func=None):
    """
    Plot a metric across methods using a Boxplot, Violin Plot, and Histogram with Gaussian Fit.
    
    Parameters:
      overall_results (dict): Dictionary with keys as method names and metric lists as values.
      metric_key (str): Metric key (e.g., 'CR', 'PRD', 'WDD', 'time', 'mem').
      title (str): Plot title.
      ylabel (str): Y-axis label.
      convert_func (function, optional): Function to convert metric values (e.g., bytes to MB).
    """
    methods = list(overall_results.keys())
    data = []
    for method in methods:
        values = np.array(overall_results[method][metric_key])
        if convert_func:
            values = convert_func(values)
        data.append(values)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].boxplot(data, labels=methods, patch_artist=True)
    axes[0].set_title(f"{title} - Boxplot", fontsize=10)
    axes[0].set_ylabel(ylabel, fontsize=10)
    axes[0].grid(alpha=0.3)
    
    axes[1].violinplot(data, showmeans=True, showextrema=True)
    axes[1].set_title(f"{title} - Violin Plot", fontsize=10)
    axes[1].set_xticks(np.arange(1, len(methods)+1))
    axes[1].set_xticklabels(methods, fontsize=8)
    axes[1].set_ylabel(ylabel, fontsize=10)
    axes[1].grid(alpha=0.3)
    
    for i, method in enumerate(methods):
        mu, sigma = norm.fit(data[i])
        bins = np.linspace(np.min(data[i]), np.max(data[i]), 15)
        n, bins, patches = axes[2].hist(data[i], bins=bins, alpha=0.3, density=True, label=method)
        y_fit = norm.pdf(bins, mu, sigma)
        axes[2].plot(bins, y_fit, '--', linewidth=2)
    axes[2].set_title(f"{title} - Histogram + Gaussian Fit", fontsize=10)
    axes[2].set_xlabel(ylabel, fontsize=10)
    axes[2].legend(methods, fontsize=8)
    
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# -------------------------------
# Print Overall Metrics Statistics
# -------------------------------
def print_overall_stats(overall_results):
    """
    Print overall statistics (mean, std, min, max) for each metric for each method.
    
    Parameters:
      overall_results (dict): Dictionary with keys as method names and metric lists as values.
    """
    print("\nOverall Metrics Statistics:")
    for method, metrics in overall_results.items():
        print(f"\nMethod: {method}")
        for key in metrics.keys():
            values = np.array(metrics[key])
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"  {key}: Mean = {mean_val:.2f}, Std = {std_val:.2f}, Min = {min_val:.2f}, Max = {max_val:.2f}")

# -------------------------------
# Scatter Plot: CR vs. PRD and CR vs. Dynamic Power
# -------------------------------
def plot_scatter_cr_vs_prd_and_power(overall_results, power_key="time"):
    """
    Create two scatter plots:
      1) Compression Ratio vs. PRD.
      2) Compression Ratio vs. Dynamic Power (using the specified power_key).
    
    Parameters:
      overall_results (dict): Dictionary with methods as keys and metric lists as values.
      power_key (str): Key for dynamic power (e.g., "time" or "mem").
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    ax2 = axes[1]
    
    ax1.set_title("Compression Ratio vs. PRD", fontsize=10)
    ax1.set_xlabel("Compression Ratio (CR)", fontsize=10)
    ax1.set_ylabel("PRD (%)", fontsize=10)
    
    ax2.set_title("Compression Ratio vs. Dynamic Power", fontsize=10)
    ax2.set_xlabel("Compression Ratio (CR)", fontsize=10)
    ax2.set_ylabel(f"{power_key} (units)", fontsize=10)
    
    markers = ["o", "s", "^", "D", "x", "p"]
    colors  = ["blue", "red", "green", "purple", "orange", "brown"]
    
    for i, (method, data_dict) in enumerate(overall_results.items()):
        if len(data_dict["CR"]) == 0:
            continue
        cr_array  = np.array(data_dict["CR"])
        prd_array = np.array(data_dict["PRD"])
        power_array = np.array(data_dict[power_key])
        ax1.scatter(cr_array, prd_array,
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    alpha=0.7,
                    label=method)
        ax2.scatter(cr_array, power_array,
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    alpha=0.7,
                    label=method)
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(handles1, labels1, loc="best", fontsize=8)
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles2, labels2, loc="best", fontsize=8)
    
    plt.tight_layout()
    plt.show()

# -------------------------------
# Data Loading Function
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
        raise FileNotFoundError(f"ECG data file not found: {file_path}.")
    record = wfdb.rdrecord(os.path.join(path, record_name))
    signals = record.p_signal[:, 0]
    return signals

# -------------------------------
# L2SB Compression/Decompression Functions
# -------------------------------
def l2sb_compress(signal, thresholds, quant_steps):
    """
    Compress the signal using a Log-2 Sub-Band approach.
    
    Parameters:
      signal (numpy.ndarray): 1D array of ECG samples.
      thresholds (list): Two increasing threshold values [t1, t2].
      quant_steps (list): Quantisation step sizes for each sub-band.
      
    Returns:
      list: Tuples (subband, q_value, sign) for each sample.
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
      encoded (list): Tuples (subband, q_value, sign).
      thresholds (list): Two threshold values [t1, t2].
      quant_steps (list): Quantisation step sizes for each sub-band.
      
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
# Metrics Functions: PRD, Compression Ratio, and WDD
# -------------------------------
def compute_prd(original, reconstructed):
    """
    Compute the Percentage Root-Mean-Square Difference (PRD) between the original and reconstructed signals.
    
    PRD = sqrt(sum((x - x_rec)^2) / sum(x^2)) * 100
    
    Parameters:
      original (numpy.ndarray): Original ECG signal.
      reconstructed (numpy.ndarray): Reconstructed ECG signal.
      
    Returns:
      float: PRD in percentage.
    """
    error = original - reconstructed
    prd = np.sqrt(np.sum(error**2) / np.sum(original**2)) * 100
    return prd

def compute_compression_ratio(encoded, num_samples, bit_depth=16):
    """
    Estimate the compression ratio (CR) given the encoded signal.
    
    Parameters:
      encoded (list): List of encoded tuples.
      num_samples (int): Number of samples in the original signal.
      bit_depth (int): Bit-depth of the original signal.
      
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
    
    WDD = sqrt(sum(w*(x - x_rec)^2) / sum(w*(x^2))) * 100, where weight w = 2 if abs(x) exceeds threshold_factor * max(abs(original)), else 1.
    
    Parameters:
      original (numpy.ndarray): Original ECG signal.
      reconstructed (numpy.ndarray): Reconstructed ECG signal.
      threshold_factor (float): Fraction of max amplitude for high weight.
      
    Returns:
      float: WDD in percentage.
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
# Huffman Compression Functions
# -------------------------------
Node = namedtuple("Node", ["freq", "symbol", "left", "right"])

def huffman_tree(symbols_freq):
    """
    Build a Huffman tree from symbol frequencies.
    
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
    Recursively determine code lengths for each symbol in the Huffman tree.
    
    Parameters:
      root (Node): Root of the Huffman tree.
      prefix (str): Current code prefix.
      
    Returns:
      dict: Mapping from symbol to its code length.
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
    Estimate the average number of bits per symbol using Huffman coding.
    
    Parameters:
      signal (numpy.ndarray): Original ECG signal (quantized to integers).
      
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
# Genetic Algorithm for Adaptive L2SB (Improved)
# -------------------------------
def evaluate_individual(ind, signal, quant_steps, prd_limit):
    """
    Evaluate an individual [t1, t2] for L2SB thresholds with an adaptive PRD penalty.
    
    Parameters:
      ind (list): Candidate thresholds [t1, t2].
      signal (numpy.ndarray): Original ECG signal.
      quant_steps (list): Quantisation step sizes.
      prd_limit (float): Maximum acceptable PRD.
      
    Returns:
      float: Fitness value (CR adjusted for PRD).
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
        return cr * (prd_limit / prd)
    else:
        return cr

def ga_optimize_l2sb(signal, quant_steps, prd_limit=5.0, pop_size=20, generations=30, cr_stop_loss=None):
    """
    Optimize L2SB thresholds using a Genetic Algorithm.
    
    Improvements include:
      - Parallel fitness evaluation.
      - Adaptive PRD penalty.
      - Roulette wheel selection.
      - Gaussian noise mutation.
      - Surrogate ML model filtering.
      - Optional stop loss condition when CR >= cr_stop_loss is achieved.
      
    Parameters:
      signal (numpy.ndarray): Original ECG signal.
      quant_steps (list): Quantisation step sizes.
      prd_limit (float): Maximum acceptable PRD.
      pop_size (int): Population size.
      generations (int): Number of generations.
      cr_stop_loss (float or None): If set, GA stops early if a candidate achieves CR >= this value.
      
    Returns:
      dict: Best thresholds, CR, and PRD.
    """
    t1_min, t1_max = 0.05, 0.3
    t2_min, t2_max = 0.3, 1.0
    population = [[random.uniform(t1_min, t1_max), random.uniform(max(t2_min, random.uniform(t1_min, t1_max) + 0.01), t2_max)]
                  for _ in range(pop_size)]
    best_ind = None
    best_fit = -1

    for gen in range(generations):
        bar_length = 20
        progress = int((gen+1) / generations * bar_length)
        bar = "[" + "#" * progress + "-" * (bar_length - progress) + "]"
        sys.stdout.write(f"\rGA Generation: {bar} {gen+1}/{generations}")
        sys.stdout.flush()
        
        fitnesses = Parallel(n_jobs=-1)(
            delayed(evaluate_individual)(ind, signal, quant_steps, prd_limit) 
            for ind in population
        )
        for ind, fit in zip(population, fitnesses):
            if fit > best_fit:
                best_fit = fit
                best_ind = ind.copy()
        
        # Optional stop loss: if enabled and achieved, break early.
        if cr_stop_loss is not None and best_fit >= cr_stop_loss:
            print(f"\nAchieved desired compression ratio (CR >= {cr_stop_loss}). Stopping further optimization.")
            break
        
        X = np.array(population)
        y = np.array(fitnesses)
        surrogate = LinearRegression().fit(X, y)
        median_fitness = np.median(y)
        
        total_fit = sum(fitnesses)
        if total_fit == 0:
            probabilities = [1.0 / pop_size] * pop_size
        else:
            probabilities = [f / total_fit for f in fitnesses]
        new_population = [population[np.random.choice(range(pop_size), p=probabilities)].copy()
                          for _ in range(pop_size)]
        
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
        
        predicted = surrogate.predict(np.array(new_population))
        for i in range(pop_size):
            if predicted[i] < median_fitness:
                predicted[i] = 0.0
        total_pred = np.sum(predicted)
        if total_pred == 0:
            probabilities = [1.0 / pop_size] * pop_size
        else:
            probabilities = predicted / total_pred
        new_population = [new_population[np.random.choice(range(pop_size), p=probabilities)].copy()
                          for _ in range(pop_size)]
        population = new_population
    sys.stdout.write("\n")
    
    best_thresholds = best_ind
    encoded = l2sb_compress(signal, best_thresholds, quant_steps)
    rec = l2sb_decompress(encoded, best_thresholds, quant_steps)
    best_prd = compute_prd(signal, rec)
    best_cr = compute_compression_ratio(encoded, len(signal))
    return {"thresholds": best_thresholds, "CR": best_cr, "PRD": best_prd}

# -------------------------------
# VHDL Code Generation Function
# -------------------------------
def generate_vhdl_description(thresholds, quant_steps):
    """
    Generate a VHDL description for the L2SB compressor.
    
    This function includes:
      - An 8-bit counter.
      - A placeholder for L2SB compression logic using the given thresholds and quantisation steps.
    
    Parameters:
      thresholds (list): List containing two threshold values [t1, t2].
      quant_steps (list): List containing quantisation steps for each sub-band.
    
    Returns:
      str: A string containing the VHDL code.
    """
    vhdl_code = f"""
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity L2SB_Compressor is
    Port (
        clk         : in STD_LOGIC;
        reset       : in STD_LOGIC;
        data_in     : in STD_LOGIC_VECTOR(15 downto 0);
        compressed_out : out STD_LOGIC_VECTOR(15 downto 0);
        done        : out STD_LOGIC
    );
end L2SB_Compressor;

architecture Behavioral of L2SB_Compressor is
    -- 8-bit counter for control/timing
    signal counter : unsigned(7 downto 0) := (others => '0');
    -- Parameters for L2SB compression
    constant t1 : real := {thresholds[0]:.3f};
    constant t2 : real := {thresholds[1]:.3f};
    constant quant_step0 : real := {quant_steps[0]:.3f};
    constant quant_step1 : real := {quant_steps[1]:.3f};
    constant quant_step2 : real := {quant_steps[2]:.3f};
begin
    process(clk, reset)
    begin
        if reset = '1' then
            counter <= (others => '0');
        elsif rising_edge(clk) then
            counter <= counter + 1;
        end if;
    end process;
    
    -- Placeholder for L2SB Compression Logic:
    -- In a full implementation, you would include logic to:
    -- 1. Determine the sub-band based on thresholds t1 and t2.
    -- 2. Compute quantisation based on the corresponding quant_step.
    -- 3. Assemble the compressed output.
    compressed_out <= data_in;  -- Replace with actual compression logic.
    
    -- Example control: set done high when counter reaches its maximum value.
    done <= '1' when counter = X"FF" else '0';
end Behavioral;
"""
    return vhdl_code

# -------------------------------
# Evaluate a Single Record
# -------------------------------
def evaluate_record(record_name, path, run_original, run_ga, run_huff,
                    ga_generations, pop_size, quant_steps, prd_limit=5.0, cr_stop_loss=None):
    """
    Evaluate compression on a single ECG record using selected methods.
    
    Parameters:
      record_name (str): Record number.
      path (str): Dataset path.
      run_original (bool): Run Original L2SB.
      run_ga (bool): Run GA L2SB.
      run_huff (bool): Run Huffman.
      ga_generations (int): Number of GA generations.
      pop_size (int): GA population size.
      quant_steps (list): Quantisation steps.
      prd_limit (float): PRD threshold.
      cr_stop_loss (float or None): CR stop loss threshold to use in GA.
      
    Returns:
      dict: Metrics dictionary for each method.
    """
    print(f"Record {record_name}: Loading data...")
    signal = load_ecg_data(record_name=record_name, path=path)
    results = {}
    
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
    
    if run_ga:
        print(f"Record {record_name}: Running GA L2SB compression (generations={ga_generations}, pop_size={pop_size})...")
        def method_ga():
            ga_res = ga_optimize_l2sb(signal, quant_steps, prd_limit=prd_limit, pop_size=pop_size, generations=ga_generations, cr_stop_loss=cr_stop_loss)
            thresholds = ga_res['thresholds']
            encoded = l2sb_compress(signal, thresholds, quant_steps)
            rec = l2sb_decompress(encoded, thresholds, quant_steps)
            return {
                'CR': compute_compression_ratio(encoded, len(signal)),
                'PRD': compute_prd(signal, rec),
                'WDD': compute_wdd(signal, rec),
                'thresholds': thresholds
            }
        res, t, mem = measure_performance(method_ga)
        print(f"Record {record_name}: GA L2SB done in {t:.2f} s, mem {mem/(1024*1024):.2f} MB")
        results['GA L2SB'] = {**res, 'time': t, 'mem': mem}
    
    if run_huff:
        print(f"Record {record_name}: Running Huffman compression...")
        def method_huff():
            cr = compute_huffman_cr(signal, bit_depth=16)
            return {'CR': cr, 'PRD': 0.0, 'WDD': 0.0}
        res, t, mem = measure_performance(method_huff)
        print(f"Record {record_name}: Huffman done in {t:.2f} s, mem {mem/(1024*1024):.2f} MB")
        results['Huffman'] = {**res, 'time': t, 'mem': mem}
    
    return results

# -------------------------------
# Main Function
# -------------------------------
def main():
    print("Select which compression methods to run:")
    run_original = input("Run Original L2SB? (y/n): ").strip().lower().startswith('y')
    run_ga = input("Run GA L2SB? (y/n): ").strip().lower().startswith('y')
    run_huff = input("Run Huffman? (y/n): ").strip().lower().startswith('y')
    
    if not any([run_original, run_ga, run_huff]):
        print("No methods selected. Exiting.")
        return
    
    # Define record IDs based on specified ranges:
    record_ids = list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125)) \
                 + list(range(200, 204)) + [205] + list(range(207, 211)) + list(range(212, 216)) \
                 + [217] + list(range(219, 224)) + [228] + list(range(230, 235))
    path = './mit-bih-arrhythmia-database-1.0.0/'
    quant_steps = [0.01, 0.05, 0.1]
    prd_limit = 5.0
    # GA parameters:
    if run_ga:
        pop_size = int(input("Enter population size for GA (e.g., 10 for quick test, 20+ for normal): "))
        ga_generations = int(input("Enter number of GA generations (e.g., 5 for quick test, 30 for normal): "))
        use_stop_loss = input("Would you like to enable compression ratio stop loss? (y/n): ").strip().lower().startswith('y')
        if use_stop_loss:
            cr_stop_loss = float(input("Enter desired compression ratio stop loss value (e.g., 2.0): "))
        else:
            cr_stop_loss = None
    else:
        pop_size = 20
        ga_generations = 30
        cr_stop_loss = None
    
    overall_results = {
        'Original L2SB': {'CR': [], 'PRD': [], 'WDD': [], 'time': [], 'mem': []},
        'GA L2SB':       {'CR': [], 'PRD': [], 'WDD': [], 'time': [], 'mem': []},
        'Huffman':       {'CR': [], 'PRD': [], 'WDD': [], 'time': [], 'mem': []}
    }
    # Dictionary to store GA thresholds per record if available.
    ga_thresholds_results = {}
    
    valid_records = []
    total_records = len(record_ids)
    
    for idx, rid in enumerate(record_ids, start=1):
        rec_str = str(rid)
        print(f"\nProcessing record {rec_str} ({idx}/{total_records})...")
        try:
            res = evaluate_record(rec_str, path, run_original, run_ga, run_huff,
                                  ga_generations, pop_size, quant_steps, prd_limit, cr_stop_loss)
            for method in res:
                overall_results[method]['CR'].append(res[method]['CR'])
                overall_results[method]['PRD'].append(res[method]['PRD'])
                overall_results[method]['WDD'].append(res[method]['WDD'])
                overall_results[method]['time'].append(res[method]['time'])
                overall_results[method]['mem'].append(res[method]['mem'])
                if method == 'GA L2SB' and 'thresholds' in res[method]:
                    ga_thresholds_results[rec_str] = res[method]['thresholds']
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
    
    overall_results = {k: v for k, v in overall_results.items() if len(v['CR']) > 0}
    
    # Print overall statistics for all metrics.
    print_overall_stats(overall_results)
    
    # Generate scatter plots.
    plot_scatter_cr_vs_prd_and_power(overall_results, power_key="time")
    
    # Generate additional graphs for each metric.
    plot_metric(overall_results, 'CR', "Compression Ratio", "CR")
    plot_metric(overall_results, 'PRD', "Percentage Root-Mean-Square Difference", "PRD (%)")
    plot_metric(overall_results, 'WDD', "Weighted Diagnostic Distortion", "WDD (%)")
    plot_metric(overall_results, 'time', "Execution Time", "Time (s)")
    plot_metric(overall_results, 'mem', "Memory Usage", "Memory (MB)", convert_func=lambda x: x/(1024*1024))
    
    # Optional: Generate VHDL description from GA L2SB thresholds.
    if run_ga and ga_thresholds_results:
        generate_vhdl = input("Generate VHDL description from a GA L2SB record? (y/n): ").strip().lower().startswith('y')
        if generate_vhdl:
            record_choice = input("Enter the record number for which to generate VHDL code (e.g., 100): ").strip()
            if record_choice in ga_thresholds_results:
                thresholds = ga_thresholds_results[record_choice]
                vhdl_code = generate_vhdl_description(thresholds, quant_steps)
                # Optionally, save the VHDL code to a file:
                vhdl_filename = f"VHDL_L2SB_{record_choice}.vhd"
                with open(vhdl_filename, "w") as f:
                    f.write(vhdl_code)
                print(f"VHDL code generated and saved to {vhdl_filename}:\n")
                print(vhdl_code)
            else:
                print(f"No GA thresholds found for record {record_choice}.")
    
    print("Processing complete.")

if __name__ == '__main__':
    main()
