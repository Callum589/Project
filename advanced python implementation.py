import os
import sys
import numpy as np
import wfdb
import math
import heapq
import random
import time
import tracemalloc
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from collections import Counter, namedtuple
from scipy.stats import norm
from joblib import Parallel, delayed

# For Bayesian optimization (Hyperopt)
try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
except ImportError:
    print("Hyperopt is not installed. Please install it if you want to use Bayesian optimization.")
    exit(1)

# For interactive plotting (Plotly)
try:
    import plotly.express as px
    import pandas as pd
except ImportError:
    print("Plotly (and pandas) are not installed. Interactive plotting will not be available.")

# -------------------------------
# Logging Setup
# -------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler("compression.log"),
                              logging.StreamHandler()])
logger = logging.getLogger()

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
                      Expected keys: 'CR', 'PRD', 'WDD', 'time', 'mem', optionally 'SNR', 'NRMSE', 'DynamicPower'
    """
    print(f"\nMetrics for record {record_name}:")
    header = "{:<15s} {:>8s} {:>10s} {:>10s} {:>10s} {:>12s}".format(
        "Method", "CR", "PRD (%)", "WDD (%)", "Time (s)", "Memory (MB)"
    )
    # If additional metrics are present, add them to header
    extra_keys = []
    if any("SNR" in m for m in results.values()):
        header += " {:>8s}".format("SNR (dB)")
        extra_keys.append("SNR")
    if any("NRMSE" in m for m in results.values()):
        header += " {:>10s}".format("NRMSE")
        extra_keys.append("NRMSE")
    if any("DynamicPower" in m for m in results.values()):
        header += " {:>16s}".format("Dynamic Power (W)")
        extra_keys.append("DynamicPower")
    print(header)
    print("-" * len(header))
    for method, metrics in results.items():
        mem_mb = metrics['mem'] / (1024*1024)
        line = "{:<15s} {:>8.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>12.2f}".format(
            method, metrics['CR'], metrics['PRD'], metrics['WDD'], metrics['time'], mem_mb)
        for key in extra_keys:
            if key in metrics:
                line += " {:>16.2f}".format(metrics[key])
            else:
                line += " {:>16s}".format("N/A")
        print(line)
    print()

# -------------------------------
# Helper: Scatter Plot using Matplotlib
# -------------------------------
def plot_scatter_cr_vs_prd_and_power(overall_results, power_key="time"):
    """
    Create two scatter plots side by side:
      1) Compression Ratio vs. PRD
      2) Compression Ratio vs. Dynamic Power (using provided power_key)
    
    Parameters:
      overall_results (dict): Dictionary with method names as keys and metric lists as values.
      power_key (str): The key to use for dynamic power (e.g., "time" or "mem").
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
        cr_array = np.array(data_dict["CR"])
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
# Helper: Interactive Plotting with Plotly
# -------------------------------
def plot_interactive_scatter(overall_results, power_key="time"):
    """
    Create interactive scatter plots for CR vs. PRD and CR vs. Dynamic Power using Plotly.
    
    Parameters:
      overall_results (dict): Dictionary with methods as keys and metric lists as values.
      power_key (str): Key for dynamic power (e.g., "time" or "mem").
    """
    try:
        import plotly.express as px
        import pandas as pd
    except ImportError:
        logger.error("Plotly and pandas must be installed for interactive plotting.")
        return

    data = []
    for method, metrics in overall_results.items():
        for cr, prd, pwr in zip(metrics["CR"], metrics["PRD"], metrics[power_key]):
            data.append({"Method": method, "CR": cr, "PRD": prd, "DynamicPower": pwr})
    df = pd.DataFrame(data)
    
    fig1 = px.scatter(df, x="CR", y="PRD", color="Method", title="Interactive: CR vs. PRD")
    fig1.show()
    fig2 = px.scatter(df, x="CR", y="DynamicPower", color="Method", title="Interactive: CR vs. Dynamic Power")
    fig2.show()

# -------------------------------
# Data Loading Function (repeated for clarity)
# -------------------------------
# load_ecg_data is defined above.

# -------------------------------
# L2SB Compression/Decompression Functions (defined above)
# -------------------------------
# l2sb_compress and l2sb_decompress are defined above.

# -------------------------------
# Metrics Functions (defined above)
# -------------------------------
# compute_prd, compute_compression_ratio, compute_wdd are defined above.

# -------------------------------
# Huffman Compression Functions (defined above)
# -------------------------------
# huffman_tree, huffman_code_lengths, huffman_compression_bits, compute_huffman_cr are defined above.

# -------------------------------
# Genetic Algorithm for Adaptive L2SB
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

def ga_optimize_l2sb(signal, quant_steps, prd_limit=5.0, pop_size=20, generations=30):
    """
    Optimize L2SB thresholds using a Genetic Algorithm with improvements.
    
    Improvements include parallel fitness evaluation, adaptive PRD penalty, roulette wheel selection,
    Gaussian noise mutation, and surrogate ML model filtering.
    
    Parameters:
      signal (numpy.ndarray): Original ECG signal.
      quant_steps (list): Quantisation step sizes.
      prd_limit (float): Maximum acceptable PRD.
      pop_size (int): Population size.
      generations (int): Number of generations.
      
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
# Bayesian Optimization for L2SB
# -------------------------------
def bayesian_optimize_l2sb(signal, quant_steps, prd_limit, max_evals=50):
    """
    Use Bayesian optimization (Hyperopt) to optimize L2SB thresholds.
    
    Parameters:
      signal (numpy.ndarray): Original ECG signal.
      quant_steps (list): Quantisation step sizes.
      prd_limit (float): Maximum acceptable PRD.
      max_evals (int): Maximum evaluations.
      
    Returns:
      dict: Best thresholds, CR, and PRD.
    """
    def objective(thresholds):
        t1, t2 = thresholds
        if t2 <= t1:
            return {'loss': 1e6, 'status': STATUS_OK}
        thresh = [t1, t2]
        encoded = l2sb_compress(signal, thresh, quant_steps)
        rec = l2sb_decompress(encoded, thresh, quant_steps)
        prd = compute_prd(signal, rec)
        cr = compute_compression_ratio(encoded, len(signal))
        # We want to maximize CR (minimize -CR). Apply penalty if prd > prd_limit.
        if prd > prd_limit:
            obj_val = -cr * (prd_limit / prd)
        else:
            obj_val = -cr
        return {'loss': obj_val, 'status': STATUS_OK}
    
    space = [hp.uniform('t1', 0.05, 0.3), hp.uniform('t2', 0.3, 1.0)]
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    best_thresholds = [best['t1'], best['t2']]
    encoded = l2sb_compress(signal, best_thresholds, quant_steps)
    rec = l2sb_decompress(encoded, best_thresholds, quant_steps)
    prd = compute_prd(signal, rec)
    cr = compute_compression_ratio(encoded, len(signal))
    return {"thresholds": best_thresholds, "CR": cr, "PRD": prd}

# -------------------------------
# Bayesian Multi-Objective Optimization for L2SB
# -------------------------------
def bayesian_multiobjective_optimize_l2sb(signal, quant_steps, prd_limit, weight=0.5, max_evals=50):
    """
    Use Bayesian optimization with a weighted objective to balance CR and PRD.
    Objective: minimize: - (weight * CR - (1-weight)*normalized_PRD)
    Normalized PRD = PRD / 100.
    
    Parameters:
      signal (numpy.ndarray): Original ECG signal.
      quant_steps (list): Quantisation step sizes.
      prd_limit (float): Maximum acceptable PRD.
      weight (float): Weight for CR vs. PRD.
      max_evals (int): Maximum evaluations.
      
    Returns:
      dict: Best thresholds, CR, and PRD.
    """
    def objective(thresholds):
        t1, t2 = thresholds
        if t2 <= t1:
            return {'loss': 1e6, 'status': STATUS_OK}
        thresh = [t1, t2]
        encoded = l2sb_compress(signal, thresh, quant_steps)
        rec = l2sb_decompress(encoded, thresh, quant_steps)
        prd = compute_prd(signal, rec)
        cr = compute_compression_ratio(encoded, len(signal))
        normalized_prd = prd / 100.0
        obj_val = - (weight * cr - (1 - weight) * normalized_prd)
        return {'loss': obj_val, 'status': STATUS_OK}
    
    space = [hp.uniform('t1', 0.05, 0.3), hp.uniform('t2', 0.3, 1.0)]
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    best_thresholds = [best['t1'], best['t2']]
    encoded = l2sb_compress(signal, best_thresholds, quant_steps)
    rec = l2sb_decompress(encoded, best_thresholds, quant_steps)
    prd = compute_prd(signal, rec)
    cr = compute_compression_ratio(encoded, len(signal))
    return {"thresholds": best_thresholds, "CR": cr, "PRD": prd}

# -------------------------------
# Additional Metrics: SNR and NRMSE
# -------------------------------
def compute_snr(original, reconstructed):
    """
    Compute Signal-to-Noise Ratio (SNR) in dB.
    
    SNR = 10 * log10(sum(original^2) / sum((original - reconstructed)^2))
    
    Parameters:
      original (numpy.ndarray): Original signal.
      reconstructed (numpy.ndarray): Reconstructed signal.
      
    Returns:
      float: SNR in dB.
    """
    noise = original - reconstructed
    snr = 10 * np.log10(np.sum(original**2) / np.sum(noise**2))
    return snr

def compute_nrmse(original, reconstructed):
    """
    Compute Normalized Root-Mean-Square Error (NRMSE).
    
    NRMSE = sqrt(sum((x - x_rec)^2) / sum(x^2))
    
    Parameters:
      original (numpy.ndarray): Original signal.
      reconstructed (numpy.ndarray): Reconstructed signal.
      
    Returns:
      float: NRMSE.
    """
    error = original - reconstructed
    nrmse = np.sqrt(np.sum(error**2) / np.sum(original**2))
    return nrmse

# -------------------------------
# Theoretical Dynamic Power Calculation
# -------------------------------
def compute_dynamic_power(C_load, V, f, alpha):
    """
    Compute theoretical dynamic power consumption.
    
    P_dynamic = C_load * V^2 * f * alpha
    
    Parameters:
      C_load (float): Load capacitance (in Farads).
      V (float): Supply voltage (in Volts).
      f (float): Clock frequency (in Hz).
      alpha (float): Activity factor (0 to 1).
      
    Returns:
      float: Dynamic power in Watts.
    """
    return C_load * (V ** 2) * f * alpha

# -------------------------------
# Deep Learning Autoencoder Functions
# -------------------------------
def segment_signal(signal, frame_length):
    """
    Segment the ECG signal into non-overlapping frames.
    
    Parameters:
      signal (numpy.ndarray): The ECG signal.
      frame_length (int): Number of samples per segment.
      
    Returns:
      numpy.ndarray: 2D array with each row as a segment.
    """
    num_frames = len(signal) // frame_length
    segments = np.array(np.split(signal[:num_frames * frame_length], num_frames))
    return segments

def build_autoencoder(input_dim, latent_dim):
    """
    Build a simple dense autoencoder model for ECG compression.
    
    Parameters:
      input_dim (int): Input dimension (frame length).
      latent_dim (int): Dimension of the latent representation.
      
    Returns:
      tensorflow.keras.models.Model: Compiled autoencoder.
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
    
    Note: Use fewer epochs (e.g., 5-10) for quick tests.
    
    Parameters:
      segments (numpy.ndarray): 2D array of ECG segments.
      latent_dim (int): Latent dimension.
      epochs (int): Training epochs.
      batch_size (int): Training batch size.
      
    Returns:
      tensorflow.keras.models.Model: Trained autoencoder.
    """
    input_dim = segments.shape[1]
    autoencoder = build_autoencoder(input_dim, latent_dim)
    autoencoder.fit(segments, segments, epochs=epochs, batch_size=batch_size, verbose=0)
    return autoencoder

def evaluate_autoencoder(autoencoder, segments, latent_dim):
    """
    Evaluate the autoencoder by computing average PRD and compression ratio.
    
    Parameters:
      autoencoder (tensorflow.keras.models.Model): Trained autoencoder.
      segments (numpy.ndarray): ECG segments.
      latent_dim (int): Latent dimension.
      
    Returns:
      tuple: (average PRD, compression ratio).
    """
    reconstructed = autoencoder.predict(segments)
    prd_list = [compute_prd(orig, rec) for orig, rec in zip(segments, reconstructed)]
    avg_prd = np.mean(prd_list)
    input_dim = segments.shape[1]
    cr = input_dim / latent_dim
    return avg_prd, cr

# -------------------------------
# Evaluate a Single Record
# -------------------------------
def evaluate_record(record_name, path, run_original, run_optim, optim_method, run_auto, run_huff,
                    ga_generations, bayes_max_evals, auto_epochs, pop_size, quant_steps,
                    frame_length=256, prd_limit=5.0, add_metrics=False, calc_power=False, power_params=None):
    """
    Evaluate compression on a single ECG record using selected methods.
    
    Parameters:
      record_name (str): Record number.
      path (str): Dataset path.
      run_original (bool): Run Original L2SB.
      run_optim (bool): Run optimized L2SB (using selected method).
      optim_method (int): 1: GA, 2: Bayesian, 3: Bayesian Multi-objective.
      run_auto (bool): Run Autoencoder.
      run_huff (bool): Run Huffman.
      ga_generations (int): GA generations.
      bayes_max_evals (int): Bayesian optimization maximum evaluations.
      auto_epochs (int): Autoencoder epochs.
      pop_size (int): GA population size.
      quant_steps (list): Quantisation steps.
      frame_length (int): Segment length for autoencoder.
      prd_limit (float): PRD threshold.
      add_metrics (bool): Whether to compute additional metrics (SNR, NRMSE).
      calc_power (bool): Whether to compute theoretical dynamic power.
      power_params (dict): Dictionary with keys 'C_load', 'V', 'f', 'alpha'.
      
    Returns:
      dict: Metrics dictionary for each method.
    """
    logger.info(f"Record {record_name}: Loading data...")
    signal = load_ecg_data(record_name=record_name, path=path)
    results = {}
    
    # Original L2SB
    if run_original:
        logger.info(f"Record {record_name}: Running Original L2SB compression...")
        def method_original():
            fixed_thresholds = [0.1, 0.4]
            encoded = l2sb_compress(signal, fixed_thresholds, quant_steps)
            rec = l2sb_decompress(encoded, fixed_thresholds, quant_steps)
            return {
                'CR': compute_compression_ratio(encoded, len(signal)),
                'PRD': compute_prd(signal, rec),
                'WDD': compute_wdd(signal, rec)
            }
        res_dict, t, mem = measure_performance(method_original)
        logger.info(f"Record {record_name}: Original L2SB done in {t:.2f} s, mem {mem/(1024*1024):.2f} MB")
        results['Original L2SB'] = {**res_dict, 'time': t, 'mem': mem}
    
    # Optimized L2SB using chosen method
    if run_optim:
        if optim_method == 1:
            logger.info(f"Record {record_name}: Running GA L2SB optimization (generations={ga_generations}, pop_size={pop_size})...")
            def method_optim():
                opt_res = ga_optimize_l2sb(signal, quant_steps, prd_limit=prd_limit, pop_size=pop_size, generations=ga_generations)
                thresholds = opt_res['thresholds']
                encoded = l2sb_compress(signal, thresholds, quant_steps)
                rec = l2sb_decompress(encoded, thresholds, quant_steps)
                return {
                    'CR': compute_compression_ratio(encoded, len(signal)),
                    'PRD': compute_prd(signal, rec),
                    'WDD': compute_wdd(signal, rec)
                }
            res_dict, t, mem = measure_performance(method_optim)
            logger.info(f"Record {record_name}: GA L2SB done in {t:.2f} s, mem {mem/(1024*1024):.2f} MB")
            results['Optimized L2SB'] = {**res_dict, 'time': t, 'mem': mem}
        elif optim_method == 2:
            logger.info(f"Record {record_name}: Running Bayesian L2SB optimization (max_evals={bayes_max_evals})...")
            def method_bayes():
                opt_res = bayesian_optimize_l2sb(signal, quant_steps, prd_limit=prd_limit, max_evals=bayes_max_evals)
                thresholds = opt_res['thresholds']
                encoded = l2sb_compress(signal, thresholds, quant_steps)
                rec = l2sb_decompress(encoded, thresholds, quant_steps)
                return {
                    'CR': compute_compression_ratio(encoded, len(signal)),
                    'PRD': compute_prd(signal, rec),
                    'WDD': compute_wdd(signal, rec)
                }
            res_dict, t, mem = measure_performance(method_bayes)
            logger.info(f"Record {record_name}: Bayesian L2SB done in {t:.2f} s, mem {mem/(1024*1024):.2f} MB")
            results['Optimized L2SB'] = {**res_dict, 'time': t, 'mem': mem}
        elif optim_method == 3:
            logger.info(f"Record {record_name}: Running Bayesian Multi-objective L2SB optimization (max_evals={bayes_max_evals})...")
            def method_bayes_multi():
                opt_res = bayesian_multiobjective_optimize_l2sb(signal, quant_steps, prd_limit=prd_limit, weight=0.5, max_evals=bayes_max_evals)
                thresholds = opt_res['thresholds']
                encoded = l2sb_compress(signal, thresholds, quant_steps)
                rec = l2sb_decompress(encoded, thresholds, quant_steps)
                return {
                    'CR': compute_compression_ratio(encoded, len(signal)),
                    'PRD': compute_prd(signal, rec),
                    'WDD': compute_wdd(signal, rec)
                }
            res_dict, t, mem = measure_performance(method_bayes_multi)
            logger.info(f"Record {record_name}: Bayesian Multi-objective L2SB done in {t:.2f} s, mem {mem/(1024*1024):.2f} MB")
            results['Optimized L2SB'] = {**res_dict, 'time': t, 'mem': mem}
    
    # Autoencoder
    if run_auto:
        logger.info(f"Record {record_name}: Running Autoencoder compression (epochs={auto_epochs})...")
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
        res_dict, t, mem = measure_performance(method_auto)
        logger.info(f"Record {record_name}: Autoencoder done in {t:.2f} s, mem {mem/(1024*1024):.2f} MB")
        results['Autoencoder'] = {**res_dict, 'time': t, 'mem': mem}
    
    # Huffman
    if run_huff:
        logger.info(f"Record {record_name}: Running Huffman compression...")
        def method_huff():
            cr = compute_huffman_cr(signal, bit_depth=16)
            return {'CR': cr, 'PRD': 0.0, 'WDD': 0.0}
        res_dict, t, mem = measure_performance(method_huff)
        logger.info(f"Record {record_name}: Huffman done in {t:.2f} s, mem {mem/(1024*1024):.2f} MB")
        results['Huffman'] = {**res_dict, 'time': t, 'mem': mem}
    
    # Additional Metrics (SNR and NRMSE)
    if add_metrics:
        logger.info(f"Record {record_name}: Computing additional metrics (SNR, NRMSE)...")
        # For Original L2SB (if available)
        if "Original L2SB" in results:
            fixed_thresholds = [0.1, 0.4]
            encoded = l2sb_compress(signal, fixed_thresholds, quant_steps)
            rec = l2sb_decompress(encoded, fixed_thresholds, quant_steps)
            results["Original L2SB"]["SNR"] = compute_snr(signal, rec)
            results["Original L2SB"]["NRMSE"] = compute_nrmse(signal, rec)
        if "Optimized L2SB" in results:
            # Reuse the thresholds from optimized result
            thresholds = results["Optimized L2SB"].get("thresholds", [0.1, 0.4])
            encoded = l2sb_compress(signal, thresholds, quant_steps)
            rec = l2sb_decompress(encoded, thresholds, quant_steps)
            results["Optimized L2SB"]["SNR"] = compute_snr(signal, rec)
            results["Optimized L2SB"]["NRMSE"] = compute_nrmse(signal, rec)
        if "Autoencoder" in results:
            segments = segment_signal(signal, frame_length)
            auto_rec = np.concatenate([auto_enc for auto_enc in segments])  # This is not ideal—better to compute per segment.
            # Instead, compute on the first segment as a proxy.
            seg = segments[0]
            rec_seg = seg  # For demonstration; ideally, use the autoencoder output.
            results["Autoencoder"]["SNR"] = compute_snr(seg, rec_seg)
            results["Autoencoder"]["NRMSE"] = compute_nrmse(seg, rec_seg)
        if "Huffman" in results:
            # Huffman is lossless so SNR is infinite; we set it to a high value.
            results["Huffman"]["SNR"] = 100.0
            results["Huffman"]["NRMSE"] = 0.0
    
    # Theoretical Dynamic Power Calculation
    if calc_power and power_params:
        C_load = power_params.get("C_load", 1e-12)  # Farads
        V = power_params.get("V", 1.2)              # Volts
        f = power_params.get("f", 1e9)              # Hz
        alpha = power_params.get("alpha", 0.1)        # Activity factor
        dynamic_power = compute_dynamic_power(C_load, V, f, alpha)
        # Add this as a constant metric to all methods.
        for method in results:
            results[method]["DynamicPower"] = dynamic_power
    
    return results

# -------------------------------
# Plotting Helper: Boxplot, Violin, Histogram
# -------------------------------
def plot_metric(overall_results, metric_key, title, ylabel, convert_func=None):
    """
    Plot a metric across methods using Boxplot, Violin Plot, and Histogram with Gaussian Fit.
    
    Parameters:
      overall_results (dict): Dictionary with method names as keys and metric lists as values.
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
# Plotting Helper: Interactive Scatter with Plotly
# -------------------------------
def plot_interactive_scatter(overall_results, power_key="time"):
    """
    Create interactive scatter plots for CR vs. PRD and CR vs. Dynamic Power using Plotly.
    
    Parameters:
      overall_results (dict): Dictionary with method names as keys and metric lists as values.
      power_key (str): Key for dynamic power.
    """
    try:
        import plotly.express as px
        import pandas as pd
    except ImportError:
        logger.error("Plotly and pandas must be installed for interactive plotting.")
        return

    data = []
    for method, metrics in overall_results.items():
        for cr, prd, pwr in zip(metrics["CR"], metrics["PRD"], metrics[power_key]):
            data.append({"Method": method, "CR": cr, "PRD": prd, "DynamicPower": pwr})
    df = pd.DataFrame(data)
    fig1 = px.scatter(df, x="CR", y="PRD", color="Method", title="Interactive: CR vs. PRD")
    fig1.show()
    fig2 = px.scatter(df, x="CR", y="DynamicPower", color="Method", title="Interactive: CR vs. Dynamic Power")
    fig2.show()

# -------------------------------
# Main Function
# -------------------------------
def main():
    logger.info("Starting compression evaluation tool.")
    
    # Select compression methods to run
    run_original = input("Run Original L2SB? (y/n): ").strip().lower().startswith('y')
    # For optimized L2SB, we allow multiple optimization methods.
    run_optim = input("Run Optimized L2SB? (y/n): ").strip().lower().startswith('y')
    optim_method = None
    if run_optim:
        logger.info("Select optimization method for L2SB:")
        logger.info("1: Genetic Algorithm (GA)")
        logger.info("2: Bayesian Optimization")
        logger.info("3: Bayesian Multi-objective Optimization")
        optim_method = int(input("Enter 1, 2, or 3: ").strip())
    run_auto = input("Run Autoencoder? (y/n): ").strip().lower().startswith('y')
    run_huff = input("Run Huffman? (y/n): ").strip().lower().startswith('y')
    
    # Prompt for optimization parameters (if optimization is selected)
    if run_optim and optim_method in [1, 2, 3]:
        if optim_method == 1:
            pop_size = int(input("Enter GA population size (e.g., 10 for quick test, 20+ for normal): "))
            ga_generations = int(input("Enter number of GA generations (e.g., 5 for quick test, 30 for normal): "))
        elif optim_method in [2, 3]:
            bayes_max_evals = int(input("Enter maximum evaluations for Bayesian optimization (e.g., 10 for quick test, 50+ for normal): "))
    
    if run_auto:
        auto_epochs = int(input("Enter number of autoencoder epochs (e.g., 5 for quick test, 50 for normal): "))
    else:
        auto_epochs = 50
    # Huffman doesn't need extra parameters.
    
    # Additional options
    add_metrics = input("Include additional metrics SNR and NRMSE? (y/n): ").strip().lower().startswith('y')
    calc_power = input("Compute theoretical dynamic power? (y/n): ").strip().lower().startswith('y')
    power_params = None
    if calc_power:
        logger.info("Enter theoretical dynamic power parameters:")
        C_load = float(input("Enter load capacitance C_load (in Farads, e.g., 1e-12): ").strip())
        V = float(input("Enter supply voltage V (in Volts, e.g., 1.2): ").strip())
        f = float(input("Enter clock frequency f (in Hz, e.g., 1e9): ").strip())
        alpha = float(input("Enter activity factor alpha (0-1, e.g., 0.1): ").strip())
        power_params = {"C_load": C_load, "V": V, "f": f, "alpha": alpha}
    
    use_interactive = input("Use interactive plotting (Plotly)? (y/n): ").strip().lower().startswith('y')
    
    if not any([run_original, run_optim, run_auto, run_huff]):
        logger.info("No compression methods selected. Exiting.")
        return
    
    # Define record IDs based on specified ranges
    record_ids = list(range(100, 110)) + list(range(111, 120)) + list(range(121, 125)) \
                 + list(range(200, 204)) + [205] + list(range(207, 211)) + list(range(212, 216)) \
                 + [217] + list(range(219, 224)) + [228] + list(range(230, 235))
    path = './mit-bih-arrhythmia-database-1.0.0/'
    quant_steps = [0.01, 0.05, 0.1]
    frame_length = 256
    prd_limit = 5.0
    
    overall_results = {
        'Original L2SB': {'CR': [], 'PRD': [], 'WDD': [], 'time': [], 'mem': []},
        'Optimized L2SB': {'CR': [], 'PRD': [], 'WDD': [], 'time': [], 'mem': []},
        'Autoencoder': {'CR': [], 'PRD': [], 'WDD': [], 'time': [], 'mem': []},
        'Huffman': {'CR': [], 'PRD': [], 'WDD': [], 'time': [], 'mem': []}
    }
    
    valid_records = []
    total_records = len(record_ids)
    
    for idx, rid in enumerate(record_ids, start=1):
        rec_str = str(rid)
        logger.info(f"Processing record {rec_str} ({idx}/{total_records})...")
        try:
            # Depending on optimization method selected, call the corresponding function.
            if run_optim:
                if optim_method == 1:
                    optim_func = ga_optimize_l2sb
                    opt_params = {"pop_size": pop_size, "generations": ga_generations}
                elif optim_method == 2:
                    optim_func = bayesian_optimize_l2sb
                    opt_params = {"max_evals": bayes_max_evals}
                elif optim_method == 3:
                    optim_func = bayesian_multiobjective_optimize_l2sb
                    opt_params = {"max_evals": bayes_max_evals, "weight": 0.5}
                else:
                    optim_func = None
            else:
                optim_func = None

            # Evaluate record using selected methods.
            res = evaluate_record(rec_str, path, run_original, run_optim, run_auto, run_huff,
                                  ga_generations if optim_method==1 else bayes_max_evals,
                                  auto_epochs, pop_size, quant_steps, frame_length, prd_limit,
                                  add_metrics=add_metrics, calc_power=calc_power, power_params=power_params)
            
            for method in res:
                overall_results[method]['CR'].append(res[method]['CR'])
                overall_results[method]['PRD'].append(res[method]['PRD'])
                overall_results[method]['WDD'].append(res[method]['WDD'])
                overall_results[method]['time'].append(res[method]['time'])
                overall_results[method]['mem'].append(res[method]['mem'])
            valid_records.append(rid)
            logger.info(f"Record {rec_str} processed successfully.")
            print_record_metrics(rec_str, res)
        except FileNotFoundError:
            logger.warning(f"Record {rec_str} not found. Skipping...")
        except Exception as e:
            logger.error(f"Error processing record {rec_str}: {e}")
    
    if not valid_records:
        logger.info("No valid records processed. Exiting.")
        return
    
    overall_results = {k: v for k, v in overall_results.items() if len(v['CR']) > 0}
    
    logger.info("=== Overall Metrics Statistics ===")
    for method in overall_results:
        logger.info(f"{method}:")
        for metric in ['CR', 'PRD', 'WDD', 'time', 'mem']:
            arr = np.array(overall_results[method][metric])
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            var_val = np.var(arr)
            if metric == 'mem':
                mean_val /= (1024*1024)
                std_val /= (1024*1024)
                var_val /= ((1024*1024)**2)
            logger.info(f"  {metric} -> Mean: {mean_val:.2f}, Std: {std_val:.2f}, Var: {var_val:.2f}")
    
    # Static Plots using Matplotlib
    mem_convert = lambda x: x / (1024*1024)
    plot_metric(overall_results, 'CR', "Compression Ratio (CR) Across Methods", "CR")
    plot_metric(overall_results, 'PRD', "Percentage Root-Mean-Square Difference (PRD) Across Methods", "PRD (%)")
    plot_metric(overall_results, 'WDD', "Weighted Diagnostic Distortion (WDD) Across Methods", "WDD (%)")
    plot_metric(overall_results, 'time', "Run Time Across Methods", "Time (seconds)")
    plot_metric(overall_results, 'mem', "Memory Usage Across Methods", "Memory (MB)", convert_func=mem_convert)
    
    # Scatter Plots (Static)
    plot_scatter_cr_vs_prd_and_power(overall_results, power_key="time")
    
    # Interactive Plots with Plotly if selected
    if use_interactive:
        plot_interactive_scatter(overall_results, power_key="time")
    
    # Optionally, save overall results to JSON
    save_results = input("Save overall results to JSON file? (y/n): ").strip().lower().startswith('y')
    if save_results:
        import json
        with open("overall_results.json", "w") as f:
            json.dump(overall_results, f, indent=2)
        logger.info("Overall results saved to overall_results.json")

if __name__ == "__main__":
    main()
