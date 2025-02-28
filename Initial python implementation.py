import os
import numpy as np
import wfdb
import math
import heapq
from collections import Counter, namedtuple
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

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
# 2. L2SB Compression / Decompression Functions
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
# 3. Metrics Functions: PRD and Compression Ratio
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
        bits_subband = 2  # for subband index (3 sub-bands)
        bits_q = 1 if q_value == 0 else math.ceil(math.log2(q_value + 1))
        total_bits += bits_subband + bits_q
    original_bits = num_samples * bit_depth
    cr = original_bits / total_bits if total_bits > 0 else 0
    return cr

# -------------------------------
# 4. Baseline Huffman Compression (Lossless)
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
# 5. Genetic Algorithm for Adaptive L2SB Optimization
# -------------------------------
def evaluate_individual(ind, signal, quant_steps, prd_limit):
    """
    Evaluate a candidate individual [t1, t2] for L2SB thresholds.
    
    If PRD exceeds prd_limit, returns 0; otherwise returns the compression ratio.
    
    Parameters:
      ind (list): Candidate thresholds [t1, t2].
      signal (numpy.ndarray): Original ECG signal.
      quant_steps (list): Quantisation step sizes for each sub-band.
      prd_limit (float): Maximum acceptable PRD.
      
    Returns:
      float: Fitness value (compression ratio) or 0 if PRD exceeds limit.
    """
    t1, t2 = ind
    if t2 <= t1:
        return 0.0
    thresholds = [t1, t2]
    encoded = l2sb_compress(signal, thresholds, quant_steps)
    rec_signal = l2sb_decompress(encoded, thresholds, quant_steps)
    prd = compute_prd(signal, rec_signal)
    if prd > prd_limit:
        return 0.0
    cr = compute_compression_ratio(encoded, len(signal))
    return cr

def ga_optimize_l2sb(signal, quant_steps, prd_limit=5.0, pop_size=20, generations=30):
    """
    Use a genetic algorithm to optimize the L2SB thresholds for maximum compression ratio
    while maintaining PRD within the acceptable limit.
    
    Parameters:
      signal (numpy.ndarray): Original ECG signal.
      quant_steps (list): Quantisation steps for each sub-band.
      prd_limit (float): Maximum acceptable PRD.
      pop_size (int): Population size for GA.
      generations (int): Number of generations.
      
    Returns:
      dict: Dictionary with best thresholds, CR, and PRD.
    """
    t1_min, t1_max = 0.05, 0.3
    t2_min, t2_max = 0.3, 1.0

    population = []
    for _ in range(pop_size):
        t1 = random.uniform(t1_min, t1_max)
        t2 = random.uniform(max(t2_min, t1 + 0.01), t2_max)
        population.append([t1, t2])
    
    best_ind = None
    best_fit = -1

    for gen in range(generations):
        fitnesses = [evaluate_individual(ind, signal, quant_steps, prd_limit) for ind in population]
        for ind, fit in zip(population, fitnesses):
            if fit > best_fit:
                best_fit = fit
                best_ind = ind.copy()
        new_population = []
        for _ in range(pop_size):
            i1, i2 = random.sample(range(pop_size), 2)
            if fitnesses[i1] > fitnesses[i2]:
                new_population.append(population[i1].copy())
            else:
                new_population.append(population[i2].copy())
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
        for i in range(pop_size):
            if random.random() < 0.3:
                new_population[i][0] += random.uniform(-0.01, 0.01)
                new_population[i][1] += random.uniform(-0.01, 0.01)
                new_population[i][0] = min(max(new_population[i][0], t1_min), t1_max)
                new_population[i][1] = min(max(new_population[i][1], t2_min), t2_max)
                if new_population[i][1] <= new_population[i][0]:
                    new_population[i][1] = new_population[i][0] + 0.01
        population = new_population

    best_thresholds = [best_ind[0], best_ind[1]]
    best_encoded = l2sb_compress(signal, best_thresholds, quant_steps)
    best_rec = l2sb_decompress(best_encoded, best_thresholds, quant_steps)
    best_prd = compute_prd(signal, best_rec)
    best_cr = compute_compression_ratio(best_encoded, len(signal))
    return {"thresholds": best_thresholds, "CR": best_cr, "PRD": best_prd}

# -------------------------------
# 6. Deep Learning Autoencoder for ECG Compression
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
    latent = Dense(latent_dim, activation='relu', name='latent')(encoded)
    decoded = Dense(64, activation='relu')(latent)
    decoded = Dense(128, activation='relu')(decoded)
    out = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(inputs=inp, outputs=out)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(segments, latent_dim, epochs=50, batch_size=32):
    """
    Train the autoencoder on segmented ECG data.
    
    Parameters:
      segments (numpy.ndarray): 2D array of ECG signal segments.
      latent_dim (int): Dimension of the latent representation.
      epochs (int): Number of training epochs.
      batch_size (int): Training batch size.
      
    Returns:
      tensorflow.keras.models.Model: Trained autoencoder model.
    """
    input_dim = segments.shape[1]
    autoencoder = build_autoencoder(input_dim, latent_dim)
    autoencoder.fit(segments, segments, epochs=epochs, batch_size=batch_size, verbose=1)
    return autoencoder

def evaluate_autoencoder(autoencoder, segments):
    """
    Evaluate the autoencoder by computing the average PRD and estimating the compression ratio.
    
    Parameters:
      autoencoder (tensorflow.keras.models.Model): Trained autoencoder model.
      segments (numpy.ndarray): Array of ECG signal segments.
      
    Returns:
      tuple: (average PRD, compression ratio) for the autoencoder.
    """
    reconstructed = autoencoder.predict(segments)
    prd_list = [compute_prd(orig, rec) for orig, rec in zip(segments, reconstructed)]
    avg_prd = np.mean(prd_list)
    input_dim = segments.shape[1]
    latent_dim = autoencoder.get_layer('latent').output_shape[-1]
    cr = input_dim / latent_dim
    return avg_prd, cr

# -------------------------------
# 7. Main Routine: Comparison and Graphical Analysis
# -------------------------------
def main():
    # Load ECG data
    try:
        signal = load_ecg_data(record_name='100', path='./mit-bih-arrhythmia-database-1.0.0/')
    except Exception as e:
        print("Error loading ECG data:", e)
        return
    print(f"Loaded ECG data with {len(signal)} samples.")
    
    # Set fixed quantisation steps for L2SB
    quant_steps = [0.01, 0.05, 0.1]
    
    # A. Original (Fixed) L2SB Compression
    fixed_thresholds = [0.1, 0.4]  # Example fixed thresholds
    encoded_fixed = l2sb_compress(signal, fixed_thresholds, quant_steps)
    rec_fixed = l2sb_decompress(encoded_fixed, fixed_thresholds, quant_steps)
    prd_fixed = compute_prd(signal, rec_fixed)
    cr_fixed = compute_compression_ratio(encoded_fixed, len(signal))
    print("Original L2SB (fixed):")
    print(f"  Thresholds: {fixed_thresholds}, CR: {cr_fixed:.2f}, PRD: {prd_fixed:.2f}%")
    
    # B. GA-Optimized L2SB Compression
    ga_results = ga_optimize_l2sb(signal, quant_steps, prd_limit=5.0, pop_size=20, generations=30)
    print("GA-Optimized L2SB:")
    print(f"  Thresholds: {ga_results['thresholds']}, CR: {ga_results['CR']:.2f}, PRD: {ga_results['PRD']:.2f}%")
    
    # C. Deep Learning Autoencoder Compression
    frame_length = 256  # Number of samples per segment
    segments = segment_signal(signal, frame_length)
    latent_dim = 32  # Target latent dimension (approximate CR = 256/32 = 8)
    autoencoder = train_autoencoder(segments, latent_dim, epochs=50, batch_size=32)
    prd_auto, cr_auto = evaluate_autoencoder(autoencoder, segments)
    print("Deep Learning Autoencoder Compression:")
    print(f"  Approximate Compression Ratio: {cr_auto:.2f}, PRD: {prd_auto:.2f}%")
    
    # D. Huffman Compression (Baseline Lossless)
    huff_cr = compute_huffman_cr(signal, bit_depth=16)
    print("Huffman Compression (simulated):")
    print(f"  Compression Ratio: {huff_cr:.2f}, PRD: 0.00%")
    
    # E. Graphical Comparison of Methods
    methods = ['Original L2SB', 'GA L2SB', 'Autoencoder', 'Huffman']
    cr_values = [cr_fixed, ga_results['CR'], cr_auto, huff_cr]
    prd_values = [prd_fixed, ga_results['PRD'], prd_auto, 0.0]
    
    x = np.arange(len(methods))
    width = 0.35  # Width of the bars

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_cr = 'tab:blue'
    ax1.set_xlabel('Compression Method')
    ax1.set_ylabel('Compression Ratio (CR)', color=color_cr)
    bars1 = ax1.bar(x - width/2, cr_values, width, label='CR', color=color_cr)
    ax1.tick_params(axis='y', labelcolor=color_cr)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)

    ax2 = ax1.twinx()  # Second y-axis for PRD
    color_prd = 'tab:red'
    ax2.set_ylabel('PRD (%)', color=color_prd)
    bars2 = ax2.bar(x + width/2, prd_values, width, label='PRD', color=color_prd)
    ax2.tick_params(axis='y', labelcolor=color_prd)

    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', color=color_cr)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', color=color_prd)

    fig.suptitle('Comparison of ECG Compression Methods')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
