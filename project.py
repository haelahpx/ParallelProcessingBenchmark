import pandas as pd
import time
import threading
import multiprocessing as mp
from multiprocessing import Pool
import platform
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

# --- Sequential Processing ---
def sequential_process(data, threshold):
    sorted_data = sorted(data)
    filtered_data = [x for x in sorted_data if x > threshold]
    return filtered_data

# --- Threading Processing ---
def thread_worker(data, result_list, index, threshold):
    sorted_data = sorted(data)
    filtered_data = [x for x in sorted_data if x > threshold]
    result_list[index] = filtered_data

def threaded_process(data, threshold):
    chunk_size = len(data) // 2
    result_list = [None, None]

    t1 = threading.Thread(target=thread_worker, args=(data[:chunk_size], result_list, 0, threshold))
    t2 = threading.Thread(target=thread_worker, args=(data[chunk_size:], result_list, 1, threshold))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    return result_list[0] + result_list[1]

# --- Multiprocessing Processing ---
def mp_worker(args):
    data, threshold = args
    sorted_data = sorted(data)
    return [x for x in sorted_data if x > threshold]

def multiprocess_process(data, threshold):
    chunk_size = len(data) // 2
    chunks = [(data[:chunk_size], threshold), (data[chunk_size:], threshold)]
    with Pool(processes=2) as pool:
        results = pool.map(mp_worker, chunks)
    return results[0] + results[1]

# --- Benchmark Function ---
def benchmark(method_name, method_func, data, threshold):
    start_time = time.time()
    method_func(data, threshold)
    elapsed_time = time.time() - start_time
    return elapsed_time

# --- Main Execution ---
if __name__ == "__main__":
    # --- System Specs (Print Only Once) ---
    print("🔧 System Specifications:")
    print("CPU:", platform.processor())
    print("Physical Cores:", psutil.cpu_count(logical=False))
    print("Logical Threads:", psutil.cpu_count(logical=True))
    print("RAM (GB):", round(psutil.virtual_memory().total / 1e9, 2))
    print()

    # --- Load Dataset ---
    df = pd.read_csv('train.csv')
    trip_duration_data = df['trip_duration']
    FILTER_THRESHOLD = 1000
    split_percentages = [0.25, 0.50, 0.75, 1.00]

    results = {
        "Method": [],
        "Data Size (%)": [],
        "Data Rows": [],
        "Time (s)": []
    }

    for pct in split_percentages:
        size = int(len(trip_duration_data) * pct)
        data_chunk = trip_duration_data[:size].copy().tolist()
        pct_label = f"{int(pct * 100)}%"

        print(f"\n📊 Processing {pct_label} of data ({size} rows)")

        # Sequential
        print("→ Running Sequential Processing...")
        t_seq = benchmark("Sequential", sequential_process, data_chunk, FILTER_THRESHOLD)
        results["Method"].append("Sequential")
        results["Data Size (%)"].append(pct_label)
        results["Data Rows"].append(size)
        results["Time (s)"].append(round(t_seq, 4))
        print(f"✔ Sequential done in {t_seq:.4f} seconds")

        # Threading
        print("→ Running Threading Processing...")
        t_thread = benchmark("Threading", threaded_process, data_chunk, FILTER_THRESHOLD)
        results["Method"].append("Threading")
        results["Data Size (%)"].append(pct_label)
        results["Data Rows"].append(size)
        results["Time (s)"].append(round(t_thread, 4))
        print(f"✔ Threading done in {t_thread:.4f} seconds")

        # Multiprocessing
        print("→ Running Multiprocessing...")
        t_mp = benchmark("Multiprocessing", multiprocess_process, data_chunk, FILTER_THRESHOLD)
        results["Method"].append("Multiprocessing")
        results["Data Size (%)"].append(pct_label)
        results["Data Rows"].append(size)
        results["Time (s)"].append(round(t_mp, 4))
        print(f"✔ Multiprocessing done in {t_mp:.4f} seconds")

    # --- Save CSV ---
    results_df = pd.DataFrame(results)
    results_df.to_csv('processing_times_comparison.csv', index=False)
    print("\n✅ Benchmark complete! Results saved to 'processing_times_comparison.csv'.")

    # --- Plot ---
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x="Data Rows", y="Time (s)", hue="Method", marker="o")
    plt.title("Processing Time Comparison by Method and Data Size")
    plt.xlabel("Data Size (rows)")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig("processing_times_plot.png")
    plt.show()
