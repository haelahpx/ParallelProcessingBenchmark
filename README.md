# Python Concurrency Benchmark: Sequential vs. Threading vs. Multiprocessing

This project provides a practical benchmark comparing the performance of **Sequential**, **Threading**, and **Multiprocessing** execution in Python for a CPU-bound task. The script processes chunks of a dataset of varying sizes, measures the execution time for each method, and visualizes the results.

***

## 🚀 Features

-   **System Specification Display**: Automatically detects and prints your CPU and RAM specifications.
-   **Three Processing Methods**: Implements and compares:
    1.  **Sequential**: A single-threaded, straightforward approach.
    2.  **Threading**: Uses the `threading` module to run tasks concurrently.
    3.  **Multiprocessing**: Uses the `multiprocessing` module to run tasks in parallel across multiple CPU cores.
-   **Variable Workload**: Benchmarks are run on increasing percentages of the input dataset (25%, 50%, 75%, 100%) to show how performance scales with data size.
-   **Result Export**: Saves the benchmark timing results to a `processing_times_comparison.csv` file.
-   **Data Visualization**: Automatically generates and saves a line plot (`processing_times_plot.png`) comparing the performance of the three methods.

***

## 📋 Requirements

-   Python 3.x
-   A dataset file named `train.csv`. The script is configured to read a column named `trip_duration`. You can adapt this to any numerical column in your CSV. A common source for this dataset is the [NYC Taxi Trip Duration Kaggle competition](https://www.kaggle.com/c/nyc-taxi-trip-duration/data).

***

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone ParallelPro
    cd ParallelProcessingBenchmark
    ```

2.  **Install the required Python libraries:**
    ```bash
    pip -r install requirements.txt
    ```

3.  **Add your dataset:**
    Place your `train.csv` file in the root directory of the project.

***

## ▶️ How to Run

Execute the main script from your terminal. The script will run the benchmarks for all data sizes and methods, save the results, and display the final plot.

```bash
python project.py
