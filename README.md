# Graph Plotting Utility for Model Comparison

This Python script creates comparison plots for machine learning model performance across different datasets.

## Features

-   ✅ **Single or Multiple Subplots**: Create single plots or multiple subplots for different base models
-   ✅ **Unified Plot Format**: Grouped bars for all datasets in each subplot
-   ✅ **Multiple Base Model Support**: Compare LLaVA-1.5, LLaVA-NeXT, and other base models
-   ✅ **Close Bar Grouping**: Groups bars closely together without spacing between datasets
-   ✅ **Baseline Visualization**: Displays baseline model as horizontal dotted lines across each group
-   ✅ **Y-axis Zooming**: Zooms in Y-axis to emphasize differences between models
-   ✅ **Professional Typography**: Uses Times New Roman font for all text elements
-   ✅ **Flexible Data Input**: Supports both dictionary and CSV data input
-   ✅ **Fully Configurable**: Reusable with different data and customizable parameters

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Method 1: Using Dictionary Data

```python
from graph import GraphPlotter

# Define your data
data = {
    'datasets': ['MSCOCO', 'A-OKVQA', 'GQA'],
    'models': {
        'LLaVA-1.5': [75.2, 68.1, 82.3],  # Baseline scores
        'LLaVA-1.5 + Truth_LLM': [76.8, 69.7, 83.9],  # Your method
        'LLaVA-1.5 + Random Gate': [75.9, 68.8, 82.8]  # Comparison method
    }
}

# Create plotter and generate graphs
plotter = GraphPlotter(data)
plotter.plot_comparison_graphs()

# For multiple subplots with different base models
plotter.plot_multi_base_comparison(
    base_models=['LLaVA-1.5', 'LLaVA-NeXT'],
    method_suffixes=['TruthProbe_LLM', 'Random Gate']
)
```

### Method 2: Using CSV Data

1. Create a CSV file with the following format:

```csv
Dataset,LLaVA-1.5,LLaVA-1.5 + Truth_LLM,LLaVA-1.5 + Random Gate
MSCOCO,75.2,76.8,75.9
A-OKVQA,68.1,69.7,68.8
GQA,82.3,83.9,82.8
```

2. Load and plot:

```python
from graph import GraphPlotter, load_data_from_csv

data = load_data_from_csv('your_data.csv')
plotter = GraphPlotter(data)
plotter.plot_comparison_graphs()
```

### Quick Start

Run the example script:

```bash
python graph.py
```

## Customization Options

-   **Colors**: Change bar colors by passing `colors=['#color1', '#color2']`
-   **Figure Size**: Adjust with `figsize=(width, height)`
-   **Models to Compare**: Specify with `compare_models=['model1', 'model2']`
-   **Save Plot**: Set `save_path='filename.png'` to save the figure
-   **Y-axis Margin**: Modify the `margin_percent` in `calculate_y_limits()`

## Example Output

## Example Output

The script generates a unified figure with:

-   Grouped bars for all datasets in a single plot
-   Three groups (one per dataset) with bars placed closely together
-   Blue bars for Truth_LLM model, Purple bars for Random Gate model
-   Black dotted lines showing baseline performance across each group
-   Zoomed Y-axis to highlight performance differences
-   Value annotations on bars and baseline scores
-   Times New Roman font for all text elements

## Requirements

-   Python 3.7+
-   matplotlib
-   numpy
-   pandas

## Files

-   `graph.py`: Main plotting script
-   `requirements.txt`: Python dependencies
-   `sample_data.csv`: Example data format
-   `README.md`: This documentation
