import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd

FONT_SCALE = 1.75
SAVE_FOLDER = "./outputs/"

class GraphPlotter:
	"""
	A flexible graph plotter for comparing model performances across datasets.
	"""

	def __init__(self, data: Dict, baseline_model: str = "LLaVA-1.5"):
		"""
		Initialize the plotter with data.

		Args:
			data: Dictionary with structure:
				{
					'datasets': ['MSCOCO', 'A-OKVQA', 'GQA'],
					'models': {
						'LLaVA-1.5': [baseline_scores],
						'LLaVA-1.5 + Truth_LLM': [truth_llm_scores],
						'LLaVA-1.5 + Random Gate': [random_gate_scores]
					}
				}
			baseline_model: Name of the baseline model to show as dotted line
		"""
		self.data = data
		self.baseline_model = baseline_model
		self.datasets = data['datasets']
		self.models = data['models']

	def calculate_y_limits(self, values: List[float], margin_percent: float = 0.1) -> Tuple[float, float]:
		"""
		Calculate y-axis limits to zoom in around the actual values.

		Args:
			values: List of values to consider
			margin_percent: Percentage margin to add above and below the range

		Returns:
			Tuple of (y_min, y_max)
		"""
		min_val = min(values)
		max_val = max(values)
		range_val = max_val - min_val

		# Add margin
		margin = range_val * margin_percent
		y_min = max(0, min_val - margin)  # Don't go below 0
		y_max = max_val + margin

		return y_min, y_max

	def plot_comparison_graphs(self,
							 compare_models: Optional[List[str]] = None,
							 figsize: Tuple[int, int] = (12, 6),
							 colors: Optional[List[str]] = None,
							 save_path: Optional[str] = None,
							 base_models: Optional[List[str]] = None,
							 subplot_titles: Optional[List[str]] = None):
		"""
		Create comparison plots with grouped bars for all datasets.
		Can create single plot or multiple subplots with different base models.

		Args:
			compare_models: List of models to compare (excluding baseline)
			figsize: Figure size as (width, height)
			colors: Colors for the comparison models
			save_path: Path to save the figure (optional)
			base_models: List of base models for multiple subplots (if None, uses single plot)
			subplot_titles: List of titles for each subplot (if None, uses base model names)
		"""
		# Set Times New Roman font for all text elements
		plt.rcParams['font.family'] = 'Times New Roman'
		plt.rcParams['font.size'] = 16 * FONT_SCALE

		assert compare_models is not None, "compare_models must be provided"

		if colors is None:
			colors = ['#432E81', '#BBA9CC']  # Blue and Purple

		# Determine if we're creating multiple subplots
		if base_models is None:
			base_models = [self.baseline_model]

		if subplot_titles is None:
			subplot_titles = base_models

		n_subplots = len(base_models)

		# Adjust figure size for multiple subplots
		if n_subplots > 1:
			figsize = (figsize[0] * n_subplots * 0.8, figsize[1])

		# Create figure with appropriate number of subplots
		fig, axes = plt.subplots(1, n_subplots, figsize=figsize)
		if n_subplots == 1:
			axes = [axes]

		# Get all values for global y-axis calculation
		all_values = []
		for base_model in base_models:
			if base_model in self.models:
				all_values.extend(self.models[base_model])
			for model in compare_models:
				if f"{base_model} + {model}" in self.models:
					all_values.extend(self.models[f"{base_model} + {model}"])

		# Calculate global y-axis limits
		global_y_min, global_y_max = self.calculate_y_limits(all_values)

		# Create each subplot
		for subplot_idx, (base_model, subplot_title) in enumerate(zip(base_models, subplot_titles)):
			ax = axes[subplot_idx]

			# Check if base model exists in data
			if base_model not in self.models:
				ax.text(0.5, 0.5, f'Base model "{base_model}" not found in data',
					   ha='center', va='center', transform=ax.transAxes,
					   fontfamily='Times New Roman')
				continue

			self._plot_single_comparison(ax, base_model, compare_models, colors, subplot_title,
									   global_y_min, global_y_max, subplot_idx == 0)

		plt.tight_layout()

		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')

		plt.show()

	def _plot_single_comparison(self, ax, base_model: str, compare_models: List[str],
							  colors: List[str], subplot_title: str,
							  y_min: float, y_max: float, show_legend: bool = True):
		"""
		Plot a single comparison subplot with grouped bars.

		Args:
			ax: Matplotlib axis object
			base_model: Base model for this subplot
			compare_models: List of models to compare
			colors: Colors for the comparison models
			subplot_title: Title for this subplot
			y_min, y_max: Y-axis limits
			show_legend: Whether to show legend (typically only for first subplot)
		"""
		# Number of datasets and models
		n_datasets = len(self.datasets)
		n_models = len(compare_models)

		# Bar width and spacing
		bar_width = 0.15
		group_spacing = 0.3  # Space between groups

		# Calculate positions for grouped bars
		group_positions = np.arange(n_datasets) * (n_models * bar_width + group_spacing)

		# Get baseline scores for this base model
		baseline_scores = [self.models[base_model][i] for i in range(n_datasets)]

		# Plot bars for each model
		bars_list = []
		actual_models = []

		for j, model in enumerate(compare_models):
			# Try to find the full model name by combining base_model + model
			full_model_name = None
			if model in self.models:
				full_model_name = model
			else:
				# Try combining base model with the compare model
				potential_name = f"{base_model} + {model}"
				if potential_name in self.models:
					full_model_name = potential_name

			if full_model_name is None:
				continue

			actual_models.append(full_model_name)
			model_scores = [self.models[full_model_name][i] for i in range(n_datasets)]

			# Calculate x positions for this model's bars
			x_positions = group_positions + len(bars_list) * bar_width

			# Create bars
			clean_label = model if model in self.models else model  # Use the short name for label
			bars = ax.bar(x_positions, model_scores, bar_width,
						 label=clean_label,
						 color=colors[len(bars_list) % len(colors)], alpha=0.8)
			bars_list.append(bars)

		# Add baseline as horizontal lines for each group
		n_actual_models = len(bars_list)
		for i, (baseline_score, group_pos) in enumerate(zip(baseline_scores, group_positions)):
			# Draw baseline line across the group
			line_start = group_pos - bar_width/2
			line_end = group_pos + (n_actual_models - 0.5) * bar_width
			ax.plot([line_start, line_end], [baseline_score, baseline_score],
				   color='black', linestyle='--', linewidth=2, alpha=0.7)

		# Set y-axis limits
		ax.set_ylim(y_min, y_max)

		# Set x-axis labels (dataset names)
		if n_actual_models > 0:
			group_centers = group_positions + (n_actual_models - 1) * bar_width / 2
			ax.set_xticks(group_centers)
			ax.set_xticklabels(self.datasets, fontfamily='Times New Roman', fontsize=28)

		# Customize the plot
		# ax.set_xlabel('Datasets', fontsize=32, fontfamily='Times New Roman')
		if show_legend:  # Only show y-label on leftmost subplot
			ax.set_ylabel('F1 Score', fontsize=16 * FONT_SCALE, fontfamily='Times New Roman')

		# Use provided subplot title
		ax.set_title(f'{subplot_title}', fontsize=18 * FONT_SCALE, fontweight='bold', fontfamily='Times New Roman')		# Set tick label fonts
		for label in ax.get_xticklabels():
			label.set_fontfamily('Times New Roman')
			label.set_fontsize(14 * FONT_SCALE)
		for label in ax.get_yticklabels():
			label.set_fontfamily('Times New Roman')
			label.set_fontsize(14 * FONT_SCALE)

		# Add legend only to the first subplot
		if show_legend and bars_list:
			# Get clean labels for legend
			legend_labels = []
			for model in compare_models:
				if model in self.models or f"{base_model} + {model}" in self.models:
					legend_labels.append(model)
			legend_labels.append("Vanilla")

			legend_handles = [bars_list[i][0] for i in range(len(bars_list))]

			# Create a dummy line for baseline in legend
			baseline_line = mlines.Line2D([0], [0], color='black', linestyle='--', linewidth=2, alpha=0.7)
			legend_handles.append(baseline_line)

			ax.legend(legend_handles, legend_labels, loc='upper left', prop={'family': 'Times New Roman', 'size': 14 * FONT_SCALE})

		# Add grid for better readability
		ax.grid(True, alpha=0.3, axis='y')

	def plot_multi_base_comparison(self,
								  base_models: List[str],
								  method_suffixes: List[str] = ['TruthProbe_LLM', 'Random Gate'],
								  figsize: Tuple[int, int] = (12, 5),
								  colors: Optional[List[str]] = None,
								  subplot_titles: Optional[List[str]] = None,
								  save_path: Optional[str] = None):
		"""
		Convenience method to create multiple subplots comparing different base models.

		Args:
			base_models: List of base model names
			method_suffixes: List of method suffixes to append to base models
			figsize: Figure size as (width, height)
			colors: Colors for the comparison models
			subplot_titles: Custom titles for subplots (if None, uses base model names)
			save_path: Path to save the figure (optional)
		"""
		if subplot_titles is None:
			subplot_titles = [f"{base_model} Performance" for base_model in base_models]

		self.plot_comparison_graphs(
			compare_models=method_suffixes,
			figsize=figsize,
			colors=colors,
			save_path=save_path,
			base_models=base_models,
			subplot_titles=subplot_titles
		)


def load_data_from_dict(data_dict: Dict) -> Dict:
	"""
	Load data from a dictionary format.

	Args:
		data_dict: Dictionary containing the data

	Returns:
		Formatted data for the plotter
	"""
	return data_dict


def load_data_from_csv(csv_path: str) -> Dict:
	"""
	Load data from a CSV file.
	Expected format:
	Dataset,LLaVA-1.5,LLaVA-1.5 + Truth_LLM,LLaVA-1.5 + Random Gate
	MSCOCO,score1,score2,score3
	A-OKVQA,score1,score2,score3
	GQA,score1,score2,score3

	Args:
		csv_path: Path to the CSV file

	Returns:
		Formatted data for the plotter
	"""
	df = pd.read_csv(csv_path)
	datasets = df['Dataset'].tolist()
	models = {}

	for column in df.columns[1:]:  # Skip 'Dataset' column
		models[column] = df[column].tolist()

	return {
		'datasets': datasets,
		'models': models
	}


# Example usage and sample data
if __name__ == "__main__":
	print("Graph Plotter for Model Comparison")
	print("=" * 40)

	# Method 1: Load from dictionary (for manual data entry)
	sample_data = {
		'datasets': ['POPE(MSCOCO)', 'POPE(A-OKVQA)', 'POPE(GQA)'],
		'models': {
			'LLaVA-1.5': [85.84, 86.54, 85.25],  # LLaVA-1.5 Baseline scores
			'LLaVA-1.5 + TruthProbe_LLM': [85.81, 86.41, 85.3],  # Our method with LLaVA-1.5
			'LLaVA-1.5 + Random Gate': [84.89, 85.79, 84.38],  # Random Gate with LLaVA-1.5
			'LLaVA-NeXT': [86.46, 87.36, 86.35],  # LLaVA-NeXT Baseline scores
			'LLaVA-NeXT + TruthProbe_LLM': [87.49, 87.93, 86.69],  # Our method with LLaVA-NeXT
			'LLaVA-NeXT + Random Gate': [85.86, 87.10, 85.55]  # Random Gate with LLaVA-NeXT
		}
	}

	# Method 2: Load from CSV (uncomment to use)
	# data = load_data_from_csv('sample_data.csv')

	# Create plotter instance
	plotter = GraphPlotter(sample_data, baseline_model='LLaVA-1.5')

	# Example 1: Single plot with LLaVA-1.5 as base model (original format)
	print("Generating single plot with LLaVA-1.5...")
	plotter.plot_multi_base_comparison(
		base_models=['LLaVA-1.5'],
		method_suffixes=['TruthProbe_LLM', 'Random Gate'],
		figsize=(12, 8),
		colors=['#432E81', '#BBA9CC'],
		save_path=SAVE_FOLDER + 'llava_1_5_comparison.png'
	)
	plotter.plot_multi_base_comparison(
		base_models=['LLaVA-NeXT'],
		method_suffixes=['TruthProbe_LLM', 'Random Gate'],
		figsize=(12, 8),
		colors=['#432E81', '#BBA9CC'],
		save_path=SAVE_FOLDER + 'llava_NeXT_comparison.png'
	)

	# Example 2: Multiple subplots comparing both base models (using convenience method)
	print("Generating multiple subplots with different base models...")
	plotter.plot_multi_base_comparison(
		base_models=['LLaVA-1.5', 'LLaVA-NeXT'],
		method_suffixes=['TruthProbe_LLM', 'Random Gate'],
		figsize=(12, 8),
		colors=['#432E81', '#BBA9CC'],
		save_path=SAVE_FOLDER + 'multi_base_comparison.png'
	)

	print("Graphs have been generated successfully!")
	print("\nFeatures implemented:")
	print("✓ Single or multiple subplot support")
	print("✓ Unified plot format with grouped bars for all datasets")
	print("✓ Support for multiple base models (LLaVA-1.5, LLaVA-NeXT, etc.)")
	print("✓ Grouped bars for each model (TruthProbe_LLM vs Random Gate)")
	print("✓ Baseline shown as black dotted lines across each group")
	print("✓ Y-axis zoomed in to emphasize differences")
	print("✓ Times New Roman font with double-sized text (32pt base, 36pt titles)")
	print("✓ Configurable and reusable with different data")
	print("\nTo use with your own data:")
	print("1. Replace the sample_data dictionary with your actual values")
	print("2. Or create a CSV file and use load_data_from_csv() function")
	print("3. Adjust colors, figure size, or other parameters as needed")
	print("4. Install requirements: pip install -r requirements.txt")
