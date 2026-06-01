import matplotlib 
# matplotlib.use('Agg')  # Switch to the Agg backend

import matplotlib.pyplot as plt
import numpy as np
from os.path import join ,split
from .colorbar import ColorBar
from matplotlib.colors import LogNorm

colorbar = ColorBar()

class SimMatrixPlot:
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot_simmatrix(correlation_matrix, ax=None, parcel_ids_positions=None, 
                       colorbar_label="Correlation Strength", parcel_labels=None, show_parcels="VH", titles=None, 
                       result_path=None, colormap="plasma", scale_factor=1, dpi=100, show_colorbar=True, 
                       colorbar_orientation="vertical", vmin=None, vmax=None, suppress_ticks=True, 
                       zscale="linear",zlabel_size=12):
        """
        Plots a similarity matrix on the provided axes. If no axes are provided, creates a new figure and axes.

        Parameters:
        - ax: Matplotlib axes object to plot on. If None, creates a new figure.
        - parcel_ids_positions: Dictionary with parcel ids as keys and tuple (min_val, max_val) as values.
        - show_parcels: String "H" to show horizontal labels, "V" for vertical, "VH" for both.
        - titles: Title of the plot.
        - result_path: Path to save the plot.
        - colormap: Color map for the plot.
        - colorbar_label: Z axis label.
        - scale_factor: Factor to scale the plot size.
        - dpi: Dots per inch for the plot resolution.
        - show_colorbar: Boolean to show the colorbar.
        - colorbar_orientation: "vertical" or "horizontal" to control colorbar placement.
        - vmin: Minimum value for colormap scaling.
        - vmax: Maximum value for colormap scaling.
        - suppress_ticks: Boolean to suppress axis ticks.
        - zscale: z-scaling ["log" or "linear"].
        """
        # Ensure the colormap is properly processed (assuming colorbar.bars is a custom function)
        colormap = colorbar.bars(colormap)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12 * scale_factor, 10 * scale_factor), dpi=dpi)
        else:
            fig = ax.figure

        # Determine normalization based on zscale.
        if zscale == "log":
            # Replace zeros with a very small value to avoid log(0) issues.
            epsilon = 1e-6  # Adjust epsilon based on the scale of your data
            correlation_matrix = np.where(correlation_matrix == 0, epsilon, correlation_matrix)
            effective_vmin = vmin if vmin is not None else correlation_matrix.min()
            effective_vmax = vmax if vmax is not None else correlation_matrix.max()
            norm = LogNorm(vmin=effective_vmin, vmax=effective_vmax)
        elif zscale == "linear":
            norm = None

        # Plot the matrix, now with the option to pass in a normalization object.
        cax = ax.matshow(correlation_matrix, interpolation='nearest', cmap=colormap, norm=norm, vmin=vmin, vmax=vmax)
        ax.grid(False)

        # Handling parcel labels or positions
        if not suppress_ticks:
            if parcel_ids_positions:
                middle_positions = [(min_val + max_val) / 2 for min_val, max_val in parcel_ids_positions.values()]
                min_positions = [min_val for min_val, _ in parcel_ids_positions.values()]
                max_positions = [max_val for _, max_val in parcel_ids_positions.values()]
                labels = list(parcel_ids_positions.keys())

                combined_positions = min_positions + middle_positions + max_positions
                combined_labels = [""] * len(min_positions) + labels + [""] * len(max_positions)

                sorted_indices = np.argsort(combined_positions)
                sorted_positions = np.array(combined_positions)[sorted_indices]
                sorted_labels = np.array(combined_labels)[sorted_indices]

                if show_parcels in ["H", "VH"]:
                    ax.set_xticks(sorted_positions)
                    ax.set_xticklabels(sorted_labels, rotation=45, fontsize=14 * scale_factor, fontweight='bold')

                if show_parcels in ["V", "VH"]:
                    ax.set_yticks(sorted_positions)
                    ax.set_yticklabels(sorted_labels, fontsize=14 * scale_factor, fontweight='bold')
            elif parcel_labels is not None:
                positions = np.arange(0, len(parcel_labels))
                if show_parcels in ["H", "VH"]:
                    ax.set_xticks(positions)
                    ax.set_xticklabels(parcel_labels, rotation=45, fontsize=14 * scale_factor, fontweight='bold') 
                if show_parcels in ["V", "VH"]:
                    ax.set_yticks(positions)
                    ax.set_yticklabels(parcel_labels, fontsize=14 * scale_factor, fontweight='bold')    
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if titles:
            ax.set_title(titles, fontsize=16, fontweight='bold')

        # Show colorbar if requested
        if show_colorbar:
            if colorbar_orientation == 'horizontal':
                cbar = fig.colorbar(cax, ax=ax, fraction=0.15 * scale_factor, pad=0.04,
                                    label=colorbar_label, orientation='horizontal')
            else:
                cbar = fig.colorbar(cax, ax=ax, fraction=0.15 * scale_factor, pad=0.04,
                                    label=colorbar_label, orientation='vertical')
            cbar.set_label(
                colorbar_label,
                fontsize=zlabel_size,     
                fontweight='bold'   
            )
            # Adjust tick label fontsize on the colorbar
            cbar.ax.tick_params(labelsize=zlabel_size-2)  # Change 14 to desired fontsize

        if result_path:
            fig.savefig(f"{result_path}.pdf", dpi=dpi)

        return fig, ax  # Always return fig and ax for further manipulation
    
    def plot_multiple_simmatrices(self,simmatrices, titles=None,result_path=None,plotshow=False, cmap='plasma',dpi=100):
        """
        Plots multiple similarity matrices on the same figure.

        Parameters:
            simmatrices (list of np.ndarray): List of similarity matrices to plot.
            titles (list of str, optional): List of titles for each subplot.
            cmap (str, optional): Colormap to use for the plots.
        """
        num_matrices = len(simmatrices)
        # Ensure titles are provided or create default titles
        if titles is None:
            titles = [f'Matrix {i + 1}' for i in range(num_matrices)]
        if len(titles) != num_matrices:
            raise ValueError("The number of titles must match the number of matrices.")
        # Determine grid size (e.g., 2x2 for 4 matrices)
        grid_size = int(np.ceil(np.sqrt(num_matrices)))
        # Create subplots
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        # Flatten the array of axes to easily iterate over it
        axs = axs.flatten()
        for idx, (matrix, title) in enumerate(zip(simmatrices, titles)):
            im = axs[idx].imshow(simmatrices[idx], cmap=cmap)
            axs[idx].set_title(title)
            axs[idx].axis('off')
            fig.colorbar(im, ax=axs[idx], fraction=0.046, pad=0.04)
        # Remove extra subplots if any
        for extra_ax in axs[num_matrices:]:
            extra_ax.axis('off')
        plt.tight_layout()
        if result_path:
            fig.savefig(f"{result_path}.pdf", dpi=dpi)  # Ensure dpi is set for saving as well
        if plotshow:
            plt.show()
        plt.close(fig)




 
