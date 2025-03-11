from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
from os.path import join, split

current_dirpath = split(os.path.abspath(__file__))[0]

class ColorBar():
    def __init__(self):
        pass

    def bars(self,color_bar="blueblackred"):
        if color_bar=="blueblackred":
            colors = [
                (0, "blue"),  # Blue for lowest values
                (0.5, "black"),  # Black for middle values
                (1, "red")  # Red for highest values
            ]
        elif color_bar=="redblackblue":
            colors = [
                (0, "red"),  # Blue for lowest values
                (0.5, "black"),  # Black for middle values
                (1, "blue")  # Red for highest values
            ]
        elif color_bar=="mask_alpha":
            colors = [
                (0, "white"),  # 
                (1, (0,0,1,0.4))  # Red for highest values
            ]
        elif color_bar=="bluewhitered":
            colors = [
                (0, "blue"),  # Blue for lowest values
                (0.5, "white"),  # Black for middle values
                (1, "red")  # Red for highest values
            ]
        elif color_bar=="redwhiteblue":
            colors = [
                (0, "red"),  # Blue for lowest values
                (0.5, "white"),  # Black for middle values
                (1, "blue")  # Red for highest values
            ]
        elif color_bar=="wg":
            colors = [
                (0, "white"),  # white for lowest values
                (1, "mediumseagreen")  # green for highest values
            ]
        elif color_bar=="bbo":
            colors = [
                (0, "dodgerblue"),  # Blue for lowest values
                (0.25, "darkblue"),
                (0.5, "black"),  # Black for middle values
                (0.75, "darkred"),  # Black for middle values
                (1, "orangered")  # Red for highest values
            ]
        elif color_bar=="bo":
            colors = [
                (0, "black"),  # Black for lowest values
                (0.5, "darkred"),  #
                (1, "orangered")  # Red for highest values
            ]
        elif color_bar=="obb":
            colors = [
                (0, "orangered"),  # Blue for lowest values
                (0.25, "darkred"),
                (0.5, "black"),  # Black for middle values
                (0.75, "darkblue"),  # Black for middle values
                (1, "dodgerblue")  # Red for highest values
            ]
        elif color_bar=="bwo":
            colors = [
                (0, "darkblue"),  # Blue for lowest values
                (0.25, "dodgerblue"),
                (0.5, "white"),  # Black for middle values
                (0.75, "orangered"),  # Black for middle values
                (1, "darkred")  # Red for highest values
            ]
        elif color_bar=="bwg":
            colors = [
                (0, "royalblue"),  # Blue for lowest values
                (0.5, "white"),  # Black for middle values
                (1, "mediumseagreen")  # Red for highest values
            ]
        elif color_bar=="bwTo":
            colors = [
                (0, "darkblue"),  # Blue for lowest values
                (0.25, "dodgerblue"),
                (0.5, (1,1,1,0.2)),  # Transparent for middle values
                (0.75, "orangered"),  # Black for middle values
                (1, "darkred")  # Red for highest values
            ]
        elif color_bar=="owb":
            colors = [
                (0, "darkred"),  # Blue for lowest values
                (0.25, "orangered"),
                (0.5, "white"),  # Black for middle values
                (0.75, "dodgerblue"),  # Black for middle values
                (1, "darkblue")  # Red for highest values
            ]
        elif color_bar=="spectrum_fsl":
            step = 0.125
            colors = [
                (0, (0, 0, 0, 0)),  # Black with full opacity for the lowest values
                (0.1*step, (255, 0, 255, 0.8)),  # Violet with alpha=0.23
                (1*step, (238/255, 130/255, 238/255, 0.8)),  # Violet with alpha=0.23
                (2*step, (0, 0, 1, 0.8)),  # Blue with alpha=0.23
                (3*step, (0, 1, 1, 0.8)),  # Cyan with alpha=0.23
                (4*step, (64/255, 224/255, 208/255, 0.8)),  # Turquoise with alpha=0.23
                (5*step, (0, 1, 0, 0.8)),  # Green with alpha=0.23
                (6*step, (1, 1, 0, 0.8)),  # Yellow with alpha=0.23
                (7*step, (1, 165/255, 0, 0.8)),  # Orange with alpha=0.23
                (8*step, (1, 0, 0, 0.8))  # Red with alpha=0.23 for the highest values
            ]
        elif color_bar == "random9":
            # Define 9 well-spaced colors
            colors = [
                (1, (255, 0, 0, 1)),       # Red
                (2, (255, 165, 0, 1)),    # Orange
                (3, (0, 255, 0, 1)),      # Green
                (4, (0, 255, 255, 1)),    # Cyan
                (5, (0, 0, 255, 1)),      # Blue
                (6, (75, 0, 130, 1)),     # Indigo
                (7, (238, 130, 238, 1)),  # Violet
                (8, (128, 0, 128, 1)),    # Purple
                (9, (128, 128, 128, 1))   # Gray
            ]
        else: 
            return color_bar


        return LinearSegmentedColormap.from_list("custom_cmap", colors)

    def load_fsl_cmap(self,map="spectrum_iso"):
        # Load the Nx3 RGB values (assuming they are between 0 and 255)
        rgb_values = np.loadtxt(join(current_dirpath,"cmaps",f"{map}.cmap"))
        
        # Normalize RGB values to be between 0 and 1
        rgb_values = rgb_values

        # Create a colormap from the normalized RGB values
        spectrum_cmap = LinearSegmentedColormap.from_list('spectrum_fsl', rgb_values)

        return spectrum_cmap