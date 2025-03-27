
import numpy as np
import os
from graphplot.simmatrix import SimMatrixPlot
from tools.datautils import DataUtils
from connectomics.network import NetBasedAnalysis
import matplotlib.pyplot as plt
from os.path import split, join
from nilearn import datasets, plotting, surface
import nibabel as nib
from matplotlib.colors import LinearSegmentedColormap
from graphplot.colorbar import ColorBar
import argparse

###################


colorbar    = ColorBar()
simplt      = SimMatrixPlot()
dutils      = DataUtils()
nba         = NetBasedAnalysis()




spectrum_colors = colorbar.load_fsl_cmap()
spectrum_colors_trans = colorbar.load_fsl_cmap("spectrum_iso_transparent")



# Load the MNI152 template for volume views
mni_template = datasets.load_mni152_template()

# Load the msi_stat map (NIfTI file in MNI space)
def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process some input parameters.")

    # Add arguments
    parser.add_argument('--atlas', type=str, default='LFMIHIFIF-3', choices=['LFMIHIFIF-2', 'LFMIHIFIF-3', 'LFMIHIFIF-4', 
                                                                     'LFMIIIFIF-2', 'LFMIIIFIF-3', 'LFMIIIFIF-4', 
                                                                     'geometric_cubeK18mm','geometric_cubeK23mm',
                                                                     'aal', 'destrieux','mist-197','schaefer-200'], 
                        help='Atlas choice (must be one of: LFMIHIFIF-2, LFMIHIFIF-3, LFMIHIFIF-4, geometric, aal, destrieux)')
    parser.add_argument('--group', type=str, default='Geneva-Study', help='Group name (default: "Geneva-Study")')
    parser.add_argument('--show_plot', type=int, default=1, help='Render and shoe 3D plots (default: 1)')

    args        = parser.parse_args()
    parc_scheme = args.atlas 
    group       = args.group
    show_plot   = bool(args.show_plot)
      
    OUTDIRPATH  = join(dutils.ANARESULTSPATH,"MSI",group)
    os.makedirs(OUTDIRPATH,exist_ok=True)
    msi_path   = join(dutils.BIDSDATAPATH,group,"derivatives","group","metabolic-similarity-map",
                        f"metabolic-similarity-map_space-mni_atlas-{parc_scheme}_proj-pca-tsne_mrsi.nii.gz")
    
    msi_stat   = nib.load(msi_path)

    # Load the fsaverage template for surface views (left and right hemispheres)
    fsaverage = datasets.fetch_surf_fsaverage()

    RADIUS = 0.1
    # Project the msi_stat map (3D volume in MNI space) onto the fsaverage surface
    texture_left  = surface.vol_to_surf(msi_stat, fsaverage.pial_left,radius=RADIUS)
    texture_right = surface.vol_to_surf(msi_stat, fsaverage.pial_right,radius=RADIUS)
    texture_left  = texture_left.clip(0, 256)
    texture_right = texture_right.clip(0, 256)

    texture_left  = 2*(texture_left-256/2)
    texture_right = 2*(texture_right-256/2)

    # Create the mesh for left and right hemisphere surfaces
    mesh_left = surface.load_surf_mesh(fsaverage.pial_left)
    mesh_right = surface.load_surf_mesh(fsaverage.pial_right)

    # Plot using plot_surf_stat_map with the 'plotly' engine for interactive visualization
    fig_left = plotting.plot_surf_stat_map(mesh_left, texture_left, hemi='left',
                                        view='medial', colorbar=True,cmap=spectrum_colors,
                                        threshold=0.001, bg_map=fsaverage.sulc_left,
                                        engine='plotly', title=f"Left Hemisphere r={RADIUS}",
                                        vmin=0,vmax=255)
    # fig_left.show()

    fig_right = plotting.plot_surf_stat_map(mesh_right, texture_right, hemi='right',
                                            view='lateral', colorbar=True,cmap=spectrum_colors,
                                            threshold=0.001, bg_map=fsaverage.sulc_right,
                                            engine='plotly', title=f"Right Hemisphere r={RADIUS}",
                                            vmin=0,vmax=255)
    # fig_right.show()

    # Plot using plot_surf_stat_map with the 'matplotlib' engine for interactive visualization
    fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage6')
    RADIUS = 0.5
    texture_left = surface.vol_to_surf(msi_stat, fsaverage.pial_left,radius=RADIUS,interpolation='nearest')
    texture_right = surface.vol_to_surf(msi_stat, fsaverage.pial_right,radius=RADIUS,interpolation='nearest')
    texture_left = texture_left.clip(0, 256)
    texture_right = texture_right.clip(0, 256)
    # Create the mesh for left and right hemisphere surfaces
    mesh_left = surface.load_surf_mesh(fsaverage.pial_left)
    mesh_right = surface.load_surf_mesh(fsaverage.pial_right)
    fig_left = plotting.plot_surf_stat_map(mesh_left, texture_left, hemi='left',
                                        view='medial', colorbar=False,cmap=spectrum_colors_trans, 
                                        bg_map=fsaverage.sulc_left,vmin=0,vmax=255,threshold=1,avg_method="max")
    plt.savefig(join(OUTDIRPATH,"msi_surface_LFMIHIFIF-3_medial_left.png"), format='png', transparent=True, bbox_inches='tight')

    fig_left = plotting.plot_surf_stat_map(mesh_left, texture_left, hemi='left',
                                        view='lateral', colorbar=False,cmap=spectrum_colors_trans, 
                                        bg_map=fsaverage.sulc_left,vmin=0,vmax=255,threshold=1,avg_method="max")
    plt.savefig(join(OUTDIRPATH,"msi_surface_LFMIHIFIF-3_lateral_left.png"), format='png', transparent=True, bbox_inches='tight')

    fig_right = plotting.plot_surf_stat_map(mesh_right, texture_right, hemi='right',
                                            view='lateral', colorbar=False,cmap=spectrum_colors_trans, 
                                            bg_map=fsaverage.sulc_right,vmin=0,vmax=255,threshold=1,avg_method="max")
    plt.savefig(join(OUTDIRPATH,"msi_surface_LFMIHIFIF-3_lateral_right.png"), format='png', transparent=True, bbox_inches='tight')

    fig_right = plotting.plot_surf_stat_map(mesh_right, texture_right, hemi='right',
                                            view='medial', colorbar=False,cmap=spectrum_colors_trans, 
                                            bg_map=fsaverage.sulc_right,vmin=0,vmax=255,threshold=1,avg_method="max")
    plt.savefig(join(OUTDIRPATH,"msi_surface_LFMIHIFIF-3_medial_right.png"), format='png', transparent=True, bbox_inches='tight')



    # Plot the first sagittal view at x = -5
    display = plotting.plot_stat_map(msi_stat, bg_img=mni_template, threshold=0.001,
                        cut_coords=[-3],black_bg=False,
                        display_mode='x', colorbar=False, cmap=spectrum_colors,
                        annotate=False, draw_cross=False)
    plt.savefig(join(OUTDIRPATH,"msi_LFMIHIFIF-3_sagital_right.png"), format='png', transparent=True)

    # Plot the second sagittal view at x = 5
    plotting.plot_stat_map(msi_stat, bg_img=mni_template, threshold=0.001,cut_coords=[3],
                        display_mode='x', colorbar=False, cmap=spectrum_colors,black_bg=False,
                        annotate=False, draw_cross=False)
    plt.savefig(join(OUTDIRPATH,"msi_LFMIHIFIF-3_sagital_left.png"), format='png',transparent=True)

    # Plot the axial view at z = 0
    plotting.plot_stat_map(msi_stat, bg_img=mni_template, threshold=0.001,cut_coords=[0],
                        display_mode='z', colorbar=False, cmap=spectrum_colors,black_bg=False,
                        annotate=False, draw_cross=False)
    plt.savefig(join(OUTDIRPATH,"msi_LFMIHIFIF-3_axial.png"), format='png',transparent=True)

    # Plot the coronal (longitudinal) view at y = -13
    plotting.plot_stat_map(msi_stat, bg_img=mni_template, threshold=0.001,cut_coords=[-13],
                        display_mode='y', colorbar=False, cmap=spectrum_colors,black_bg=False,
                        annotate=False, draw_cross=False)
    plt.savefig(join(OUTDIRPATH,"msi_LFMIHIFIF-3_coronal_right.png"), format='png', transparent=True)

    # Show the plots
    if show_plot:
        plotting.show()

if __name__ == "__main__":
    main()








