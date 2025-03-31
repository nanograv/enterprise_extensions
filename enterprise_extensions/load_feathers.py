"""Contains functions to load Pulsar data from folders containing feather files."""

import os
import glob
from enterprise.pulsar import Pulsar


def load_feathers_from_folder(folder, time_span_cut_yr=None, pulsar_name_list=None, exclude_pattern=None):
    """
    Load Pulsar data from a folder containing feather files.
    :param folder: Path to the folder containing feather files.
    :param time_span_cut_yr: Optional time span cut to filter pulsars [units of years].
    :param pulsar_name_list: Optional list of pulsar names to include (all others excluded).
    :param exclude_pattern: Optional string pattern - files containing this pattern will be excluded.
    :return: List of FeatherPulsar objects loaded from the feather files.
    """
    # Check if the folder exists
    if not os.path.isdir(folder):
        raise ValueError(f"Folder {folder} does not exist.")
    # Check if the folder is empty
    if not os.listdir(folder):
        raise ValueError(f"Folder {folder} is empty.")
    # Make list of feather files
    feather_files = glob.glob(os.path.join(folder, '*.feather'))
    # make list of pulsars to be returned
    pulsars = []
    # Loop through each feather file and create a Pulsar object
    for feather_file in feather_files:
        # Get the filename for filtering
        feather_name = os.path.basename(feather_file)
        pulsar_name = feather_name.split('_')[0]
        
        # Check if the feather file is in pulsar_name_list if provided
        if pulsar_name_list is not None and pulsar_name not in pulsar_name_list:
            continue
            
        # Check if the file should be excluded based on pattern
        if exclude_pattern is not None and exclude_pattern in feather_name:
            continue
            
        # Load the feather file into a Pulsar object
        psr = Pulsar(feather_file)
        
        # Check if the time span cut is provided and filter the pulsars
        if time_span_cut_yr is not None:
            years_observed = (psr.toas.max() - psr.toas.min()) / (525600 * 60)
            if years_observed < time_span_cut_yr:
                print(f"Skipping {psr.name} because it has been observed for {years_observed:.2f} years (< {time_span_cut_yr} years).")
                continue
                
        # Append the Pulsar object to the list
        pulsars.append(psr)

    # Check for duplicate names
    if len(pulsars) != len(set([p.name for p in pulsars])):
        print("Warning: Duplicate pulsar names found.")
    
    return pulsars


    return pulsars
