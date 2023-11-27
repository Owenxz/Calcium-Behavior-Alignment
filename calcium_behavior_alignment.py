import glob
import os

import numpy as np
import pandas as pd


def main():
    # Add code for the main function here
    combined_timestamps, combined_behavior = load_adjust_concatenate_datasets(
        'your_data_path', 'timestamps_hungry.csv', 'behavior_hungry.csv', 
        'timestamps_satiated.csv', 'behavior_satiated.csv')
    
    tracealigned, labelsaligned = align_and_interpolate(dpath, combined_timestamps, combined_behavior, tracenew, labelsnew)
    pass

if __name__ == "__main__":
    main()

# Step 1 parse
def parse_scope_times(exp_path, id_path):
    """
    Parses the scope times from the given experiment path and animal ID path.

    Args:
        exp_path (str): The path to the experiment directory.
        id_path (str): The path to the animal ID directory.

    Returns:
        dict: A dictionary containing the parsed scope times with keys as [animal_id]_[date]_[time].
    """
    files = os.listdir(id_path)

    animal_ids = [file for file in files if os.path.isdir(os.path.join(id_path, file)) and len(file) == 3 and file[0].isdigit()]

    scope_times = {}  # Dictionary to store CSV data with keys as [animal_id]_[date]_[time]

    for animal_id in animal_ids:
        animal_path = os.path.join(exp_path, animal_id)
        entries_count = 0  # Counter for the number of entries for this animal_id
        scope_times[animal_id] = {}  # Initialize an empty dictionary for this animal_id

        # Date pattern: YYYY_MM_DD
        date_dirs = glob.glob(os.path.join(animal_path, '*/[12][09][0-9][0-9]_[01][0-9]_[0-3][0-9]'))
        for date_dir in date_dirs:
            # Extract date for naming
            date = os.path.basename(date_dir)

            # Time pattern: HH_MM_SS
            time_dirs = glob.glob(os.path.join(date_dir, '[0-2][0-9]_[0-5][0-9]_[0-5][0-9]'))
            for time_dir in time_dirs:
                # Extract time for naming
                time = os.path.basename(time_dir)
                final_path = os.path.join(time_dir, 'miniscopeDeviceName')

                # Construct the variable name
                var_name = f"{animal_id}_{date}_{time}"

                # Read the CSV file
                csv_file_path = os.path.join(final_path, 'timeStamps.csv')
                if os.path.exists(csv_file_path):
                    scope_times[animal_id][var_name] = pd.read_csv(csv_file_path)

                entries_count += 1

                if entries_count > 2:
                    print(f"Alert: More than two entries found for animal_id {animal_id}")
                    break  # Stop processing more entries for this animal_id

            if entries_count > 2:
                break  # Break out of the outer loop as well

    return scope_times

# Step two process animal data based on ID
def process_animal_data(dpath, scope_times, animal_id):
    """
    Concatenates animal data based on the provided animal ID. If the animal ID is found in the scope_times dictionary
    and has exactly 2 entries, it combines hungry and satiated behavior datasets. Otherwise, it returns the timeStamps.csv
    and behavior.csv files for the given animal ID.

    Args:
        dpath (str): The path to the directory containing the animal data.
        scope_times (dict): A dictionary containing scope times for different animal IDs.
        animal_id (str): The ID of the animal for which data needs to be concatenated.

    Returns:
        tuple or str: If the animal ID is found in scope_times and has 2 entries, it returns a tuple containing the
        combined timestamps and behavior datasets. Otherwise, it returns the paths to the timeStamps.csv and behavior.csv
        files.
    """
    if animal_id in scope_times and len(scope_times[animal_id]) == 2:
        path = os.path.join(dpath, animal_id)

        # Assuming specific file naming convention
        timestamps_hungry_file = dpath + 'timeStamps_hunger.csv'
        behavior_hungry_file = dpath + f'{animal_id}_Satiation_recording_behavior_hungry.csv'
        timestamps_satiated_file = dpath + 'timeStamps_satiated.csv'
        behavior_satiated_file = dpath + f'{animal_id}_Satiation_recording_behavior_satiated.csv'

        combined_timestamps, combined_behavior = concatenate_datasets(dpath=path, 
                                                                      timestamps_hungry_file=timestamps_hungry_file, 
                                                                      behavior_hungry_file=behavior_hungry_file,
                                                                      timestamps_satiated_file=timestamps_satiated_file,
                                                                      behavior_satiated_file=behavior_satiated_file)
        
        return combined_timestamps, combined_behavior
    
    else:
        # Just return the timeStamps.csv and behavior.csv files
        path = os.path.join(dpath, animal_id)

        # Assuming specific file naming convention
        timeStamps_file = dpath + 'timeStamps.csv'
        behavior_file = dpath + f'{animal_id}_Satiation_recording_behavior.csv'
        return timeStamps_file, behavior_file


def concatenate_datasets(dpath, timestamps_hungry_file, behavior_hungry_file, 
                                    timestamps_satiated_file, behavior_satiated_file):
    """
    Concatenates two datasets of timestamps and behavior, adjusting the timestamps
    of the second dataset to align with the first dataset. This function also removes
    any time gaps between the the two behavior datasets.

    Args:
        dpath (str): The path to the directory containing the dataset files.
        timestamps_hungry_file (str): The filename of the timestamps file for the hungry dataset.
        behavior_hungry_file (str): The filename of the behavior file for the hungry dataset.
        timestamps_satiated_file (str): The filename of the timestamps file for the satiated dataset.
        behavior_satiated_file (str): The filename of the behavior file for the satiated dataset.

    Returns:
        tuple: A tuple containing the concatenated timestamps and behavior datasets.
    """

    # Load datasets
    timestamps_hungry = pd.read_csv(os.path.join(dpath, timestamps_hungry_file))
    behavior_hungry = pd.read_csv(os.path.join(dpath, behavior_hungry_file))

    timestamps_satiated = pd.read_csv(os.path.join(dpath, timestamps_satiated_file))
    behavior_satiated = pd.read_csv(os.path.join(dpath, behavior_satiated_file))

    # Calculate the gap and adjust the satiated behavior timestamps
    last_time_hungry = behavior_hungry['Time'].iloc[-1]
    first_time_satiated = behavior_satiated['Time'].iloc[0]
    time_gap = first_time_satiated - last_time_hungry

    # Adjusting the satiated behavior dataset
    behavior_satiated['Time'] -= time_gap

    # Correct the satiated timestamps
    time_offset = behavior_satiated['Time'].iloc[0] * 1000 # convert to ms
    time_stamps_satiated_corrected['Time Stamp (ms)'] += time_offset

    # Concatenate datasets
    combined_timestamps = pd.concat([timestamps_hungry, time_stamps_satiated_corrected], ignore_index=True)
    combined_behavior = pd.concat([behavior_hungry, behavior_satiated], ignore_index=True)

    return combined_timestamps, combined_behavior

def align_and_interpolate(dpath, timestamps_file_name, behavior_file_name, tracenew, labelsnew):
    """
    Aligns and interpolates behavioral data with calcium timestamps.

    Args:
        dpath (str): The directory path where the CSV files are located.
        timestamps_file_name (str): The file name of the CSV file containing calcium timestamps.
        behavior_file_name (str): The file name of the CSV file containing behavioral data.
        tracenew (numpy.ndarray): The calcium trace data.
        labelsnew (numpy.ndarray): The labels for the calcium trace data.

    Returns:
        tuple: A tuple containing the aligned trace data and labels.

    """
    # Load calcium timestamps from a CSV file
    CA = pd.read_csv(os.path.join(dpath, timestamps_file_name))

    # Load behavioral data from another CSV file
    Behavior = pd.read_csv(os.path.join(dpath, behavior_file_name))

    # Extract time column from behavioral data and adjust it for when miniscope record active is triggered
    behaviortime = Behavior['Time'].values
    behaviortime = behaviortime - behaviortime[np.where(Behavior['Miniscope record active'] > 0)[0][0]]

    # Convert calcium timestamps from milliseconds to seconds
    catime = CA['Time Stamp (ms)'].values / 1000

    # Extract cue and bar columns from the behavioral data
    cue = Behavior['Tone active'].values
    bar = Behavior['Bar Press active'].values

    # Interpolate the behavioral data to align with the calcium timestamps
    cuealigned = np.interp(catime, behaviortime, cue)
    baraligned = np.interp(catime, behaviortime, bar)

    # Check if miniscope data is shorter than timestamp data and pad if necessary ?
    if tracenew.shape[1] < len(catime):
        padding_length = len(catime) - tracenew.shape[1]
        padding = np.zeros((tracenew.shape[0], padding_length))
        tracenew_padded = np.hstack((tracenew, padding))
    else:
        tracenew_padded = tracenew

    # Stack the aligned behavioral data
    tracealigned = np.vstack((tracenew_padded.T, cuealigned, baraligned))

    # Concatenate labels for the aligned data
    labelsaligned = np.hstack((labelsnew, 'cue', 'bar'))

    return tracealigned, labelsaligned

