import glob
import os

import numpy as np
import pandas as pd

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

        combined_timestamps, combined_behavior = _concatenate_datasets(dpath=path, 
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

        timestamps = pd.read_csv(os.path.join(dpath, timeStamps_file))
        behavior = pd.read_csv(os.path.join(dpath, behavior_file))

        return timestamps, behavior

# Part of step two
def _concatenate_datasets(dpath, timestamps_hungry_file, behavior_hungry_file, 
                                    timestamps_satiated_file, behavior_satiated_file):
    """
    Helper function that concatenates two datasets of timestamps and behavior, adjusting the timestamps
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

    # Convert behavior time to ms
    behavior_hungry['Time'] *= 1000
    behavior_satiated['Time'] *= 1000

    # Calculate the gap and adjust the satiated behavior timestamps
    last_time_hungry = behavior_hungry['Time'].iloc[-1]
    first_time_satiated = behavior_satiated['Time'].iloc[0]
    time_gap = first_time_satiated - last_time_hungry

    # Adjusting the satiated behavior dataset
    behavior_satiated['Time'] -= time_gap

    # Correct the satiated timestamps
    time_offset = behavior_satiated['Time'].iloc[0] * 1000
    timestamps_satiated['Time Stamp (ms)'] += time_offset

    # Concatenate datasets
    combined_timestamps = pd.concat([timestamps_hungry, timestamps_satiated], ignore_index=True)
    combined_behavior = pd.concat([behavior_hungry, behavior_satiated], ignore_index=True)

    return combined_timestamps, combined_behavior

# Step 3 Spikes
def process_spikes_and_calcium(minian_ds):
    """
    Processes spikes and calcium data from the given minian dataset.

    Args:
        minian_ds (xarray.core.dataset.Dataset): The minian dataset.

    Returns:
        tuple: A tuple containing the processed spike and calcium data.
    
    """
    # Process Spikes
    tracenew_spike, labelsnew_spike = _process_helper(minian_ds, "S")

    # Process Calcium
    tracenew_calcium, labelsnew_calcium = _process_helper(minian_ds, "C")

    return tracenew_spike, labelsnew_spike, tracenew_calcium, labelsnew_calcium

# Part of step 3
def _process_helper(minian_ds, label_str):
    """
    Helper function for processing spikes or calcium data from the given minian dataset.

    Args:
        minian_ds (xarray.core.dataset.Dataset): The minian dataset.
        label_str (str): The label string.

    Returns:
        tuple: A tuple containing the processed spike or calcium data.
    
    """
    # frameend = subset1.get('frame').stop - subset1.get('frame').start+1
    trace = minian_ds[label_str].values.T
    # trace = trace[:frameend] # I don't know if this is still needed

    neuron_ori = len(trace[0])
    labels = minian_ds[label_str].unit_labels.values
    id = minian_ds[label_str].unit_id.values
    tracenew = trace
    (unique, counts) = np.unique(labels, return_counts = True)
    repeated = np.where(counts > 1)
    repeatedvalue = unique[repeated]
    repeatedind=np.array([],dtype = int)
    repeatedind1st=np.array([],dtype = int)
    for i in range(0,len(repeatedvalue)):
        t = np.where(labels==repeatedvalue[i])[0]
        f = t[0]
        repeatedind = np.append(repeatedind, t)
        repeatedind1st = np.append(repeatedind1st, f)
        tracenew[:,f] = np.max(trace[:,t],axis = 1)

    todelete = np.union1d(np.setdiff1d(repeatedind,repeatedind1st), np.where(labels == -1))
    tracenew = np.delete(tracenew,todelete,1)
    labelsnew = np.delete(labels, todelete)
    neuron_del = len(todelete)
    neuron_now = len(tracenew[0])

    print ('Deleted ' + str(neuron_del) + ' neurons out of the original ' + str(neuron_ori) + ' neurons, ' + str(neuron_now) + ' neurons remain')

    return tracenew, labelsnew

# Step 4 align and interpolate
def align_and_interpolate(dpath, ca_timestamps, behavior_data, tracenew, labelsnew):
    """
    Aligns and interpolates behavioral data with calcium timestamps.

    Args:
        dpath (str): The directory path where the CSV files are located.
        ca_timestamp (pandas.core.frame.DataFrame): The calcium timestamps.
        behavior_data (pandas.core.frame.DataFrame): The behavioral data.
        tracenew (numpy.ndarray): The calcium trace data.
        labelsnew (numpy.ndarray): The labels for the calcium trace data.

    Returns:
        tuple: A tuple containing the aligned trace data and labels.

    """
    # Load calcium timestamps from a CSV file
    CA = ca_timestamps

    # Load behavioral data from another CSV file
    Behavior = behavior_data

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

# Step 5 save
def save_trace_and_labels(tracealigned, labelsaligned, output_path_calcium, animal_id):
    """
    Saves the aligned trace and labels to a pickle file.

    Args:
        tracealigned (numpy.ndarray): The aligned trace data.
        labelsaligned (numpy.ndarray): The labels for the aligned trace data.
        output_path_calcium (str): The output path where the pickle file will be saved.
        animal_id (str): The ID of the animal for which the pickle file will be saved.
    """
    # Save the aligned trace and labels to a pickle file
    df = pd.DataFrame(data=tracealigned, index=labelsaligned)
    df.to_pickle(os.path.join(output_path_calcium, str(animal_id + ".pkl")))


