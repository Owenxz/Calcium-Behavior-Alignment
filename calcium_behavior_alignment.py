import glob
import multiprocessing as mp
import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minian.utilities import open_minian

def main():
    # Command-line arguments parsing
    arg_parser = argparse.ArgumentParser(description="Calcium-behavior alignment pipeline")

    # Define the arguments
    arg_parser.add_argument("exp_path", type=str, help="The path to the experiment directory (e.g. /scratch/09117/xz6783/minian/Satiation/)")
    arg_parser.add_argument("id_path", type=str, help="The path to the animal ID directory (e.g. /scratch/09117/xz6783/minian/Satiation/Session Combined/)")
    arg_parser.add_argument("beh_path", type=str, help="The path to the behavioral data directory (e.g. /scratch/09117/xz6783/minian/Behavior_Files/)")
    arg_parser.add_argument("experiment", type=str, help="The name of the experiment (e.g. Satiation)")
    arg_parser.add_argument("-n", "--num_processes", type=int, default=2, help="The number of processes to use for multiprocessing (default: 2)")
    arg_parser.add_argument("-v", "--verbose", action="store_true", help="Whether to print verbose output")

    # Parse the arguments
    args = arg_parser.parse_args()

    # Initialize the argument variables
    exp_path = args.exp_path
    id_path = args.id_path
    beh_path = args.beh_path
    experiment = args.experiment
    num_processes = args.num_processes
    verbose = args.verbose

    # Execute the pipeline
    if verbose:
        print("Executing calcium-behavior alignment pipeline")

    execute(exp_path, id_path, beh_path, experiment, num_processes, verbose)

    if verbose:
        print("Finished executing calcium-behavior alignment pipeline")



def execute(exp_path, id_path, beh_path, experiment, num_processes=2, verbose=False):
    """
    Executes the calcium-behavior alignment pipeline.

    Args:
        exp_path (str): The path to the experiment directory.
        id_path (str): The path to the animal ID directory.
        beh_path (str): The path to the behavioral data directory.
        experiment (str): The name of the experiment.
        num_processes (int): The number of processes to use for multiprocessing.
        verbose (bool): Whether to print verbose output.

    Returns:
        bool: True if the pipeline executed successfully, False otherwise.
    """
    # For each path, check if it exists
    if verbose:
        if not os.path.exists(exp_path):
            print(f"Error: Experiment path {exp_path} does not exist")
            return False
        else:
            print(f"Experiment path: {exp_path}")

        if not os.path.exists(id_path):
            print(f"Error: Id path {id_path} does not exist")
            return False
        else:
            print(f"Id path: {id_path}")

        if not os.path.exists(beh_path):
            print(f"Error: Behavior path {beh_path} does not exist")
            return False
        else:
            print(f"Behavior path: {beh_path}")

    # Step 1 parse scope_times and behavior_data
    scope_times = parse_scope_times(exp_path, id_path, verbose)
    animal_ids = list(scope_times.keys())
    behavior_data = parse_behavior_times(beh_path, experiment, animal_ids)

    # Step 2 process animal data based on ID
    # Process and align spikes and calcium via multiprocessing
    if verbose:
        print(f"Processing and aligning {len(animal_ids)} animal IDs with {num_processes} process(es)")

    with mp.Pool(processes=num_processes) as pool:
        results = [pool.apply_async(process_and_align, args=(animal_id, id_path, scope_times[animal_id], behavior_data, verbose))
                   for animal_id in animal_ids]
        
        output = [p.get() for p in results]

    output = np.array(output)

    # Print summary of results
    if verbose:
        print(f"Successfully processed and aligned {len(output[output[:, 0] == 'True'])} animal IDs: {output[output[:, 0] == 'True'][:, 1]}")
        print(f"Failed to process and align {len(output[output[:, 0] == 'False'])} animal IDs: {output[output[:, 0] == 'False'][:, 1]}")

    return output

# Step 1 parse
def parse_scope_times(exp_path, id_path, verbose=False):
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

                # Convert timestamps to seconds
                scope_times[animal_id][var_name]['Time Stamp (ms)'] /= 1000.0

                # Change column name
                scope_times[animal_id][var_name].rename(columns={'Time Stamp (ms)': 'Time Stamp (s)'}, inplace=True)

                entries_count += 1

                if entries_count > 2:
                    print(f"Alert: More than two entries found for animal_id {animal_id}")
                    break  # Stop processing more entries for this animal_id

            if entries_count > 2:
                break  # Break out of the outer loop as well

    # Print summary of scope times
    if verbose:
        print(f"Found {len(scope_times)} animal IDs")
        print(f"Animal IDs: {scope_times.keys()}")

        # Number of animals with two entries
        two_entries_count = len([animal_id for animal_id in scope_times if len(scope_times[animal_id]) == 2])
        print(f"Found {two_entries_count} animal IDs with two entries")
        
        # Number of animals with one entry
        one_entry_count = len([animal_id for animal_id in scope_times if len(scope_times[animal_id]) == 1])
        print(f"Found {one_entry_count} animal IDs with one entry")

        # Erronous animal IDs, i.e. animals with no entries or more than two entries
        erronous_animal_ids = [animal_id for animal_id in scope_times if len(scope_times[animal_id]) not in [1, 2]]
        print(f"Erronous animal IDs: {'None' if len(erronous_animal_ids) == 0 else erronous_animal_ids}")

    return scope_times

# Step two process animal data based on ID
def parse_behavior_times(beh_path, experiment, animal_ids):
    """
    Read the behavioral data csv files from beh_path based on experiment name and animal_id and stores the contents in a
    dictionary keyed by these IDs.

    Args:
        dpath (str): The path to the directory containing the animal data.
        scope_times (dict): A dictionary containing scope times for different animal IDs.
        animal_id (str): The ID of the animal for which data needs to be concatenated.

    Returns:
        tuple or str: If the animal ID is found in scope_times and has 2 entries, it returns a tuple containing the
        combined timestamps and behavior datasets. Otherwise, it returns the paths to the timeStamps.csv and behavior.csv
        files.
    """
    
    # Get a list of all files in the base directory and directly construct the full file paths for the relevant CSV files
    file_paths = [os.path.join(beh_path, file) for file in os.listdir(beh_path)
                      if file.startswith(experiment) and file.endswith(".csv")]

    # Dictionary to hold the CSV contents, keyed by animal IDs
    behavior_data = {}

    for file_path in file_paths:
        # Extract the filename from the path
        filename = os.path.basename(file_path)
        
        # Extract the animal ID from the filename
        parts = filename.split('_')
        if len(parts) > 2:
            if parts[1] in animal_ids: # Animal ID is between the first and second underscore
                
                animal_id = parts[1]
                
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Add the DataFrame to the animal_data dictionary
                if animal_id not in behavior_data:
                    behavior_data[animal_id] = df
                else:
                    print(f"File number error: {animal_id} has more tha one behavior file for {experiment}")
        else:
            print(f"Filename format error: {filename}")

    return behavior_data

# Part of step two 
def combine_datasets(scope_times, behavior_data, animal_id, verbose=False):
    """
    Helper function that reads directly from scope_times and behavior, and concatenates scope_time if there are two entires.

    Args:
        scope_times (dict): A dictionary containing scope times for different animal IDs.
        behavior_data (dict): A dictionary containing behavioral data for different animal IDs.
        animal_id (str): The ID of the animal for which data needs to be concatenated.
        verbose (bool): Whether to print verbose output.

    Returns:
        tuple: A tuple containing the concatenated timestamps and behavior datasets.
    """
    # Retrieve the behavior data
    if animal_id in behavior_data:
        # Retrieve indicies in behavior_data where the recording resets
        time_diff = np.where(np.diff(behavior_data[animal_id]['Miniscope record active']) != 0)[0] + 1

        # Section out time_diff into separate recording resets; i.e. if there is an index gap of at these 50, then there is a reset
        # recording_sections is a list of tuples, where each tuple is the (start, end) of a recording
        recording_sections = []
        section_start = time_diff[0]
        
        breaks = np.where(np.diff(time_diff) >= 50)[0] + 1
        
        if breaks:
            for idx in breaks:
                section_end = idx
                recording_sections.append((section_start, section_end))
                section_start = time_diff[idx + 1]
                
        # Append the last (section_start, section_end)
        section_end = time_diff[-1]
        recording_sections.append((section_start, section_end))
                
        ret_behavior = behavior_data[animal_id]  #minus the beginning of the first recording onset

        # Offset ret_Behaviors by the first recording onset
        ret_behavior['Time (s)'] = ret_behavior['Time (s)'] - ret_behavior['Time (s)'].iloc[recording_sections[0][0]]

    else:
        # Just set ret_behavior to None and print an error message
        print(f"Error: No behavior data found for {animal_id}")
        ret_behavior = None

    # Retrieve the timestamps
    if animal_id in scope_times and len(scope_times[animal_id]) > 1:
        # Concatenate datasets, accounting for gaps in recording using behavior miniscope record active.

        # Initialize ret_timestamps to the first timestamps dataset
        first_key = list(scope_times[animal_id].keys())[0]
        ret_timestamps = scope_times[animal_id][first_key]

        for idx in range(1, len(scope_times[animal_id])):
            # Get the current timestamps dataset
            key = list(scope_times[animal_id].keys())[idx]
            timestamps = scope_times[animal_id][key]

            # Get the gap based off of the time in behavior data, i.e. start of second part - end of first part
            start_of_next_recording = ret_behavior['Time (s)'].iloc[recording_sections[idx][0]]
            end_of_prev_recording = ret_behavior['Time (s)'].iloc[recording_sections[idx - 1][1]]
            gap = start_of_next_recording - end_of_prev_recording

            # Add the gap to the current timestamps dataset along with end of the previous timestamps dataset
            timestamps['Time Stamp (s)'] += ret_timestamps['Time Stamp (s)'].iloc[-1] + gap

            # Concatenate the current timestamps dataset with the previous timestamps dataset
            ret_timestamps = pd.concat([ret_timestamps, timestamps], ignore_index=True)
    else:
        # Just set ret_timestamps to the singular timestamps dataset
        first_key = list(scope_times[animal_id].keys())[0]
        ret_timestamps = scope_times[animal_id][first_key]

    # Sanity check
    # Check if miniscope recording active - 1 is similar value as concatenated ret_timestamps last value
    if verbose:
        print(f"Behavior end of recording: {ret_behavior['Miniscope record active'].iloc[-1] - 1}")
        print(f"Timestamp last value {ret_timestamps['Time Stamp (s)'].iloc[-1]}")
        print(f"time_diff: {time_diff}")
        print(f"recording sections: {recording_sections}")
        print(f"breaks: {breaks}")
        
    return ret_timestamps, ret_behavior

# Main computing function for Step 3 and 3
def process_and_align(animal_id, id_path, scope_times, behavior_data, verbose=False):
    """
    Processes and aligns the given minian dataset with the given scope times and behavior data.

    Args:
        animal_id (str): The ID of the animal for which data needs to be processed and aligned.
        id_path (str): The path to the animal ID directory.
        scope_times (dict): A dictionary containing scope times for different animal IDs.
        behavior_data (dict): A dictionary containing behavioral data for different animal IDs.

    Returns:
        tuple: A tuple containing the processed and aligned trace data and labels.
    """
    # Get minian_ds
    dpath = os.path.join(id_path, animal_id)
    minian_ds_path = os.path.join(dpath, "minian")
    if not os.path.exists(minian_ds_path) and verbose:
        print(f"Minian dataset path {minian_ds_path} not found for animal ID {animal_id}")
        return (False, animal_id)
    
    if verbose:
        print(f"Processing animal ID {animal_id} from minian_ds_path: {minian_ds_path}")

    minian_ds = open_minian(minian_ds_path)

    # Step 3 Spikes
    tracenew_spike, labelsnew_spike, tracenew_calcium, labelsnew_calcium = process_spikes_and_calcium(minian_ds)

    # Step 4 align and interpolate
    animal_timestamps, animal_behavior = combine_datasets(scope_times, behavior_data, animal_id, verbose)

    # Calcium
    tracealigned_calcium, labelsaligned_calcium = align_and_interpolate(animal_timestamps, 
                                                                        animal_behavior, 
                                                                        tracenew_calcium, 
                                                                        labelsnew_calcium)
    # Spike
    tracealigned_spike, labelsaligned_spike = align_and_interpolate(animal_timestamps,
                                                                    animal_behavior,
                                                                    tracenew_spike,
                                                                    labelsnew_spike)
    
    # Step 5 Save
    # Reference:
    # output_path_spike = r"E:\Xu\Miniscope\PL\Raw Data\Satiation\Session Combined\Spikes"   
    # output_path_calcium = r"E:\Xu\Miniscope\PL\Raw Data\Satiation\Session Combined\Calcium" 
    output_path_spike = os.path.join(id_path, "Spikes")
    output_path_calcium = os.path.join(id_path, "Calcium")

    # Save calcium
    save_trace_and_labels(tracealigned_calcium, labelsaligned_calcium, output_path_calcium, animal_id)
    if verbose:
        print(f"Saved {animal_id} Calciums to {output_path_calcium}")
    # Save spike
    save_trace_and_labels(tracealigned_spike, labelsaligned_spike, output_path_spike, animal_id)
    if verbose:
        print(f"Saved {animal_id} Spikes to {output_path_spike}")

    if verbose:
        print(f"Successfully aligned animal ID {animal_id}")

    return (True, animal_id)


# Step 3 Spikes and Calcium
def process_spikes_and_calcium(minian_ds, verbose=False):
    """
    Processes spikes and calcium data from the given minian dataset.

    Args:
        minian_ds (xarray.core.dataset.Dataset): The minian dataset.

    Returns:
        tuple: A tuple containing the processed spike and calcium data.
    
    """
    # Process Spikes
    tracenew_spike, labelsnew_spike = _process_helper(minian_ds, "S", verbose)

    # Process Calcium
    tracenew_calcium, labelsnew_calcium = _process_helper(minian_ds, "C", verbose)

    return tracenew_spike, labelsnew_spike, tracenew_calcium, labelsnew_calcium

# Part of step 3
def _process_helper(minian_ds, label_str, verbose=False):
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

    if verbose:
        print('Deleted ' + str(neuron_del) + ' neurons out of the original ' + str(neuron_ori) + ' neurons, ' + str(neuron_now) + ' neurons remain')

    return tracenew, labelsnew

# Step 4 align and interpolate
def align_and_interpolate(animal_timestamps, animal_behavior, tracenew, labelsnew):
    """
    Aligns and interpolates behavioral data with calcium timestamps.

    Args:
        timestamp (pandas.core.frame.DataFrame): The calcium timestamps.
        behavior_data (pandas.core.frame.DataFrame): The behavioral data.
        tracenew (numpy.ndarray): The calcium trace data.
        labelsnew (numpy.ndarray): The labels for the calcium trace data.

    Returns:
        tuple: A tuple containing the aligned trace data and labels.

    """
    # Load calcium timestamps from a CSV file
    CA = animal_timestamps

    # Load behavioral data from another CSV file
    Behavior = animal_behavior

    # Extract time column from behavioral data and adjust it for when miniscope record active is triggered
    behaviortime = Behavior['Time (s)'].values
    behaviortime = behaviortime - behaviortime[np.where(Behavior['Miniscope record active'] > 0)[0][0]]

    # Extract time column from calcium timestamps
    catime = CA['Time Stamp (s)'].values

    # Extract cue and bar columns from the behavioral data
    cue = Behavior['Tone active'].values
    bar = Behavior['Bar Press active'].values

    # Interpolate the behavioral data to align with the calcium timestamps
    cuealigned = np.interp(catime, behaviortime, cue)
    baraligned = np.interp(catime, behaviortime, bar)

    # Check if miniscope data is shorter than timestamp data and pad if necessary ?
    if tracenew.shape[0] < len(catime):
        padding_length = len(catime) - tracenew.shape[0]
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


if __name__ == "__main__":
    main()


