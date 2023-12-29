import numpy as np 
import pandas as pd

def transform_dataframe(input_path: str) -> pd.DataFrame:
    original_df = pd.read_pickle(input_path)

    # Create "Time" column
    time_steps = len(original_df)
    new_df = pd.DataFrame({"Time": np.arange(1, time_steps + 1)}) # Offset by 1 to start at 1

    # Categorical columns of interest from original dataframe
    category_map = {'Tone active': 1, 'Bar Press active': 2, 'Freezing': 3}

    # Initialize 'Label' and 'Label_num' columns
    new_df['Label'] = 'Baseline'
    new_df['Label Num'] = 0

    # Updating 'Label' and 'Label Num' based on original dataframe
    for category, num in category_map.items():
        if category in original_df.columns:
            condition = original_df[category] == 1
            new_df.loc[condition, 'Label'] = category
            new_df.loc[condition, 'Label Num'] = num

    # Updating the neurons
    max_neuron_number = max([int(column) for column in original_df.columns if column.isdigit()])

    for neuron in range(1, max_neuron_number + 1):
        neuron_col = f'X{neuron}'
        if neuron_col in original_df.columns:
            new_df[neuron_col] = original_df[neuron_col]
        else:
            new_df[neuron_col] = 0

    return new_df
