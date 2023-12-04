# Calcium-Behavior-Alignment
Align miniscope data with behavior

## Command Line Interface (CLI) Overview
In addition to modulated functions for Jupyter notebook usage, there is also a CLI functionality for running on the terminal. Below is an overview of how to use the CLI for the calcium behavior alignment script.

- The script can be executed from within the directory of `calcium_behavior_alignment.py` 

### Usage
`calcium_behavior_alignment.py [-h] [-n NUM_PROCESSES] [-v] exp_path id_path beh_path experiment`

### Positional Arguments
1. **exp_path**: The path to the experiment directory. 
   - Example: `/scratch/09117/xz6783/minian/Satiation/`
2. **id_path**: The path to the animal ID directory.
   - Example: `/scratch/09117/xz6783/minian/Satiation/Session Combined/`
3. **beh_path**: The path to the behavioral data directory.
   - Example: `/scratch/09117/xz6783/minian/Behavior_Files/`
4. **experiment**: The name of the experiment.
   - Example: `Satiation`

### Optional Arguments
- `-h, --help`: Display the help message and exit the program.
- `-n, --num_processes`: The number of processes to use for multiprocessing (default: 2)
- `-v, --verbose`: Whether to print verbose output. Helpful for debugging.

### Example
``` bash
python calcium_behavior_alignment.py /scratch/09117/xz6783/minian/Satiation/ /scratch/09117/xz6783/minian/Satiation/Session Combined/ /scratch/09117/xz6783/minian/Behavior_Files/ Satiation -n 5 -v
```

### Important Note:
- Ensure that you are running the python script on an environment with minian installed, e.g. `conda activate minian`
