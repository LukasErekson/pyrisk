""" Takes in a list of folders and loads each dataframe, extracting the
nth turn from each DataFrame (where possible) and puts it into a single
DataFrame.

Format
------
python3 nth_turn_df.py dir_list turn_num output_name

Example
-------
To run the script going through dir1, dir2, and dir3 (and their
subdirectories) and extract the 57th turn from each *.hdf file and save the
resulting DataFrame to 57_turn_df.hdf, run the following command:

$ python3 nth_turn_df.py [dir1,dir2,dir3] 57 57_turn_df
"""

import sys
import h5py
import pandas as pd
from glob import glob

if __name__ == '__main__':
    assert len(sys.argv) == 4, "Please input list of directories, turn number, and output file name."

    # Extract the list of directories
    dirs = sys.argv[1]

    # Get rid of brackets if they exist
    if dirs[0] == '[':
        dirs = dirs[1:-1].split(',')
    else:
        dirs = dirs.split(',')

    # Get the turn number to extract
    turn_num = int(sys.argv[2])

    # Get the output file name for the DataFrame
    output = sys.argv[3]

    # Initialize the empty output DataFrame
    out_df = pd.DataFrame()

    # For each directory, load the dataframe, take the nth turn, and
    for dir in dirs:
        # Get the files in each directory and subdirectory
        files = glob(dir + "/**/*.hdf", recursive=True)
        
        for file in files:
            # Load each DataFrame file
            file_df = pd.read_hdf(file)
            player_file = h5py.File(file, 'r')

            # Get the nth turn from each DataFrame (that has it)
            try:
                out_df = out_df.append(file_df.iloc[turn_num])

                # Add column if it's not in there yet
                # TODO : Nicer way to do this not in a for loop?
                if 'Player 0 AI' not in out_df.columns:
                    for i in range(len(player_file['players'])):
                        out_df[f'Player {i} AI'] = ''

                # Get player AI types
                for AItype, num in player_file['players']:
                    # Have the AI Type in the appropriate column
                    out_df[f'Player {int(num)} AI'].iloc[-1] = str(AItype).split(';')[-1][:-1]

            # Continue if the game doesn't have this particular turn number
            except IndexError:
                continue

            finally:
                # Close the player_file
                player_file.close()
                
                # Delete for memory
                del file_df

    # Reassign the index
    out_df.index = range(out_df.shape[0])

    # Save the dataframe output
    out_df.to_hdf(output + '.hdf', 'dataframe')
