"""Draft of a log parsing script. This should make it easy to take a log
file and create the data types and files we want for each game.
"""

"""
Saves dictionary via HDF
{"Unit List": State_of_Board_array,
"Turns": Number of turns, 
"Winner": Who won, 
"Players": List of players}
"""


import numpy as np
import h5py
import re
import sys
import os


if __name__ == "__main__":
    # Take in an argument for the file name and ouput file. If none is
    # specified, use the defaults.
    num_args = len(sys.argv)
    if num_args == 1:
        filename = 'pyrisk.log'

    else:
        filename = sys.argv[1]

    if num_args == 4:
        output_dir = sys.argv[2]
        output_file = sys.argv[3]

    else:
        output_dir = 'default_log_data'
        output_file = '/parsed_log.hdf'

    # Create the output folder if it doesn't already exist
    if not os.path.exists(os.getcwd() + '/' + output_dir):
        os.system('mkdir {}'.format(output_dir))

    file = ""
    with open(filename, 'r') as fi:
        file = fi.read()

    # Get the list of the players and map them to integers
    p_list_pattern = re.compile("INFO:pyrisk:\[0, 0, 'players'.*\n")
    p_list_str = re.findall(p_list_pattern, file)[0]

    # Player name pattern
    p_name_pattern = re.compile("P;[A-Z;a-z]+")
    p_name_list = re.findall(p_name_pattern, p_list_str)

    # Dictionaries mapping players to indicies and vice versa
    player_index = {}
    index_player = {}
    for i, player in enumerate(p_name_list):
        player_index[player] = i
        index_player[i] = player

    num_players = len(p_name_list)

    # Get the winner of the game
    winner_pattern = re.compile("([0-9]+), 'victory', '(P;[A-Z;a-z]+)'")
    total_turns, winner = re.findall(winner_pattern, file)[0]
    total_turns = int(total_turns)

    # State of the board after each turn
    board_state_pattern = re.compile("([0-9]+), 'State of the Board', '(P;[A-Z;a-z]+)', (\"|\')({.*})(\"|\')")
    states = re.findall(board_state_pattern, file)

    # Index of territories to their integer index
    T_INDEX = {'Alaska': 0,
               'Northwest Territories': 1,
               'Greenland': 2,
               'Alberta': 3,
               'Ontario': 4,
               'Quebec': 5,
               'Western United States': 6,
               'Eastern United States': 7,
               'Mexico': 8,
               'Venezuala': 9,
               'Peru': 10,
               'Argentina': 11,
               'Brazil': 12,
               'Iceland': 13,
               'Great Britain': 14,
               'Scandanavia': 15,
               'Western Europe': 16,
               'Northern Europe': 17,
               'Southern Europe': 18,
               'Ukraine': 19,
               'North Africa': 20,
               'Egypt': 21,
               'East Africa': 22,
               'Congo': 23,
               'South Africa': 24,
               'Madagascar': 25,
               'Middle East': 26,
               'Ural': 27,
               'Siberia': 28,
               'Yakutsk': 29,
               'Irkutsk': 30,
               'Kamchatka': 31,
               'Afghanistan': 32,
               'Mongolia': 33,
               'China': 34,
               'Japan': 35,
               'India': 36,
               'South East Asia': 37,
               'Indonesia': 38,
               'New Guinea': 39,
               'Western Australia': 40,
               'Eastern Australia': 41}

    # Initialize empty unit list arrays
    Unit_lists = np.zeros((42, num_players, total_turns))
    territory_forces = re.compile("\'([a-zA-z ]+)\': ([0-9]+)")

    # Gather the unit lists at every turn of the game.
    for state in states:
        turn = int(state[0])
        player = state[1]
        forces = re.findall(territory_forces, state[3])
        for territory, troop_count in forces:
            Unit_lists[T_INDEX[territory], player_index[player], turn] = int(troop_count)

    # TODO : Save all the formats that we care about
    data = {"Unit List": Unit_lists,
            "Turns": total_turns,
            "Winner": winner,
            "Players": [np.string_(p) for p in p_name_list]}  # HDF5 is picky about strings

    # Save as an HDF
    hf = h5py.File(output_dir + "/" + output_file, 'w')
    for k in data.keys():
        hf[k] = data[k]

    hf.close()

