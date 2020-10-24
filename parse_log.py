import numpy as np
import pandas as pd
from glob import glob
import h5py
import re
import sys
import os, os.path
from world import T_INDEX
from graph_features import get_graph_features

"""Draft of a log parsing script. This should make it easy to take a log
file and create the data types and files we want for each game.
"""
#turn to false in vim, before running on big file directory
debug=False
log_file_format = '.txt' #joey has used this
log_file_format = '.log'

AREA_INDEX = {'North America': 0,
              'South America': 1,
              'Africa': 2,
              'Europe': 3,
              'Asia': 4,
              'Australia': 5}

def list_player_continents(player_num):
    """  A robust way to select which columns are needed to compute
    continental control and rewards.

    Parameters:
        player_num (int): the index pertaining to the player

    Returns:
        x (list): A list that is used to select specific columns in dataframe

     """
    x = [f'Player {player_num} North America',
         f'Player {player_num} South America',
         f'Player {player_num} Africa',
         f'Player {player_num} Europe',
         f'Player {player_num} Asia',
         f'Player {player_num} Australia']
    return x

def list_player_countries(player_num):
    """ A robust way to select which columns are needed to compute total
    troop count and other features, pertaining to state of the board

    Parameters:
        player_num (int): the index pertaining to the player

    Returns:
        x (list): A list that is used to select specific columns in dataframe

    """

    x = [f'Player {player_num} Alaska',
         f'Player {player_num} Northwest Territories',
         f'Player {player_num} Greenland',
         f'Player {player_num} Alberta',
         f'Player {player_num} Ontario',
         f'Player {player_num} Quebec',
         f'Player {player_num} Western United States',
         f'Player {player_num} Eastern United States',
         f'Player {player_num} Mexico',
         f'Player {player_num} Venezuala',
         f'Player {player_num} Peru',
         f'Player {player_num} Argentina',
         f'Player {player_num} Brazil',
         f'Player {player_num} Iceland',
         f'Player {player_num} Great Britain',
         f'Player {player_num} Scandanavia',
         f'Player {player_num} Western Europe',
         f'Player {player_num} Northern Europe',
         f'Player {player_num} Southern Europe',
         f'Player {player_num} Ukraine',
         f'Player {player_num} North Africa',
         f'Player {player_num} Egypt',
         f'Player {player_num} East Africa',
         f'Player {player_num} Congo',
         f'Player {player_num} South Africa',
         f'Player {player_num} Madagascar',
         f'Player {player_num} Middle East',
         f'Player {player_num} Ural',
         f'Player {player_num} Siberia',
         f'Player {player_num} Yakutsk',
         f'Player {player_num} Irkutsk',
         f'Player {player_num} Kamchatka',
         f'Player {player_num} Afghanistan',
         f'Player {player_num} Mongolia',
         f'Player {player_num} China',
         f'Player {player_num} Japan',
         f'Player {player_num} India',
         f'Player {player_num} South East Asia',
         f'Player {player_num} Indonesia',
         f'Player {player_num} New Guinea',
         f'Player {player_num} Western Australia',
         f'Player {player_num} Eastern Australia']

    return x

def troop_income_due_to_country_possesion(s):
    """
    Get the portion of troop income pertaining to country count

    Parameters
        s : the number of countries the player has
        i : the index pertaining to the player

    Returns:
        n (int): number of troops to receive next turn
    """
    # if a player has no countries they will receive no incoming troops
    if s == 0:
        return 0
    # each player receives at least 3 troops per turn, must have 12
    # countries or more to get 4+ troops
    if s < 12:
        return 3
    else:
        return s // 3

def get_board_names_for_players(num_players):
    """ Similar to list_player_countries, this returns a 1d list
    of all the countries from n players in order, thus robustly
    selecting the columns that describe the state of the board. This
    function is useful for carefully choosing the input for graph features

    Output is similar to the array below (which is for a 6 player game):

    array(['Player 0 Alaska', 'Player 0 Northwest Territories',
       'Player 0 Greenland',

       ....

       ,'Player 5 New Guinea',
       'Player 5 Western Australia', 'Player 5 Eastern Australia'])


    Parameters:
        num_players (int): 2 \leq num_players \leq 6

    Returns:
        names_of_player_countries ((42*n,) ndarray): names of columns describing state of the game

    """

    names_of_player_countries = np.array(list_player_countries(player_num=0))
    for i in range(1,num_players):
        names_of_player_countries = np.hstack((names_of_player_countries,list_player_countries(player_num=i)))

    return names_of_player_countries

if __name__ == "__main__":
    # Take in an argument for the file name and ouput file. If none is
    # specified, use the defaults.
    num_args = len(sys.argv)
    if num_args == 1:
        input_dir = 'logs'

    else:
        input_dir = sys.argv[1]

    if num_args == 4:
        output_dir = sys.argv[2]
        output_file = sys.argv[3]

    else:
        output_dir = 'default_log_data'
        output_file = '/parsed_log'

    # Create the output folder if it doesn't already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files_to_parse = glob(input_dir + '/**/*' + log_file_format, recursive=True)
    if debug:
        print('# of files to parse',len(files_to_parse))
    for k, filename in enumerate(files_to_parse):

        if "win_summary" in filename:
            #skip win summaries because they are formatted differently
            continue

        file = ""
        with open(filename, 'r') as fi:
            file = fi.read()

        # Get the list of the players and map them to integers
        p_list_pattern = re.compile("\[0, 0, 'players'.*\n")
        p_list_str = re.findall(p_list_pattern, file)[0]

        # Player name pattern
        p_name_pattern = re.compile("P;[A-Z;a-z;_]+")
        p_name_list = re.findall(p_name_pattern, p_list_str)

        # Dictionaries mapping players to indicies and vice versa
        player_index = {}
        index_player = {}
        for i, player in enumerate(p_name_list):
            player_index[player] = i
            index_player[i] = player

        num_players = len(p_name_list)

        # Get the winner of the game
        try:
            winner_pattern = re.compile("([0-9]+), 'victory', '(P;[A-Z;a-z;_]+)'")
            total_turns, winner = re.findall(winner_pattern, file)[0]
            total_turns = int(total_turns)

        # No winner, stalemate
        except IndexError:
            try:
                stalemate_pattern = re.compile("([0-9]+), 'Stalemate'")
                total_turns = re.findall(stalemate_pattern, file)[0]
                winner = "None"
                total_turns = int(total_turns)

            except Exception as e:
                print(filename, e)
                continue

        # Player areas after each turn
        player_area_pattern = re.compile("([0-9]+), 'Player Areas', '([A-Z_a-z;]+)', ['\"]\[(.*)\]['\"]\]")
        p_areas = re.findall(player_area_pattern, file)

        # Initialize empty area lists
        Area_lists = np.zeros((6, num_players, total_turns))

        # Populate the Area_lists array with correct values
        for p_area in p_areas:
            turn = int(p_area[0])
            player = p_area[1]
            players_area_list = p_area[2].replace("'", "").split(',')
            for area in players_area_list:
                if len(area.strip()) > 0:  # Don't include empty areas
                    Area_lists[AREA_INDEX[area.strip()], player_index[player], turn] = 1

        # State of the board after each turn
        board_state_pattern = re.compile("([0-9]+), 'State of the Board', '(P;[A-Z;a-z;_]+)', (\"|\')({.*})(\"|\')")
        states = re.findall(board_state_pattern, file)

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

        # Turn the Unit list into something DataFrame Friendly
        df_Unit_list = np.zeros((total_turns, 42 * num_players))
        for turn in range(total_turns):
            df_Unit_list[turn, :] = Unit_lists[:, :, turn].T.reshape(42 * num_players)

        # Get the headers for the dataframe
        header = []
        for player in p_name_list:
            for territory in T_INDEX.keys():
                header.append('Player ' + str(player_index[player]) + ' ' + territory)

        # Create and populate the dataframe
        unit_df = pd.DataFrame(df_Unit_list)
        unit_df.columns = header

        # Add area control columns
        for player in p_name_list:
            p_index = player_index[player]
            for area in AREA_INDEX.keys():
                col_title = 'Player ' + str(p_index) + ' ' + area
                unit_df[col_title] = Area_lists[AREA_INDEX[area], p_index, :]

        #create some features
        for i in range(num_players):
            #get the columns for the continental control for that player
            x = list_player_continents(i)
            # Assuming that the following ordering, and rewards per continent
            #order = [North America,South America, Africa, Europe, Asia, Australia]
            #rewards = [5,2,3,5,7,2]
            # then matrix multiplication gives the continental rewards
            unit_df[f'Player {i} Continental Reward'] = unit_df[x].values @ [5,2,3,5,7,2]

            #get number of troops per player and country count
            # then get total troop increase per player per turn
            x = list_player_countries(player_num=i)
            unit_df[f'Player {i} Troop Count'] = unit_df[x].sum(axis=1)
            unit_df[f'Player {i} Country Count'] = (unit_df[x] > 0).sum(axis=1)
            unit_df[f'Player {i} Troop Increase Due to Country Count'] = unit_df[f'Player {i} Country Count'].apply(troop_income_due_to_country_possesion)
            unit_df[f'Player {i} Total Reinforcements'] = unit_df[f'Player {i} Troop Increase Due to Country Count'] + unit_df[f'Player {i} Continental Reward']

        if debug:
            print(filename,unit_df.shape)

        # if new features are added, then these should be changed
        graph_features = ['player_cut_edges'
            , 'player_number_boundary_nodes'
            , 'player_boundary_fortifications'
            , 'player_average_boundary_fortifications'
            , 'player_connected_components']
        x = get_board_names_for_players(num_players)
        results = []
        for t in range(unit_df.shape[0]):
            l = get_graph_features(list(unit_df[x].iloc[t].values))
            results.append(l)
        r = np.array(results)
        #this is just to make sure the reshaping is done correctly
        # unit_df['graph_features'] = [x for x in np.array(results)]
        #get names for new columns / features
        new_names = []
        for feature in graph_features:
            for i in range(num_players):
                new_names.append(f'Player {i} {feature}')
        #add the features
        new = pd.DataFrame(r.reshape(unit_df.shape[0],len(graph_features)*num_players),columns=new_names)
        unit_df = unit_df.merge(new,how='inner',left_index=True,right_index=True)

        # Add winner column
        if winner is 'None':
            unit_df['winner'] = np.nan
            for place in ['Second', 'Third', 'Fourth', 'Fifth', 'Sixth'][:num_players - 1]:
                unit_df[place] = np.nan
            for i in range(num_players): #add per player soft score
                unit_df[f'Player {i} soft score'] = num_players**-1
        else:
            unit_df['winner'] = player_index[winner]
            unit_df[f'Player {int(player_index[winner])} soft score'] = 1
            # Add Loser columns columns
            loser_pattern = re.compile("'elimination', .*, '(P;[A-Z;a-z;_]+)'")
            losers = re.findall(loser_pattern, file)
            for i, place in enumerate(['Second', 'Third', 'Fourth', 'Fifth', 'Sixth'][:num_players - 1]):
                unit_df[place] = int(player_index[losers[-1]])
                unit_df[f'Player {int(player_index[losers[-1]])} soft score'] = (i+2)**-1
                losers.pop()
                
        # Save the dataframe to the hdf file.
        unit_df.to_hdf(output_dir + "/" + output_file + str(k) + '.hdf', 'dataframe')

        # Dictionary of other data that we may care about (other features)
        data = {"players": [(np.string_(p), player_index[p]) for p in p_name_list]}  # HDF5 is picky about strings

        # Save as an HDF
        with h5py.File(output_dir + "/" + output_file + str(k) + '.hdf', 'a') as hf:
            for k in data.keys():
                hf[k] = data[k]
