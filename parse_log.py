import numpy as np
import pandas as pd
from glob import glob
import h5py
import re
import sys
import os

"""Draft of a log parsing script. This should make it easy to take a log
file and create the data types and files we want for each game.
"""

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
        output_file = '/parsed_log.hdf'

    # Create the output folder if it doesn't already exist
    if not os.path.exists(os.getcwd() + '/' + output_dir):
        os.system('mkdir {}'.format(output_dir))

    files_to_parse = glob(input_dir + '/**/*.log', recursive=True)
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


        # Add winner column
        if winner is 'None':
            unit_df['winner'] = np.nan
            for place in ['Second', 'Third', 'Fourth', 'Fifth', 'Sixth'][:num_players - 1]:
                unit_df[place] = np.nan
        else:
            unit_df['winner'] = player_index[winner]
            # Add Loser columns columns
            loser_pattern = re.compile("'elimination', .*, '(P;[A-Z;a-z;_]+)'")
            losers = re.findall(loser_pattern, file)
            for place in ['Second', 'Third', 'Fourth', 'Fifth', 'Sixth'][:num_players - 1]:
                unit_df[place] = int(player_index[losers[-1]])
                losers.pop()

        # Save the dataframe to the hdf file.
        unit_df.to_hdf(output_dir + "/" + output_file + str(k) + '.hdf', 'dataframe')

        # Dictionary of other data that we may care about (other features)
        data = {"players": [(np.string_(p), player_index[p]) for p in p_name_list]}  # HDF5 is picky about strings

        # Save as an HDF
        hf = h5py.File(output_dir + "/" + output_file + str(k) + '.hdf', 'a')
        for k in data.keys():
            hf[k] = data[k]

        hf.close()
