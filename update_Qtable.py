import random
import numpy as np
import cv2
import time
from matplotlib.pyplot import plot
import pandas as pd
import os
from cube_model import Rcube
import math
from statistics import mean


def get_q_table_reward(table, state_string):
    if table[table['state'] == state_string].empty:
        n_moves = table.columns.shape[0]-1
        table = table.append(pd.Series([state_string] + [0] * n_moves, index=table.columns),
                             ignore_index=True)

    return table, table[table['state'] == state_string].values[0][1:].mean()


if __name__ == "__main__":
    rubik_cube = Rcube()
    q_table_filename = 'q_table.pkl'

    complimentary_move = {'rpu': ['rpd', 'lpu'],
                          'rpd': ['rpu', 'lpd'],
                          'lpu': ['lpd', 'rpu'],
                          'lpd': ['lpu', 'rpd'],
                          'uyr': ['uyl', 'dyr'],
                          'uyl': ['uyr', 'dyl'],
                          'dyr': ['dyl', 'uyr'],
                          'dyl': ['dyr', 'uyl'],
                          'frr': ['frl', 'brr'],
                          'frl': ['frr', 'brl'],
                          'brr': ['brl', 'frr'],
                          'brl': ['brr', 'frl']}

    if os.path.exists(q_table_filename):  # check for the pickle file
        q_table = pd.read_pickle(q_table_filename)
        print('Q-table loaded')
    else:
        q_table = pd.DataFrame(columns=['state'] + rubik_cube.moves)

    alpha = 0.5
    discount_factor = 0.8 
    state_mode = 'pattern'

    max_era = 1000
    for era in range(1, max_era):
        rubik_cube.factory_reset()
        rubik_cube.change_perspective_random()

        max_steps = 100
        step = 0
        prev_move = 'rpu'
        while step < max_steps:
            present_state = rubik_cube.state_string(return_mode=state_mode)
            jackpot_reward = rubik_cube.get_reward()
            q_table, q_table_mean_reward = get_q_table_reward(q_table, present_state)

            present_move_random = random.choice(rubik_cube.moves)
            while present_move_random in complimentary_move[prev_move]:
                present_move_random = random.choice(rubik_cube.moves)
            rubik_cube.orient(present_move_random)

            next_state = rubik_cube.state_string(return_mode=state_mode)
            q_table, _ = get_q_table_reward(q_table, next_state)

            # ------------------ Q TABLE UPDATE -------------------------------------------------------------
            complement_present_move = random.choice(complimentary_move[present_move_random])
            q_value = (1 - alpha) * q_table[q_table['state'] == next_state][complement_present_move].values[0] \
                    + alpha * (jackpot_reward + discount_factor * q_table_mean_reward)
            q_table.loc[q_table['state'] == next_state, complement_present_move] = q_value

            prev_move = present_move_random
            step += 1


        # ------------------ TRY SOLVING -------------------------------------------------------------
        # count = 0
        # while not (rubik_cube.faces_solved() == 6 or count > max_steps):
        #     try_table = 0
        #     while try_table < 24:
        #         if try_table:
        #             rubik_cube.change_perspective_random(1)
        #         present_state = rubik_cube.state_string(state_mode)
        #         if not q_table[q_table['state'] == present_state].empty:
        #             a = q_table.loc[q_table['state'] == present_state, :].values[0].tolist()[1:]
        #             if sum(a) == 0:
        #                 present_move_table = random.choice(rubik_cube.moves)
        #                 try_table += 1
        #             else:
        #                 present_move_table = rubik_cube.moves[a.index(max(a))]
        #                 try_table += 100
        #         else:
        #             present_move_table = random.choice(rubik_cube.moves)
        #             try_table += 1
        #             print('==================')
        #
        #
        #     # ------------------ ORIENT CUBE -------------------------------------------------------------
        #     rubik_cube.orient(present_move_table)
        #     count += 1

        q_table.to_pickle(q_table_filename)
        print("era = {}, steps taken = {}".format(era, 0))



