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

    if os.path.exists('q_table.pkl'):  # check for the pickle file
        q_table = pd.read_pickle('q_table.pkl')
        print('Q-table loaded')
    else:
        q_table = pd.DataFrame(columns=['state'] + rubik_cube.moves)

    alpha = 0.5
    discount_factor = 0.8
    state_mode = 'pattern'

    max_era = 1000

    result_era = {'table': [], 'random': [], 'balanced': []}
    result_episode = {'table': [], 'random': [], 'balanced': []}
    for era in range(1, max_era):
        max_episodes = 27
        for mode in ['table', 'random', 'balanced']:
            result_episode[mode] = []
        for episode in range(1, max_episodes):
            episode_number = (era-1)*max_episodes + episode
            rubik_cube.factory_reset()
            rubik_cube.change_perspective_random()
            if episode % 3 == 0:
                weights = [1, 0]
                mode = 'table'
            elif episode % 3 == 1:
                weights = [0, 1]
                mode = 'random'
            else:
                weights = [1, 1]
                mode = 'balanced'

            # ---------------------- SCRAMBLE CUBE ------------------------------------
            n_scramble = 2 + int(math.log(era, 4))
            while rubik_cube.faces_solved() == 6:
                rubik_cube.scramble_up(n_scramble)
            q_table, _ = get_q_table_reward(q_table, rubik_cube.state_string(state_mode))  # to add initial state to the table
            # ---------------------- TRY TO SOLVE THE CUBE ------------------------------------
            steps = 0
            max_steps = n_scramble + 2 #2**n_scramble
            while rubik_cube.faces_solved() < 6 and steps < max_steps:
                # ------------------ DECIDE THE MOVE -----------------------------------------------------------
                present_state = rubik_cube.state_string(state_mode)
                if not q_table[q_table['state'] == present_state].empty:
                    a = q_table.loc[q_table['state'] == present_state, :].values[0].tolist()[1:]
                    if sum(a) == 0:
                        present_move_table = random.choice(rubik_cube.moves)
                    else:
                        # print("selected from Q-table:", a)
                        present_move_table = rubik_cube.moves[a.index(max(a))]
                else:
                    present_move_table = random.choice(rubik_cube.moves)

                present_move_random = random.choice(rubik_cube.moves)
                present_move = random.choices([present_move_table, present_move_random], weights=weights)[0]
                # ------------------ ORIENT CUBE -------------------------------------------------------------
                rubik_cube.orient(present_move)

                # ------------------ CALCULATE REWARD --------------------------------------------------------
                next_state = rubik_cube.state_string(return_mode=state_mode)
                # rubik_cube.paint_cube(1)

                jackpot_reward = rubik_cube.get_reward()

                q_table, q_table_reward_max = get_q_table_reward(q_table, rubik_cube.state_string(state_mode))
                try_table = 0
                while try_table < 24:
                    rubik_cube.change_perspective_random(1)
                    q_table, q_table_reward = get_q_table_reward(q_table, rubik_cube.state_string(state_mode))
                    if q_table_reward > q_table_reward_max:
                        q_table_reward_max = q_table_reward
                        try_table += 2
                    try_table += 1

                # ------------------ Q TABLE UPDATE -------------------------------------------------------------
                q_value = (1 - alpha) * q_table[q_table['state'] == present_state][present_move].values[0] + \
                           alpha * (jackpot_reward + discount_factor * q_table_reward_max)
                q_table.loc[q_table['state'] == present_state, present_move] = q_value

                steps += 1

            result_episode[mode].append(int(steps < max_steps))  # at the end of each episode

        for mode in ['table', 'random', 'balanced']:
            result_era[mode].append(round(100*mean(result_episode[mode]), 2))   # at the end of each era
        print(result_episode['table'])
        print("era = {}, n_scramble = {}, Table = {}, Random = {}, Balanced = {}".format(era,
                                                                                         n_scramble,
                                                                                         result_era['table'][-1],
                                                                                         result_era['random'][-1],
                                                                                         result_era['balanced'][-1]))

        # print("Episode = {},  Accuracy = {}% , n_scramble = {}, steps = {}".format(episode, int(accuracy), n_scramble, steps))
        q_table.to_pickle("q_table.pkl")



