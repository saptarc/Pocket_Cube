import random
import numpy as np
import cv2
import time
from matplotlib.pyplot import plot
import pandas as pd
import os
from cube_model import Rcube


def get_q_table_reward(table, state):
    if table[table['state'] == state].empty:
        n_moves = table.columns.shape[0]-1
        table = table.append(pd.Series([state] + [0] * n_moves, index=table.columns),
                             ignore_index=True)

    return table, table[table['state'] == state].values[0][1:].mean()


if __name__ == "__main__":

    initial_state = {'up':    np.array(['w', 'w', 'w', 'w']),
                     'left':  np.array(['o', 'o', 'o', 'o']),
                     'front': np.array(['g', 'g', 'g', 'g']),
                     'right': np.array(['r', 'r', 'r', 'r']),
                     'down':  np.array(['y', 'y', 'y', 'y']),
                     'back':  np.array(['b', 'b', 'b', 'b'])
                     }
    rubik_cube = Rcube(initial_state.copy())

    if os.path.exists('q_table.pkl'):  # check for the pickle file
        q_table = pd.read_pickle('q_table.pkl')
        print('Q-table loaded')
    else:
        q_table = pd.DataFrame(columns=['state'] + rubik_cube.moves)

    weights = [1000, 0]
    alpha = 0.2
    discount_factor = 0.1
    episodes = 100000

    episode_steps = 0
    for episode in range(1, episodes):
        rubik_cube.factory_reset()
        # ---------------------- SCRAMBLE CUBE ------------------------------------
        n_scramble = 1 + episode//1000

        while rubik_cube.faces_solved() == 6:
            rubik_cube.scramble_up(n_scramble)

        q_table, _ = get_q_table_reward(q_table, rubik_cube.state_string('pattern'))  # to add initial state to the table

        # ---------------------- TRY TO SOLVE THE CUBE ------------------------------------
        steps = 0
        max_steps = 20
        while rubik_cube.faces_solved() < 6 and steps < max_steps:

            # ------------------ DECIDE THE MOVE -----------------------------------------------------------
            present_state = rubik_cube.state_string('pattern')
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

            if episode % 200 == 0:
                weights = [10, 1]
            elif episode % 100 == 0:
                weights = [1, 10]
            else:
                pass
            present_move = random.choices([present_move_table, present_move_random], weights=weights)[0]

            # ------------------ ORIENT CUBE -------------------------------------------------------------
            rubik_cube.orient(present_move)

            # ------------------ CALCULATE REWARD --------------------------------------------------------
            next_state = rubik_cube.state_string(return_mode='pattern')
            jackpot_reward = rubik_cube.get_reward()
            q_table, q_table_reward = get_q_table_reward(q_table, next_state)

            # ------------------ Q TABLE UPDATE -------------------------------------------------------------
            q_value = (1 - alpha) * q_table[q_table['state'] == present_state][present_move].values[0] + \
                       alpha * (jackpot_reward + discount_factor * q_table_reward)
            q_table.loc[q_table['state'] == present_state, present_move] = q_value

            # if q_value > 0:
            #     print(q_value)
            steps += 1
            episode_steps += 1

        if episode % 100 == 0:
            accuracy = 100 - 100*(episode_steps-episode)/(episode*(max_steps-1))
            print("Episode = {},  Accuracy = {}% , n_scramble = {}, steps = {}".format(episode, int(accuracy), n_scramble, steps))
        if episode % 500 == 0:
            q_table.to_pickle("q_table.pkl")



