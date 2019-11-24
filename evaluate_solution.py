import random
import numpy as np
import cv2
import time
from matplotlib.pyplot import plot
import pandas as pd
import os
from cube_model import Rcube
import math

rubik_cube = Rcube()
state_mode = 'pattern'
q_table = pd.read_pickle('q_table.pkl')


for i_scramble in range(1, 20):
    count_list = []
    for _ in range(50):
        rubik_cube.factory_reset()
        rubik_cube.scramble_up(i_scramble)

        count = 0
        while not (rubik_cube.faces_solved() == 6 or count > 100):
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

            # ------------------ ORIENT CUBE -------------------------------------------------------------
            rubik_cube.orient(present_move_table)
            count += 1
        # print(count)
        count_list.append(count)

    print("n_scramble = {}, Accuracy= {}%".format(i_scramble, 100 * sum(np.asarray(count_list) < 100)/len(count_list)))
