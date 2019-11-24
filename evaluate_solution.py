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
max_count = 20

for i_scramble in range(1, 20):
    count_list = []
    for _ in range(100):
        rubik_cube.factory_reset()
        while rubik_cube.faces_solved() == 6:
            rubik_cube.scramble_up(i_scramble)
        rubik_cube.paint_cube(1, 'input')
        count = 0
        while not (rubik_cube.faces_solved() == 6 or count > max_count):
            try_table = 0
            while try_table < 24:
                rubik_cube.change_perspective_random(1)
                present_state = rubik_cube.state_string(state_mode)

                if not q_table[q_table['state'] == present_state].empty:
                    a = q_table.loc[q_table['state'] == present_state, :].values[0].tolist()[1:]
                    if sum(a) == 0:
                        present_move_table = random.choice(rubik_cube.moves)
                        try_table += 1
                    else:
                        present_move_table = rubik_cube.moves[a.index(max(a))]
                        try_table += 100
                else:
                    present_move_table = random.choice(rubik_cube.moves)
                    try_table += 1

            # ------------------ ORIENT CUBE -------------------------------------------------------------
            rubik_cube.orient(present_move_table)
            count += 1
            # if try_table > 100:
            #     print(try_table)

        # print(count)
        count_list.append(count)

        rubik_cube.paint_cube(1)
    c = np.asarray(count_list)
    accuracy = round(100 * sum(c < max_count) / len(count_list), 2)
    n_steps_avg = round(c[c < max_count].mean(), 2)
    print("n_scramble = {}, n_steps_avg = {}, Accuracy= {}%".format(i_scramble, n_steps_avg, accuracy))
