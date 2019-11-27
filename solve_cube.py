import random
import numpy as np
import cv2
import time
from matplotlib.pyplot import plot
import pandas as pd
import os
from cube_model import Rcube
import math

initial_state = {'up': np.array(['w', 'w', 'w', 'w']),
                 'left': np.array(['g', 'g', 'o', 'b']),
                 'front': np.array(['r', 'r', 'r', 'g']),
                 'right': np.array(['b', 'b', 'r', 'b']),
                 'down': np.array(['y', 'y', 'g', 'o']),
                 'back': np.array(['y', 'y', 'o', 'o'])
                 }

rubik_cube = Rcube(initial_state)
state_mode = 'pattern'
q_table = pd.read_pickle('q_table.pkl')

max_count = 10000
count = 0
while not (rubik_cube.faces_solved() == 6 or count > max_count):
    try_table = 0
    while try_table < 24:
        if try_table:
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
    print(rubik_cube.state['front'])
    rubik_cube.paint_cube(1)
    rubik_cube.orient(present_move_table)
    print("step = {}, move = {}".format(count, present_move_table))
    rubik_cube.paint_cube(1)


    count += 1

rubik_cube.paint_cube(0)
print(count)
