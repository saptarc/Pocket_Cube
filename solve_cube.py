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
                 'left': np.array(['g', 'g', 'g', 'y']),
                 'front': np.array(['r', 'r', 'o', 'r']),
                 'right': np.array(['b', 'b', 'b', 'y']),
                 'down': np.array(['y', 'b', 'r', 'o']),
                 'back': np.array(['g', 'y', 'o', 'o'])
                 }

rubik_cube = Rcube(initial_state)
state_mode = 'pattern'
q_table = pd.read_pickle('q_table.pkl')

max_count = 1000
count = 0
while not (rubik_cube.faces_solved() == 6 or count > max_count):
    try_table = 0
    while try_table < 30:
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

        present_move_random = random.choice(rubik_cube.moves)
        present_move = random.choices([present_move_table, present_move_random], weights=[4, 1])[0]

        # ------------------ ORIENT CUBE -------------------------------------------------------------
        rubik_cube.orient(present_move_table)
        rubik_cube.paint_cube(1)
        print(count)
        count += 1

rubik_cube.paint_cube(0)
print(count)
