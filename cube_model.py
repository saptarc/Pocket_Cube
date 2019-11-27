import random
import numpy as np
import cv2
import time
from matplotlib.pyplot import plot
import pandas as pd
import os


class Rcube:
    def __init__(self, cube_state=None):
        self.scale = 4*10
        if cube_state:
            self.state = cube_state
        else:
            self.factory_reset()
        self.moves = ['rpu', 'rpd', 'lpu', 'lpd', 'uyr', 'uyl', 'dyr', 'dyl', 'frr', 'frl', 'brr', 'brl']
        self.color_code = {
            'r': [0, 0, 255],
            'g': [0, 255, 0],
            'y': [0, 255, 255],
            'b': [255,  0, 0],
            'o': [51, 153, 255],
            'w': [255, 255, 255],
            'k': [0, 0, 0]
        }
        self.opposite_color = {
            'r': 'o',
            'o': 'r',
            'w': 'y',
            'y': 'w',
            'g': 'b',
            'b': 'g'
        }
        self.canvas = None

    def factory_reset(self):
        factory_state = {'up': np.array(['w', 'w', 'w', 'w']),
                         'left': np.array(['o', 'o', 'o', 'o']),
                         'front': np.array(['g', 'g', 'g', 'g']),
                         'right': np.array(['r', 'r', 'r', 'r']),
                         'down': np.array(['y', 'y', 'y', 'y']),
                         'back': np.array(['b', 'b', 'b', 'b'])
                         }
        self.state = factory_state

    def state_string(self, return_mode='full'):
        s = ''
        if return_mode == 'full':
            for face in ['up', 'front', 'down', 'back', 'left', 'right']:
                face_color_array = self.state[face]
                s = s + ''.join(face_color_array.tolist())

        elif return_mode == 'pattern':
            for face in ['up', 'front', 'down', 'back', 'left', 'right']:
                face_color_array = self.state[face]
                s = s + ''.join(face_color_array.tolist())
            front_color = self.state['front'][0]
            back_color = self.opposite_color[front_color]
            left_color = self.state['left'][1]
            right_color = self.opposite_color[left_color]
            up_color = self.state['up'][3]
            down_color = self.opposite_color[up_color]

            s = s.replace(front_color, '1')
            s = s.replace(left_color, '2')
            s = s.replace(up_color, '3')
            s = s.replace(back_color, '4')
            s = s.replace(right_color, '5')
            s = s.replace(down_color, '6')

        return s

    def paint_cube(self, wait_time=1, window_name='cube'):
        self.canvas = np.zeros([13*self.scale, 10*self.scale, 3], dtype=np.uint8)
        self.paint_face(self.state['up'], (1, 4))
        self.paint_face(self.state['front'], (4, 4))
        self.paint_face(self.state['down'], (7, 4))
        self.paint_face(self.state['back'], (10, 4))
        self.paint_face(self.state['left'], (4, 1))
        self.paint_face(self.state['right'], (4, 7))
        cv2.imshow(window_name, self.canvas)
        cv2.waitKey(wait_time)

    def paint_face(self, arr, start_pos):
        colored_face = np.zeros([2, 2, 3], dtype=np.uint8)
        colored_face[0, 0, :] = self.color_code[arr[0]]
        colored_face[0, 1, :] = self.color_code[arr[1]]
        colored_face[1, 0, :] = self.color_code[arr[3]]
        colored_face[1, 1, :] = self.color_code[arr[2]]

        colored_face = cv2.resize(colored_face, (2*self.scale, 2*self.scale), interpolation=cv2.INTER_NEAREST)
        colored_face = cv2.line(colored_face, (self.scale, 0), (self.scale, self.scale*2), color=(0, 0, 0), thickness=2)
        colored_face = cv2.line(colored_face, (0, self.scale), (2*self.scale, self.scale), color=(0, 0, 0), thickness=2)
        # cv2.imshow("face", colored_face)
        # cv2.waitKey(0)
        self.canvas[start_pos[0]*self.scale:(start_pos[0]+2)*self.scale, start_pos[1]*self.scale:(start_pos[1]+2)*self.scale,:] = colored_face

    def faces_solved(self):
        n_solved_faces = 0
        for face in self.state.keys():
            n_solved_faces += all(color == self.state[face][0] for color in self.state[face])
        return n_solved_faces

    def orient(self, cube_move):
        if cube_move == 'rpu':
            self.cycle_face('right', direction='clockwise')

            temp = self.state['front'][([1, 2])]
            self.state['front'][([1, 2])] = self.state['down'][([1, 2])]
            self.state['down'][([1, 2])] = self.state['back'][([1, 2])]
            self.state['back'][([1, 2])] = self.state['up'][([1, 2])]
            self.state['up'][([1, 2])] = temp

        if cube_move == 'rpd':
            self.cycle_face('right', direction='anticlockwise')

            temp = self.state['front'][([1, 2])]
            self.state['front'][([1, 2])] = self.state['up'][([1, 2])]
            self.state['up'][([1, 2])] = self.state['back'][([1, 2])]
            self.state['back'][([1, 2])] = self.state['down'][([1, 2])]
            self.state['down'][([1, 2])] = temp

        if cube_move == 'lpu':
            self.cycle_face('left', direction='anticlockwise')

            temp = self.state['front'][([0, 3])]
            self.state['front'][([0, 3])] = self.state['down'][([0, 3])]
            self.state['down'][([0, 3])] = self.state['back'][([0, 3])]
            self.state['back'][([0, 3])] = self.state['up'][([0, 3])]
            self.state['up'][([0, 3])] = temp
        if cube_move == 'lpd':
            self.cycle_face('left', direction='clockwise')

            temp = self.state['front'][([0, 3])]
            self.state['front'][([0, 3])] = self.state['up'][([0, 3])]
            self.state['up'][([0, 3])] = self.state['back'][([0, 3])]
            self.state['back'][([0, 3])] = self.state['down'][([0, 3])]
            self.state['down'][([0, 3])] = temp

        if cube_move == 'uyr':
            self.cycle_face('up', direction='anticlockwise')

            temp = self.state['front'][([0, 1])]
            self.state['front'][([0, 1])] = self.state['left'][([0, 1])]
            self.state['left'][([0, 1])] = self.state['back'][([2, 3])]
            self.state['back'][([2, 3])] = self.state['right'][([0, 1])]
            self.state['right'][([0, 1])] = temp

        if cube_move == 'uyl':
            self.cycle_face('up', direction='clockwise')

            temp = self.state['front'][([0, 1])]
            self.state['front'][([0, 1])] = self.state['right'][([0, 1])]
            self.state['right'][([0, 1])] = self.state['back'][([2, 3])]
            self.state['back'][([2, 3])] = self.state['left'][([0, 1])]
            self.state['left'][([0, 1])] = temp

        if cube_move == 'dyr':
            self.cycle_face('down', direction='clockwise')

            temp = self.state['front'][([2, 3])]
            self.state['front'][([2, 3])] = self.state['left'][([2, 3])]
            self.state['left'][([2, 3])] = self.state['back'][([0, 1])]
            self.state['back'][([0, 1])] = self.state['right'][([2, 3])]
            self.state['right'][([2, 3])] = temp

        if cube_move == 'dyl':
            self.cycle_face('down', direction='anticlockwise')

            temp = self.state['front'][([2, 3])]
            self.state['front'][([2, 3])] = self.state['right'][([2, 3])]
            self.state['right'][([2, 3])] = self.state['back'][([0, 1])]
            self.state['back'][([0, 1])] = self.state['left'][([2, 3])]
            self.state['left'][([2, 3])] = temp

        if cube_move == 'frr':
            self.cycle_face('front', direction='clockwise')

            temp = self.state['up'][([2, 3])]
            self.state['up'][([2, 3])] = self.state['left'][([1, 2])]
            self.state['left'][([1, 2])] = self.state['down'][([0, 1])]
            self.state['down'][([0, 1])] = self.state['right'][([3, 0])]
            self.state['right'][([3, 0])] = temp

        if cube_move == 'frl':
            self.cycle_face('front', direction='anticlockwise')

            temp = self.state['up'][([2, 3])]
            self.state['up'][([2, 3])] = self.state['right'][([3, 0])]
            self.state['right'][([3, 0])] = self.state['down'][([0, 1])]
            self.state['down'][([0, 1])] = self.state['left'][([1, 2])]
            self.state['left'][([1, 2])] = temp

        if cube_move == 'brr':
            self.cycle_face('back', direction='anticlockwise')

            temp = self.state['up'][([0, 1])]
            self.state['up'][([0, 1])] = self.state['left'][([3, 0])]
            self.state['left'][([3, 0])] = self.state['down'][([2, 3])]
            self.state['down'][([2, 3])] = self.state['right'][([1, 2])]
            self.state['right'][([1, 2])] = temp
        if cube_move == 'brl':
            self.cycle_face('back', direction='clockwise')

            temp = self.state['up'][([0, 1])]
            self.state['up'][([0, 1])] = self.state['right'][([1, 2])]
            self.state['right'][([1, 2])] = self.state['down'][([2, 3])]
            self.state['down'][([2, 3])] = self.state['left'][([3, 0])]
            self.state['left'][([3, 0])] = temp

    def cycle_face(self, cube_face, direction):
        if direction == 'clockwise':
            self.state[cube_face] = np.roll(self.state[cube_face], 1)
        else:
            self.state[cube_face] = np.roll(self.state[cube_face], -1)

    def get_reward(self):
        faces_solved_for_reward = self.faces_solved()
        if faces_solved_for_reward == 6:
            return 1000
        else:
            return 0

    def scramble_up(self, step=1):
        for _ in range(step):
            self.orient(random.choice(self.moves))

    def change_perspective_random(self, step=1):
        perspective_moves = [['rpu','lpu'], ['rpd', 'lpd'], ['uyr', 'dyr'], ['uyl', 'dyl'], ['frr', 'brr'], ['frl', 'brl']]

        for _ in range(step):
            perspective_move = random.choice(perspective_moves)
            for move in perspective_move:
                self.orient(cube_move=move)
