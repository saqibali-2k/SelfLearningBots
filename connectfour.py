from interfaces import *
import numpy as np
from copy import deepcopy
from ResidualModel import DefaultModel

EPOCHS = 5
BATCH_SIZE = 512


class Connect4Game(Game):
    def __init__(self):
        self.game_over = False
        self.reward = None
        self._result = '*'
        self.p1_array = np.zeros((6, 7))
        self.p2_array = np.zeros((6, 7))
        # turn 1 = p1, turn -1 = p2
        self.turn = 1

    def state(self) -> State:
        return Connect4State(deepcopy(self))

    def take_action(self, action: int):
        if action < 0 or action > 6 or self.p1_array[0, action] == 1 or self.p2_array[0, action] == 1:
            raise ValueError
        else:
            if self.turn == 1:
                arr = self.p1_array
            else:
                arr = self.p2_array

            i = 0
            for i in range(5, -1, -1):
                if self.p1_array[i, action] == 0 and self.p2_array[i, action] == 0:
                    arr[i, action] = 1
                    break

            # Check if four in a row occurred
            self.game_over = self._check_four(i, action, arr)
            if self.game_over:
                self.reward = 1
                self._result = {1: "1-0", -1: "0-1"}[self.turn]

            tie = True
            for i in range(7):
                if self.p1_array[0, i] == 0 and self.p2_array[0, i] == 0:
                    tie = False
                    break
            if tie:
                self.game_over = True
                self.reward = 0

            # Switch turns
            self.turn *= -1

    @staticmethod
    def _follow_line(row: int, col: int, dir_x: int, dir_y: int, arr):
        count = 0
        while 0 <= row < 6 and 0 <= col < 7 and count != 4:
            if arr[row, col] == 1:
                col += dir_x
                row += dir_y
                count += 1
            else:
                return False
        return count == 4

    def _check_four(self, row: int, col: int, arr):

        four_in_row = False

        for i in range(3):
            four_in_row = self._follow_line(row, col + i, -1, 0, arr)
            four_in_row = self._follow_line(row + i, col + i, -1, -1, arr) or four_in_row
            four_in_row = self._follow_line(row + i, col, 0, -1, arr) or four_in_row
            four_in_row = self._follow_line(row + i, col - i, 1, -1, arr) or four_in_row
            four_in_row = self._follow_line(row, col - i, 1, 0, arr) or four_in_row
            four_in_row = self._follow_line(row - i, col - i, 1, 1, arr) or four_in_row
            four_in_row = self._follow_line(row - i, col, 0, 1, arr) or four_in_row
            four_in_row = self._follow_line(row - i, col + i, -1, 1, arr) or four_in_row

            if four_in_row:
                return four_in_row

        return four_in_row

    def is_over(self) -> bool:
        return self.game_over

    def result(self) -> str:
        return self._result

    def get_actions(self):
        lst = []
        if not self.is_over():
            for i in range(7):
                slots_used = self.p1_array[:, i].sum() + self.p2_array[:, i].sum()
                assert slots_used <= 6
                if slots_used != 6:
                    lst.append(i)
        return lst


class Connect4State(State):
    def __init__(self, game: Connect4Game):
        self.game = game

    def transition_state(self, action) -> State:
        new_state = deepcopy(self)
        new_state.game.take_action(action)
        return new_state

    def get_actions(self) -> list:
        return self.game.get_actions()

    def get_representation(self) -> tuple:
        return self.game.p1_array, self.game.p2_array

    def is_end(self) -> Optional[int]:
        if self.game.is_over():
            return self.game.reward

    def get_nn_input(self):
        p1_board, p2_board = self.get_representation()
        turn_board = np.zeros((6, 7))
        turn_board[:, :] = self.game.turn
        return np.array([p1_board, p2_board, turn_board])

    def policy_size(self) -> int:
        return 7


class Connect4WrapperModel(DefaultModel):

    def __init__(self):
        super().__init__(input_size=(3, 6, 7), policy_size=7, num_res_blocks=3)

    def action_to_index(self, action):
        # actions range from 0-6 (column on the connect4 board), this is directly mapped to a vector of length 7
        return action
