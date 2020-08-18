from __future__ import annotations
from typing import Optional
import numpy as np


class State:
    def transition_state(self, action) -> State:
        """
        Return a new state that results from playing action at current State
        """
        raise NotImplementedError

    def get_actions(self) -> np.ndarray:
        """
        Return vector of
        """
        raise NotImplementedError

    def is_end(self) -> Optional[int]:
        """
        If is end state, return the reward, otherwise return None. If player whose turn it is won, return 1, else return
        -1.
        """
        raise NotImplementedError

    def get_nn_input(self):
        """
        Return the inputs for your neural network. This will be called when predicting during self-play, and
        generating training data for training.
        :return:
        """
        raise NotImplementedError

    def policy_size(self) -> int:
        """
        Return the size of the policy vector, that is the maximum number of possible moves at a given time.
        """
        raise NotImplementedError


class Game:
    def state(self) -> State:
        """
        Return a State instance representing the current state of the game.
        """
        raise NotImplementedError

    def take_action(self, action):
        """
        Play the action in the game.
        :param action: the action to take
        :return:
        """
        raise NotImplementedError

    def is_over(self) -> bool:
        """
       Return true iff the game has ended.
       """

        raise NotImplementedError

    def get_reward(self) -> Optional[int]:
        """
        Return 1, if the player whose turn it is won. Return -1 if the player whose turn it is lost. Return 0 for ties.
        If game has not ended. Return None.
        """

        raise NotImplementedError

    def result(self) -> str:
        """
        Return '1-0' if player 1 wins. Return '0-1' if player 2 wins, and return '*' if game has not ended, or is a tie.
        """

        raise NotImplementedError


# Does not need to be implemented, you may create a wrapper for DefaultModel in ResidualModel.py
class Nnet:
    def load_weights(self, path: str):
        """Loads weights into the neural network from storage in path.
        """
        raise NotImplementedError

    def save_weights(self, path: str):
        """Saves weights to storage indicated by path.
        """
        raise NotImplementedError

    def train_model(self, lr: int, inputs: list, expected_values: list, expected_policies: list):
        """Train the model using inputs and expected_values.

        Optional: If you want to implement a progress bar, use tqdm.
        """
        raise NotImplementedError

    def evaluate(self, inputs):
        raise NotImplementedError

    @staticmethod
    def action_to_index(action) -> int:
        """
          Maps an action to an index on the policy vector.
        :param action: the action to be mapped.
        :return: the index of the action on the policy vector
        """
        raise NotImplementedError

    @staticmethod
    def index_to_action(index):
        """
          Maps an index to an action on the policy vector.
        :param index: the index to be mapped.
        :return: the action of the index on the policy vector
        """
        raise NotImplementedError
