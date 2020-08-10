from __future__ import annotations
from typing import Optional

import numpy as np


class State:
    def transition_state(self, action) -> State:
        """
        Return a new state that results from playing action at current State
        """
        raise NotImplementedError

    def get_actions(self) -> list:
        """
        Return list of legal actions to take at this state.
        """
        raise NotImplementedError

    def get_representation(self) -> np.ndarray:
        """
        Return an array representation of this state
        """
        raise NotImplementedError

    def is_end(self) -> Optional[int]:
        """
        If is end state, return the reward, otherwise return None.
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

class Nnet:
    def load_weights(self, path: str):
        """Loads weights into the neural network from storage in path.
        """
        raise NotImplementedError

    def save_weights(self, path: str):
        """Saves weights to storage indicated by path.
        """
        raise NotImplementedError

    def train_model(self, inputs: list, expected_values: list, expected_policies: list):
        """Train the model using inputs and expected_values.

        Optional: If you want to implement a progress bar, use tqdm.
        """
        raise NotImplementedError

    def evaluate(self, inputs):
        raise NotImplementedError

    def action_to_index(self, action) -> int:
        """
          Maps an action to an index on the policy vector.
        :param action: the action to be mapped.
        :return: the index of the action on the policy vector
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

    def result(self) -> str:
        """
        Return '1-0' if player 1 wins. Return '0-1' if player 2 wins, and return '*' if game has not ended, or is a tie.
        """

        raise NotImplementedError


