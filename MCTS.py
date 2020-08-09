from __future__ import annotations
from interfaces import State, Nnet, action_to_index
from typing import Tuple, Union, Optional
from random import choices
import numpy as np

NUM_SIMULATIONS = 1000
C_PUCT = 1.0


class TreeNode:
    # Maps moves/actions to states
    children: dict
    state: State

    def __init__(self, state: State):
        self.state = state
        self.W_state_val = 0
        self.N_num_visits = 0
        self.P_init_policy = None
        self.children = {}

    def get_Q(self):
        """
        Returns total value / num_visits.
        :return: W/N or expected win at the current state.
        """
        return self.W_state_val / max(self.N_num_visits, 1)


class MonteCarloTS:
    curr: TreeNode
    root: TreeNode

    def __init__(self, initial_state, neural_net: Nnet):
        self.root = initial_state
        self.curr = self.root
        self.visited = set()
        self.nnet = neural_net

    def get_best_action(self, node: TreeNode, training: bool):
        if not training:
            max_N, best = -float("inf"), None
            for move in node.state.get_actions():
                if move in node.children:
                    n = node.children[move].N_num_visits
                else:
                    n = 0
                if n > max_N:
                    max_N = n
                    best = move
            return best
        else:
            move_lst, prob = self.get_improved_policy(node)
            if len(move_lst) != 0:
                best = choices(move_lst, weights=prob)
                best = best[0]
            else:
                best = None
            return best

    def search(self, training=True):
        for _ in range(NUM_SIMULATIONS):
            self._simulation(self.curr)
        best = self.get_best_action(self.curr, training=training)
        if best is not None:
            # indicates bug, we shouldn't be choosing moves we haven't explored
            assert best in self.curr.children

            self.curr = self.curr.children[best]
        return best

    def enemy_move(self, move):
        if move in self.curr.children:
            self.curr = self.curr.children[move]
        else:
            # we let it happen
            new_state = self.curr.state.transition_state(move)
            self.curr.children[move] = TreeNode(new_state)
            self.curr = self.curr.children[move]

    def _simulation(self, curr: TreeNode):
        reward = curr.state.is_end()
        if reward is not None:
            return -1 * reward

        if curr not in self.visited:
            curr.P_init_policy, value = self.feed_network(curr)
            curr.W_state_val = 0
            curr.N_num_visits = 0
            self.visited.add(curr)
            return -1 * value
        else:
            best_action, selected_child, max_u = None, None, -float("inf")
            sum_visits = curr.N_num_visits

            for action in curr.state.get_actions():
                if action in curr.children:
                    node = curr.children[action]
                    u = node.get_Q() + C_PUCT * self.get_policy(curr, action) * (
                            np.sqrt(sum_visits) / (1 + node.N_num_visits))
                else:
                    u = C_PUCT * self.get_policy(curr, action) * np.sqrt(sum_visits + 1e-8)  # to encourage exploring

                if u > max_u:
                    max_u = u
                    best_action = action

            if best_action in curr.children:
                selected_child = curr.children[best_action]
            else:
                selected_child = TreeNode(curr.state.transition_state(best_action))
                curr.children[best_action] = selected_child

            value = self._simulation(selected_child)
            curr.N_num_visits += 1
            curr.W_state_val += value
            return -value

    def get_policy(self, node: TreeNode, action) -> float:
        return node.P_init_policy[action_to_index(action)]

    def get_improved_policy(self, curr: TreeNode, include_empty_spots: bool = False) -> Union[
        Tuple[list, list], np.ndarray]:
        array = np.zeros(curr.state.policy_size())

        sum_visits = max(curr.N_num_visits - 1, 1)

        move_lst = []
        probab = []
        for move in curr.children:
            policy = curr.children[move].N_num_visits / sum_visits
            move_lst += [move]
            probab += [policy]
            array[action_to_index(move)] = policy

        if include_empty_spots:
            return array
        return move_lst, probab

    def feed_network(self, curr: TreeNode) -> tuple:
        policy, value = self.nnet.evaluate(curr.state.get_nn_input())

        return policy[0], value.item()
