from __future__ import annotations
from interfaces import State, Nnet
import numpy as np
import concurrent.futures
import time
import pickle

ALPHA = 0.03
EPSILON = 0.1
C_PUCT = 1.5
# Cutoff at which we choose deterministically instead of sampling from multinomial distribution
TURN_CUTOFF = 10

# Virtual loss for parallel simulations -> reduce Q + U so that simulations traverse different paths
VIRTUAL_LOSS = 3
NUM_PARALLEL_SIMS = 2
NUM_SIMULATIONS = 50


class TreeNode:
    # Maps moves/actions to states
    children: dict
    state: State

    def __init__(self, state: State):
        self.state = state
        self.W_state_val = np.zeros(state.policy_size())
        self.N_num_visits = np.zeros(state.policy_size())
        self.P_init_policy = None

    def get_Q(self):
        """
        Returns total value / num_visits.
        :return: W/N or expected win at the current state.
        """
        return self.W_state_val / np.maximum(self.N_num_visits, np.ones(self.state.policy_size()))

    def serialise(self, reward):
        return pickle.dumps([self.state.get_nn_input(), self.get_improved_policy(), reward])

    def get_improved_policy(self) -> np.ndarray:
        array = self.N_num_visits / max(float(np.sum(self.N_num_visits)), 1.0)
        return array

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other: TreeNode):
        return hash(self) == hash(other)


class MonteCarloTS:
    curr: TreeNode
    root: TreeNode

    def __init__(self, initial_state, neural_net: Nnet, cpuct=C_PUCT, noise_epsilon=EPSILON, turn_cutoff=TURN_CUTOFF):
        self.root = TreeNode(initial_state)
        self.curr = self.root
        self.expanding = set()
        self.visited = {}
        self.nnet = neural_net
        self.noise_eps = noise_epsilon
        self.cpuct = cpuct
        self.turn_cutoff = turn_cutoff

    def get_best_action(self, node: TreeNode, turns: int):
        prob = node.get_improved_policy()
        legal_moves = node.state.get_actions()
        prob = prob * legal_moves

        if node.state.is_end() is not None:
            return None

        if turns >= self.turn_cutoff:
            action_idx = np.argmax(prob)
            assert legal_moves[int(action_idx)] == 1

            return self.nnet.action_to_index(action_idx)
        else:
            action_idx = np.random.multinomial(1, prob)
            action_idx = np.argmax(action_idx)
            assert legal_moves[int(action_idx)] == 1

            best = self.nnet.action_to_index(action_idx)
            return best

    def search(self, turns: int):

        key = hash(self.curr)
        if key in self.visited:
            self.curr = self.visited[key]

        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PARALLEL_SIMS) as executor:
            for _ in range(NUM_SIMULATIONS):
                executor.submit(self._simulation, curr=self.curr, is_root=True)

        best = self.get_best_action(self.curr, turns)
        self.curr = TreeNode(self.curr.state.transition_state(best))

        return best

    def enemy_move(self, move):
        self.curr = TreeNode(self.curr.state.transition_state(move))

    def _simulation(self, curr: TreeNode, is_root: bool = False):
        reward = curr.state.is_end()
        if reward is not None:
            return -1 * reward

        key = hash(curr)
        while key in self.expanding:
            time.sleep(0.000001)

        if key not in self.visited and key not in self.expanding:
            self.expanding.add(key)
            curr.P_init_policy, value = self.feed_network(curr)
            self.visited[key] = curr
            self.expanding.remove(key)
            return -1 * value
        else:
            curr = self.visited[key]

            sum_visits = max(1.0, float(np.sum(curr.N_num_visits)))

            legal_moves = curr.state.get_actions()

            if is_root:
                multiplier = self.cpuct * ((1 - self.noise_eps) * curr.P_init_policy +
                                           self.noise_eps * np.random.dirichlet([ALPHA] * len(legal_moves)))
            else:
                multiplier = self.cpuct * curr.P_init_policy

            q_plus_u = curr.get_Q() + multiplier * np.sqrt(sum_visits)/(1 + curr.N_num_visits)

            # Only invalid values are 0, all others are positive
            action_index = np.argmax((q_plus_u + 1000000) * legal_moves)
            action = self.nnet.index_to_action(action_index)

            selected_child = TreeNode(curr.state.transition_state(action))
            curr.N_num_visits[action] += VIRTUAL_LOSS
            curr.W_state_val[action] -= VIRTUAL_LOSS

            value = self._simulation(selected_child)

            curr.N_num_visits[action] += 1 - VIRTUAL_LOSS
            curr.W_state_val[action] += value + VIRTUAL_LOSS
            return -value

    def get_policy(self, node: TreeNode, action) -> float:
        return node.P_init_policy[self.nnet.action_to_index(action)]

    def feed_network(self, curr: TreeNode) -> tuple:
        policy, value = self.nnet.evaluate(curr.state.get_nn_input())
        policy = np.array(policy)

        return policy, value
