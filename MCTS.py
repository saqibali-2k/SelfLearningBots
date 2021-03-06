from __future__ import annotations
from interfaces import State, Nnet
import numpy as np

ALPHA = 0.03
EPSILON = 0.1
C_PUCT = 1.5
TURN_CUTOFF = 10

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
        self.visited = {}
        self.nnet = neural_net
        self.noise_eps = noise_epsilon
        self.cpuct = cpuct
        self.turn_cutoff = turn_cutoff

    def get_best_action(self, node: TreeNode, turns: int):
        prob = self.get_improved_policy(node)
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

        for _ in range(NUM_SIMULATIONS):
            self._simulation(self.curr, is_root=True)

        best = self.get_best_action(self.curr, turns)
        print(self.curr.N_num_visits)
        print(self.curr.P_init_policy)
        self.curr = TreeNode(self.curr.state.transition_state(best))

        return best

    def enemy_move(self, move):
        self.curr = TreeNode(self.curr.state.transition_state(move))

    def _simulation(self, curr: TreeNode, is_root: bool = False):
        reward = curr.state.is_end()
        if reward is not None:
            return -1 * reward

        key = hash(curr)
        if key not in self.visited:
            curr.P_init_policy, value = self.feed_network(curr)
            self.visited[key] = curr
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

            value = self._simulation(selected_child)
            curr.N_num_visits[action_index] += 1
            curr.W_state_val[action_index] += value
            return -value

    def get_policy(self, node: TreeNode, action) -> float:
        return node.P_init_policy[self.nnet.action_to_index(action)]

    @staticmethod
    def get_improved_policy(curr: TreeNode) -> np.ndarray:
        array = curr.N_num_visits / max(float(np.sum(curr.N_num_visits)), 1.0)
        return array

    def feed_network(self, curr: TreeNode) -> tuple:
        policy, value = self.nnet.evaluate(curr.state.get_nn_input())
        policy = np.array(policy)

        return policy, value
