from interfaces import *
from typing import Union
import numpy as np
from copy import deepcopy
import torch
import tqdm

EPOCHS = 20
BATCH_SIZE = 256


class Connect4Game(Game):
    def __init__(self):
        self.game_over = False
        self.reward = None
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

            for i in range(5, -1, -1):
                if self.p1_array[i, action] == 0 and self.p2_array[i, action] == 0:
                    arr[i, action] = 1
                    break

            # Check if four in a row occurred
            self.game_over = self._check_four(i, action, arr)
            if self.game_over:
                self.reward = 1

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

    def _follow_line(self, row: int, col: int, dir_x: int, dir_y: int, arr):
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
        four_in_row = self._follow_line(row, col, -1, 0, arr)
        four_in_row = self._follow_line(row, col, -1, -1, arr) or four_in_row
        four_in_row = self._follow_line(row, col, 0, -1, arr) or four_in_row
        four_in_row = self._follow_line(row, col, 1, -1, arr) or four_in_row
        four_in_row = self._follow_line(row, col, 1, 0, arr) or four_in_row
        four_in_row = self._follow_line(row, col, 1, 1, arr) or four_in_row
        four_in_row = self._follow_line(row, col, 0, 1, arr) or four_in_row
        four_in_row = self._follow_line(row, col, -1, 1, arr) or four_in_row

        return four_in_row

    def is_over(self) -> bool:
        return self.game_over

    def result(self) -> Optional[int]:
        return self.reward

    def get_actions(self):
        lst = []
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

    def get_representation(self) -> np.ndarray:
        return self.game.p1_array, self.game.p2_array

    def is_end(self) -> Optional[int]:
        if self.game.is_over():
            return self.game.result()

    def get_nn_input(self):
        p1_board, p2_board = self.get_representation()
        turn_board = np.zeros((6, 7))
        turn_board[:, :] = self.game.turn
        return np.array([p1_board, p2_board, turn_board])

    def policy_size(self) -> int:
        return 7


class StatWrapper:

    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, x):
        self.total += x
        self.count += 1

    def __repr__(self):
        avg = self.total / self.count
        return f'{avg: .2f}'


class ValuePolicyNet(torch.nn.Module):
    def __init__(self, num_res_blocks: int=3):
        super(ValuePolicyNet, self).__init__()

        self.conv_layer_1 = torch.nn.Conv2d(3, 256, 3, padding=1)  # input size (3, 6, 7), output: (256, 6, 7)
        self.batch_norm1 = torch.nn.BatchNorm2d(256)
        self._num_res_blocks = num_res_blocks

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks += [torch.nn.Conv2d(256, 256, 3, padding=1)]
            res_blocks += [torch.nn.BatchNorm2d(256)]
            res_blocks += [torch.nn.Conv2d(256, 256, 3, padding=1)]
            res_blocks += [torch.nn.BatchNorm2d(256)]
            # input: (256, 6, 7), output: (256, 6, 7)

        self.res_blocks = torch.nn.ModuleList(res_blocks)
        self.relu_activation = torch.nn.functional.relu

        self.fc_layer_policy = torch.nn.Linear(10752, 7)

        self.fc_layer_value1 = torch.nn.Linear(10752, 256)
        self.fc_layer_value2 = torch.nn.Linear(256, 1)

    def forward(self, states: Union[torch.tensor, np.ndarray]):

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        reshaped_states = torch.tensor(states, dtype=torch.float32, device=device)

        block1 = self.conv_layer_1(reshaped_states)
        block1 = self.batch_norm1(block1)
        block1 = self.relu_activation(block1)

        res_in = block1
        for i in range(self._num_res_blocks):
            res_out = self.res_blocks[i * 4 + 0](res_in)  # Conv
            res_out = self.res_blocks[i * 4 + 1](res_out)  # BatchNorm
            res_out = self.relu_activation(res_out)

            res_out = self.res_blocks[i * 4 + 2](res_out)  # Conv
            res_out = self.res_blocks[i * 4 + 3](res_out)  # BatchNorm
            res_out = torch.add(res_out, res_in)
            res_out = self.relu_activation(res_out)
            res_in = res_out
        res_out = res_in

        # Value head
        value = self.fc_layer_value1(torch.reshape(res_out, (res_out.shape[0], -1)))
        value = self.relu_activation(value)
        value = self.fc_layer_value2(value)
        value = torch.tanh(value)

        # Policy head
        policy = self.fc_layer_policy(torch.reshape(res_out, (res_out.shape[0], -1)))
        policy = torch.nn.functional.log_softmax(policy, dim=1)

        return value, policy


class Connect4Model(Nnet):
    def __init__(self):

        self.model = ValuePolicyNet(3)

        if torch.cuda.is_available():
            self.model.to(torch.device("cuda:0"))
        else:
            self.model.to(torch.device("cpu"))

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def save_weights(self, path: str):
        torch.save(self.model.state_dict(), path)

    def train_model(self, inputs: list, expected_values: list, expected_policies: list):
        self.model.train()
        inputs = np.array(inputs)
        expected_values = np.array(expected_values)
        expected_policies = np.array(expected_policies)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-4)

        for i in range(EPOCHS):

            num_batches = expected_values.shape[0] // min(BATCH_SIZE, expected_values.shape[0])
            p_bar = tqdm.tqdm(range(num_batches), desc=f"Training Network Epoch-{i}")

            rand_indices = torch.randperm(expected_values.shape[0])
            expected_values = expected_values[rand_indices]
            expected_policies = expected_policies[rand_indices]
            inputs = inputs[rand_indices]

            # Stats to report
            avg_loss = StatWrapper()
            avg_policy_loss = StatWrapper()
            avg_value_loss = StatWrapper()

            for _ in p_bar:
                # Make batch
                end_index = min((_ + 1) * BATCH_SIZE, expected_values.shape[0])

                value_expected_batch = expected_values[_ * BATCH_SIZE: end_index]
                policy_expected_batch = expected_policies[_ * BATCH_SIZE: end_index, :]
                states_batch = inputs[_ * BATCH_SIZE: end_index, :, :]

                policy_expected_batch = torch.tensor(policy_expected_batch, dtype=torch.float32, device=device)
                value_expected_batch = torch.tensor(value_expected_batch, dtype=torch.float32, device=device)

                # Get outputs
                value_pred, policy_pred = self.model(states_batch)
                value_pred = torch.flatten(value_pred)

                # Calculate losses
                policy_loss = -1 * policy_expected_batch.mm(policy_pred.to(device).transpose(0, 1)).sum()
                value_loss = (value_pred.to(device) - value_expected_batch).pow(2).sum()

                loss = policy_loss + value_loss

                # Update stats
                avg_loss.update(loss.item())
                avg_value_loss.update(value_loss.item())
                avg_policy_loss.update(policy_loss.item())
                p_bar.set_postfix(loss=avg_loss, policy_loss=avg_policy_loss, value_loss=avg_value_loss)

                # Perform back prop and update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        del policy_expected_batch, value_expected_batch
        del value_pred, policy_pred

    def evaluate(self, inputs):
        self.model.eval()
        with torch.no_grad():
            value, policy = self.model(np.array([inputs]))
            # Return only the policy vector, and the value scalar
            return policy.cpu()[0], value.cpu().item()

    def action_to_index(self, action):
        # Actions are numbers from 0 - 6, which corresponds directly to an index
        return action



