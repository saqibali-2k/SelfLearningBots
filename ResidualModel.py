from interfaces import Nnet
from typing import Union, Tuple
import numpy as np
import torch
import tqdm

EPOCHS = 5
BATCH_SIZE = 512


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
    def __init__(self, input_size: Tuple[int, int, int], output_policy_size: int, num_res_blocks: int = 3):
        super(ValuePolicyNet, self).__init__()

        self.conv_layer_1 = torch.nn.Conv2d(input_size[0], 256, 3, padding=1)  # output: (256, in[1], in[2])
        self.batch_norm1 = torch.nn.BatchNorm2d(256)
        self._num_res_blocks = num_res_blocks

        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks += [torch.nn.Conv2d(256, 256, 3, padding=1)]
            res_blocks += [torch.nn.BatchNorm2d(256)]
            res_blocks += [torch.nn.Conv2d(256, 256, 3, padding=1)]
            res_blocks += [torch.nn.BatchNorm2d(256)]
            # input: (256, in[1], in[2]), output: (256, in[1], in[2])

        self.res_blocks = torch.nn.ModuleList(res_blocks)
        self.relu_activation = torch.nn.functional.relu

        self.fc_layer_policy = torch.nn.Linear(256 * input_size[1] * input_size[2], output_policy_size)

        self.fc_layer_value1 = torch.nn.Linear(256 * input_size[1] * input_size[2], 256)
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
        policy = torch.nn.functional.softmax(policy, dim=1)

        return value, policy


class DefaultModel(Nnet):
    def __init__(self, input_size: Tuple[int, int, int], policy_size: int, num_res_blocks: int):

        self.model = ValuePolicyNet(input_size=input_size, output_policy_size=policy_size,
                                    num_res_blocks=num_res_blocks)

        if torch.cuda.is_available():
            self.model.to(torch.device("cuda:0"))
        else:
            self.model.to(torch.device("cpu"))

    def load_weights(self, path: str):
        self.model.load_state_dict(torch.load(path))

    def save_weights(self, path: str):
        torch.save(self.model.state_dict(), path)

    def train_model(self, lr, inputs: list, expected_values: list, expected_policies: list):
        self.model.train()
        inputs = np.array(inputs)
        expected_values = np.array(expected_values)
        expected_policies = np.array(expected_policies)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        steps = 0
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
                policy_loss = -1 * torch.sum(policy_expected_batch * torch.log(policy_pred + 1e-7))
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
                steps += 1

        del policy_expected_batch, value_expected_batch
        del value_pred, policy_pred
        return steps

    def evaluate(self, inputs):
        self.model.eval()
        with torch.no_grad():
            value, policy = self.model(np.array([inputs]))
            # Return only the policy vector, and the value scalar
            return policy.cpu()[0], value.cpu().item()

    def action_to_index(self, action):
        raise NotImplementedError
