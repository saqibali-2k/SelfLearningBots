import torch.multiprocessing as mp
from tqdm import tqdm
from MCTSasync import MonteCarloTS
from os import makedirs
import pickle

WIN_RATIO_REQUIREMENT = 0.55

SELF_EXAMPLES = 30000
NUM_TRAINS = 100
BOT_GAMES = 20

MAX_EXAMPLES = 180000

CPU_COUNT = mp.cpu_count() - 1
BEST_GENERATIONS_PATH = "./models/best_v"
BEST_PATH = "./models/temp/vbest"
CONTENDER_PATH = "./models/temp/vcontender"
DATA_DIR = "./data/data_checkpoint"


class Trainer:
    def __init__(self, net_model, game_class, num_self_play: int = SELF_EXAMPLES,
                 num_train_iterations: int = NUM_TRAINS, num_bot_battles: int = BOT_GAMES,
                 self_play_cpu: int = CPU_COUNT, bot_battle_cpu: int = CPU_COUNT):
        """
        Create a Trainer instance.
        :param net_model: the neural network class, subclass of Nnet
        :param game_class: the game class, subclass of Game
        :param num_self_play: number of games that the bot will play against self
        :param num_train_iterations: number of iterations of the full training loop
        :param num_bot_battles: number of games that the bot will play against a possible better bot
        :param self_play_cpu: number of CPUs used during self_play
        :param bot_battle_cpu: number of CPUs used during bot_battle
        """
        self.net_model = net_model
        self.game_class = game_class
        self.num_self_play = num_self_play
        self.num_train_iterations = num_train_iterations
        self.num_bot_battles = num_bot_battles
        self.self_play_cpu = self_play_cpu
        self.bot_battle_cpu = bot_battle_cpu

    def training_pipeline(self):
        while True:
            mode = input("mode? (new/continue)")
            if mode == "continue" or mode == "new":
                break
            print("Not recognized")

        best_model = self.net_model()
        contender = self.net_model()
        total_steps = 0
        best_model_gen = 1
        inputs, improved_policy, win_loss = [], [], []

        makedirs("./models/temp", exist_ok=True)

        if mode == "continue":
            try:
                best_model.load_weights(BEST_PATH)
                contender.load_weights(CONTENDER_PATH)
                params, data = self._load_checkpoint()
                total_steps, best_model_gen = params
                inputs, improved_policy, win_loss = data
                total_steps, best_model_gen = int(total_steps), int(best_model_gen)
            except FileNotFoundError:
                print("file not found, mode is new")
        best_model.save_weights(BEST_PATH)

        for _ in range(self.num_train_iterations):
            examples = self.self_play(_)
            inputs += examples[0]
            improved_policy += examples[1]
            win_loss += examples[2]
            self.trim_to_length(inputs, improved_policy, win_loss)

            if total_steps < 500:
                lr = 1e-2
            elif total_steps < 2000:
                lr = 1e-3
            elif total_steps < 9000:
                lr = 1e-4
            else:
                lr = 2.5e-5

            total_steps += contender.train_model(lr, inputs, win_loss, improved_policy)
            contender.save_weights(CONTENDER_PATH)

            contender_wins, best_wins = self.bot_fight(_)

            win_ratio = contender_wins / self.num_bot_battles
            if win_ratio >= WIN_RATIO_REQUIREMENT:
                best_model.load_weights(CONTENDER_PATH)
                best_model.save_weights(BEST_GENERATIONS_PATH + str(best_model_gen))
                best_model_gen += 1

            print(f'Training iter {_}: new model won {contender_wins}, best model won {best_wins} '
                  f'(GEN: {best_model_gen - 1})')
            best_model.save_weights(BEST_PATH)
            self._save_checkpoint([total_steps, best_model_gen], [inputs, improved_policy, win_loss])

    def save_to_file(self, data):
        makedirs(DATA_DIR, exist_ok=True)

        serialised_data = pickle.dumps(data)
        size = len(serialised_data)
        file = open(DATA_DIR, 'wb')
        file.write(size.to_bytes(length=64, byteorder='big', signed=False))
        file.write(serialised_data)

    def load_from_file(self):
        file = open(DATA_DIR, 'rb')
        obj_size = int.from_bytes(file.read(64), byteorder='big', signed=False)
        data_serialised = file.read(obj_size)
        return pickle.loads(data_serialised)

    def _save_checkpoint(self, param, data):
        file = open("parameters_checkpoint.txt", 'w')
        for item in param:
            file.write(str(item) + ",")
        file.close()
        self.save_to_file(data)

    def _load_checkpoint(self) -> list:
        file = open("parameters_checkpoint.txt", 'r')
        items = file.readline()
        items = items.split(",")
        file.close()
        return items[:-1], self.load_from_file()

    def trim_to_length(self, inputs: list, policy: list, value: list):
        if len(inputs) < NUM_TRAINS:
            return
        for _ in range(len(inputs) - MAX_EXAMPLES):
            inputs.pop(0)
            policy.pop(0)
            value.pop(0)

    def self_play(self, iteration):
        inputs, policy, value = [], [], []
        total_examples = 0

        p_bar = tqdm(total=self.num_self_play, desc=f"Self-play-{iteration}")

        while total_examples < self.num_self_play:
            pool = mp.Pool(self.self_play_cpu)
            results_objs = [pool.apply_async(self.async_episode) for _ in range(16)]
            pool.close()

            for result in results_objs:
                result = result.get()
                inputs += result[0]
                value += result[1]
                policy += result[2]
                p_bar.update(len(result[0]))
                total_examples += len(result[0])
        p_bar.close()
        return inputs, policy, value

    def async_episode(self) -> tuple:
        inputs, policy, value = [], [], []

        best_model = self.net_model()
        best_model.load_weights(BEST_PATH)

        game = self.game_class()
        mcts = MonteCarloTS(game.state(), best_model)
        turn_count = 0

        visited_nodes = []
        while not game.is_over():
            visited_nodes.append(mcts.curr)
            move = mcts.search(turn_count)
            game.take_action(move)
            turn_count += 1

        turn_multiplier = -1
        for node in visited_nodes[::-1]:
            new_policy = node.get_improved_policy()
            z = game.get_reward()
            z *= turn_multiplier
            turn_multiplier *= -1
            inputs.append(node.state.get_nn_input())
            policy.append(new_policy)
            value.append(z)
        return inputs, value, policy

    def bot_fight(self, iteration) -> tuple:
        new_model_wins = 0
        best_model_wins = 0
        result_objs = []
        pool = mp.Pool(self.bot_battle_cpu)
        for j in range(self.num_bot_battles):
            result_objs += [pool.apply_async(self.async_arena, args=(j,))]
        pool.close()

        p_bar = tqdm(range(len(result_objs)), desc=f"Bot battle-{iteration}")

        for i in p_bar:
            result = result_objs[i]
            result = result.get()
            new_model_wins += result[0]
            best_model_wins += result[1]

        return new_model_wins, best_model_wins

    def async_arena(self, iteration):

        new_model_wins = 0
        best_model_wins = 0
        game = self.game_class()

        best_model = self.net_model()
        best_model.load_weights(BEST_PATH)
        new_model = self.net_model()
        new_model.load_weights(CONTENDER_PATH)

        mcts_best = MonteCarloTS(game.state(), best_model, cpuct=1.0, noise_epsilon=0, turn_cutoff=0)
        mcts_new = MonteCarloTS(game.state(), new_model, cpuct=1.0, noise_epsilon=0, turn_cutoff=0)

        if iteration % 2 == 0:
            turns = {"best": "p1",
                     "new": "p2"}
        else:
            turns = {"best": "p2",
                     "new": "p1"}
        turn_count = 0
        while not game.is_over():
            if turns["best"] == "p1":
                move = mcts_best.search(turn_count)
                game.take_action(move)
                mcts_new.enemy_move(move)
                turn_count += 1

                move = mcts_new.search(turn_count)
                if move is None:
                    break
                game.take_action(move)
                mcts_best.enemy_move(move)
                turn_count += 1
            else:
                move = mcts_new.search(turn_count)
                game.take_action(move)
                mcts_best.enemy_move(move)
                turn_count += 1

                move = mcts_best.search(turn_count)
                if move is None:
                    break
                game.take_action(move)
                mcts_new.enemy_move(move)
                turn_count += 1

        z = game.result()

        if z == "1-0" and turns["new"] == "p1":
            new_model_wins += 1
        elif z == "0-1" and turns["new"] == "p2":
            new_model_wins += 1
        elif z == "1-0" and turns["best"] == "p1":
            best_model_wins += 1
        elif z == "0-1" and turns["best"] == "p2":
            best_model_wins += 1

        return new_model_wins, best_model_wins
