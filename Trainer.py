import torch.multiprocessing as mp
from tqdm import tqdm
from MCTS import MonteCarloTS
from os import mkdir, path

SELF_GAMES = 80
NUM_TRAINS = 100
BOT_GAMES = 20

CPU_COUNT = mp.cpu_count() - 1

BEST_PATH = "./models/vbest"
CONTENDER_PATH = "./models/vcontender"


class Trainer:
    def __init__(self, net_model, game_class, num_self_play: int = SELF_GAMES, num_train_iterations: int = NUM_TRAINS,
                 num_bot_battles: int = BOT_GAMES, self_play_cpu: int = CPU_COUNT, bot_battle_cpu: int = CPU_COUNT):
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
        if not path.exists("./models"):
            mkdir("./models")

        if mode == "continue":
            try:
                best_model.load_weights(BEST_PATH)
            except FileNotFoundError:
                print("file not found, mode is new")
        best_model.save_weights(BEST_PATH)

        for _ in range(NUM_TRAINS):
            inputs, improved_policy, win_loss = self.self_play(_)

            contender = self.net_model()
            contender.load_weights(BEST_PATH)

            contender.train_model(inputs, win_loss, improved_policy)
            contender.save_weights(CONTENDER_PATH)

            contender_wins, best_wins = self.bot_fight(_)

            win_ratio = contender_wins / self.num_bot_battles
            if win_ratio >= 0.55:
                best_model = contender

            print(f'Training iter {_}: new model won {contender_wins}')
            best_model.save_weights(BEST_PATH)

    def self_play(self, iteration):
        inputs, policy, value = [], [], []

        pool = mp.Pool(self.self_play_cpu)
        results_objs = [pool.apply_async(self.async_episode) for _ in range(self.num_self_play)]
        pool.close()

        p_bar = tqdm(range(len(results_objs)), desc=f"Self-play-{iteration}")

        for i in p_bar:
            result = results_objs[i]
            result = result.get()
            inputs += result[0]
            value += result[1]
            policy += result[2]
        return inputs, policy, value

    def async_episode(self) -> tuple:
        inputs, policy, value = [], [], []

        best_model = self.net_model()
        best_model.load_weights(BEST_PATH)

        game = self.game_class()
        mcts = MonteCarloTS(game.state(), best_model)

        visited_nodes = []
        while not game.is_over():
            visited_nodes.append(mcts.curr)
            move = mcts.search()
            game.take_action(move)

        turn_multiplier = 1
        for node in visited_nodes:
            new_policy = mcts.get_improved_policy(node, include_empty_spots=True)
            z = game.result()
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

        mcts_best = MonteCarloTS(game.state(), best_model)
        mcts_new = MonteCarloTS(game.state(), new_model)

        if iteration % 2 == 0:
            turns = {"best": "p1",
                     "new": "p2"}
        else:
            turns = {"best": "p2",
                     "new": "p1"}

        while not game.is_over():
            if turns["best"] == "p1":
                move = mcts_best.search(training=True)
                game.take_action(move)
                mcts_new.enemy_move(move)

                move = mcts_new.search(training=True)
                if move is None:
                    break
                game.take_action(move)
                mcts_best.enemy_move(move)
            else:
                move = mcts_new.search(training=True)
                game.take_action(move)
                mcts_best.enemy_move(move)

                move = mcts_best.search(training=True)
                if move is None:
                    break
                game.take_action(move)
                mcts_new.enemy_move(move)

        z = game.result()

        if z > 0 and turns["new"] == "p1":
            new_model_wins += 1
        elif z < 0 and turns["new"] == "p2":
            new_model_wins += 1
        elif z > 0 and turns["best"] == "p1":
            best_model_wins += 1
        elif z < 0 and turns["best"] == "p2":
            best_model_wins += 1

        return new_model_wins, best_model_wins
