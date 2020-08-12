import torch.multiprocessing as mp
from Trainer import Trainer
from connectfour import Connect4Game, Connect4Model


if __name__ == '__main__':
    # To prevent unix leakages (prevents Error initialising CUDA)
    mp.set_start_method('spawn')
    trainer = Trainer(Connect4Model, Connect4Game, num_self_play=25000, num_train_iterations=200, num_bot_battles=100,
                      self_play_cpu=8, bot_battle_cpu=6)
    trainer.training_pipeline()
