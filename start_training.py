import torch.multiprocessing as mp
from Trainer import Trainer
from connectfour import Connect4Game, Connect4WrapperModel


if __name__ == '__main__':
    # To prevent unix leakages (prevents Error initialising CUDA)
    mp.set_start_method('spawn')

    # Self-play until 25000 examples are generated (USE 8 CPU CORES), then train and produce new model. Evaluate new
    # model in a best of 100 with the old model (USE 6 CPU cores).

    trainer = Trainer(Connect4WrapperModel, Connect4Game, num_self_play=25000, num_train_iterations=200,
                      num_bot_battles=100, self_play_cpu=8, bot_battle_cpu=6)

    # Bot battle has less processes because it uses more VRAM as each process loads two models instead of one

    trainer.training_pipeline()
