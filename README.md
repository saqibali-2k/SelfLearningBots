# SelfLearningBots

A library for creating your own self-learning AI based off Deepmind's Alpha-Zero. Uses multiprocessing to speed up training. 

Note: If u decide to implement your own Neural Net model, be wary if compatibility issues with multiprocessing. Both Tensorflow and Pytorch worked fine on windows, but I had some trouble with tensorflow on a google cloud instance running Ubuntu. The provided model is an adjustable ResNet implemented in Pytorch.

## Setup Environment ##
Python version: 3.7.8
1. Clone Repo
2. Create and activate virtual environment.
3. Install packages: `pip install -r requirments.txt`

    If pytorch does not install correctly, use:
   `pip install torch, torchvision`
4. To use graphics card, setup CUDA and CUDnn

    Tutorial on setting up CUDA on unix through terminal: 
https://towardsdatascience.com/installing-cuda-on-google-cloud-platform-in-10-minutes-9525d874c8c1
## Setup Game ## 
**_For examples, see branch: Connect4-example._**

You are required to implement interfaces State, Game and create a subclass of DefaultModel. You can also implement your own Neural net model, but a default one is provided.

## Run the Trainer ##
Create a Trainer instance and choose appropriate arguments. From start_training.py in Connect4-example:

```
if __name__ == '__main__':
    # To prevent unix leakages (prevents Error initialising CUDA)
    multiprocessing.set_start_method('spawn')

    # Self-play until 25000 examples are generated (USE 8 CPU CORES), then train and produce new model. Evaluate new
    # model in a best of 100 with the old model (USE 6 CPU cores).

    trainer = Trainer(Connect4WrapperModel, Connect4Game, num_self_play=25000, num_train_iterations=200,
                      num_bot_battles=100, self_play_cpu=8, bot_battle_cpu=6)

    # Bot battle has less processes because it uses more VRAM as each process loads two models instead of one

    trainer.training_pipeline()
```

## TODOs and Update Logs ##
| Date         | Update       |
|--------------|--------------|
| Aug. 15 2020 | Started Logs |

_TODOs_:
* Asynchronous MCTS
* Asynchronize the whole training process
