from training import train
import CSGO_ENV
import gym

env = gym.make('CSGO_ENV/CSGO_DUST2-v0')

train(env)