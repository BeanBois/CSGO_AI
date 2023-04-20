from training_alpha import train

import CSGO_ENV
import gym

# env = gym.make('CSGO_ENV/CSGO_DUST2-v0')

# train(env)

env_alpha = gym.make('CSGO_ENV/CSGO_DUST2-v1')
train(env_alpha)