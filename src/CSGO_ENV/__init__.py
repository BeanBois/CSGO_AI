from gym.envs.registration import register

register(
    id='CSGO_ENV/CSGO_DUST2-v0',
    entry_point='CSGO_ENV.csgo_env:CSGO_Env',
    max_episode_steps=450,
)

register(
    id='CSGO_ENV/CSGO_DUST2-v1',
    entry_point='CSGO_ENV.csgo_env_alpha:CSGO_Env_Alpha',
    max_episode_steps=450,
)
