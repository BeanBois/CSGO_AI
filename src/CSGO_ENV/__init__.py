from gym.envs.registration import register

register(
    id='CSGO_ENV/CSGO_DUST2-v0',
    entry_point='CSGO_ENV.csgo_env_charlie_delta:CSGO_Env',
    max_episode_steps=450,
)