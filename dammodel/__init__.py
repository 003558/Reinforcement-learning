from gym.envs.registration import register

register(
    id='dam-v0',
    entry_point='dammodel.env:dam',
)
register(
    id='damtest-v0',
    entry_point='dammodel.env:damtest',
)
