from gym.envs.registration import register

register(
    id="contosocabs-v0",
    entry_point="contoso_cabs.envs:ContosoCabs_v0",
)
