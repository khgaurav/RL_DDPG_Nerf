DDPG folder contains the implementation of the DDPG algorithm. It has 2 files:
training_agent_panda.py: This file contains the implementation of the DDPG algorithm for the Panda robot. It uses the PandaEnv environment.
training_agent_mujoco.py: This file contains the implementation of the DDPG algorithm for the Metaworld Mujoco environment. It uses the MujocoEnv environment.
## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--test` | `False` | Mention .pth file path to load learned model |
| `--env` | `PandaReachJointsDense-v3` | Environment name |
| `--runs` | `1` | Number of training/testing runs |
| `--timesteps` | `10000` | Number of timesteps per run |
| `--save` | `False` | Save model checkpoints, plots |
| `--param_optimize` | `False` | Enable hyperparameter optimization |

## Usage
python training_agent_panda.py --param_optimize --save --runs 10 --env PandaPushJointsDense-v3 --timesteps 20000
