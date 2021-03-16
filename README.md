This is the code for paper: 

Provably Correct Optimization and Exploration with Non-linear Policies (Fei Feng, Wotao Yin, Alekh Agarwal, Lin F. Yang)

Find main function and hyperparameters at: code/DeepRL/run.py

Change network structure at: code/DeepRL/deep_rl/network/network_bodies.py

Find RL-agent implementations at: code/DeepRL/deep_rl/agent/, where RPG_agent.py is used for ENIAC and PC-PG; PPO_agent.py is used for PPO and PPO-RND.

Run mountaincar with ENIAC:
python3 run.py -alg ppo-rpg -env MountainCarContinuous-v0 -seed 0 -bonus width

Run mountaincar with PCPG:
python3 run.py -alg ppo-rpg -env MountainCarContinuous-v0 -seed 0 -bonus rbf-kernel

Run mountaincar with PPO:
python3 run.py -alg ppo -env MountainCarContinuous-v0 -seed 0

Run mountaincar with PPO-RND
python3 run.py -alg ppo-rnd -env MountainCarContinuous-v0 -seed 0



