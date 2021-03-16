#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


from deep_rl import *
import argparse, os
import envs
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
#from IPython import embed

import torch
torch.set_default_tensor_type(torch.FloatTensor)

# PPO
def ppo_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 10
    config.log_interval = 10000
    if config.alg != 'ppo':
        if config.norm_rew_b == 1:
            config.reward_bonus_normalizer = MeanStdNormalizer()
        else:
            config.reward_bonus_normalizer = RescaleNormalizer()
    
    if config.rnd:
        config.rnd_bonus = 1000.
        config.lr = 0.0001

    if config.system == 'gcr':
        config.log_dir = './log/'
    elif config.system == 'philly':
        config.log_dir = os.getenv('PT_OUTPUT_DIR') + '/'

    agent.logger = Logger(None, config.log_dir + '/tensorboard/')

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers, single_process=True, seed=config.seed, horizon = config.horizon, noise=config.noise)
    config.eval_env = Task(config.game, seed=config.seed, horizon=config.horizon, noise=config.noise)
    # set action_space seed
    config.eval_env.action_space.seed(config.seed)
    
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, config.lr)
#    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, FCBody(config.state_dim))


    if isinstance(config.task_fn().action_space, Box):
        config.network_fn = lambda: GaussianActorCriticNet(config.state_dim, config.action_dim, FCBody(config.state_dim))
    elif isinstance(config.task_fn().action_space, Discrete):
        config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, FCBody(config.state_dim))
    
    
    
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.optimization_epochs = 5
    config.gradient_clip = 5
    config.rollout_length = config.horizon
    config.mini_batch_size = 32 * 5
    config.ppo_ratio_clip = 0.2
    config.max_steps = 3e6*config.horizon
    run_steps(PPOAgent(config))

def ppo_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)


    if config.system == 'gcr':
        config.log_dir = './log/'
    elif config.system == 'philly':
        config.log_dir = os.getenv('PT_OUTPUT_DIR') + '/'

    agent.logger = Logger(None, config.log_dir + '/tensorboard/')
   
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    # set action_space seed
    config.eval_env.action_space.seed(config.seed)
    
    config.num_workers = 8
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.00025, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, NatureConvBody())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    config.rollout_length = 128
    config.optimization_epochs = 3
    config.mini_batch_size = 32 * 8
    config.ppo_ratio_clip = 0.1
    config.log_interval = 128 * 8
    config.max_steps = int(2e7)
    run_steps(PPOAgent(config))
    


def rpg_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)
    
    # running setting
    config.num_workers = 10
    config.log_interval = 100
    config.norm_rew_b = 0
    config.norm_rew = 0
    config.init_new_policy = 0 
    config.n_policy_loops = 10
    config.n_traj_per_loop = 50
    
    if config.norm_rew_b == 1:
        config.reward_bonus_normalizer = MeanStdNormalizer()
    else:
        config.reward_bonus_normalizer = RescaleNormalizer()

    if config.norm_rew == 1:
        config.reward_normalizer = MeanStdNormalizer()
    
    if config.system == 'gcr':
        config.log_dir = './log/'
    elif config.system == 'philly':
        config.log_dir = os.getenv('PT_OUTPUT_DIR') + '/'

    
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers, single_process=True, seed=config.seed, horizon = config.horizon, noise=config.noise)
    config.eval_env = Task(config.game, seed=config.seed, horizon=config.horizon, noise=config.noise)
    # set action_space seed
    config.eval_env.action_space.seed(config.seed)
    
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, config.lr, weight_decay = config.weight_decay)
    lr2 = config.lr if config.lr2 == -1.0 else config.lr2
    config.optimizer_fn2 = lambda params: torch.optim.RMSprop(params, lr2, weight_decay = config.weight_decay)
    config.eval_episodes = 100 #eval over 100 trajectories

    # construct observations from states
    # 1. product; 2. inverse; 3. noise
    config.noise_dim = 0
    config.obs_type = 0
    if config.obs_type == 2:
        config.noise_dim = 5
    config.noise_low = -0.1
    config.noise_high = 0.1
    config.obs_dim = config.state_dim + config.noise_dim
    
    if isinstance(config.task_fn().action_space, Box):
        #config.task_fn().action_space._np_seed 
        config.network_fn = lambda: GaussianActorCriticNet(config.obs_dim, config.action_dim, FCBody(config.obs_dim))
    elif isinstance(config.task_fn().action_space, Discrete):
        config.network_fn = lambda: CategoricalActorCriticNet(config.obs_dim, config.action_dim, FCBody(config.obs_dim))
    
    
    config.flag = 0
    config.print = 0
    config.save = 0 # set to 1 if plot visitations
    config.counter = 0
    config.plot = 0 # set to 1 if plot bonus functions
        
    # hyperparameters for width on Mountaincar
    if config.game == 'MountainCarContinuous-v0' and config.bonus == 'width':
        config.bonus_coeff = 0.005 # set 0 for test 'ZERO'
        config.width_max = 0
        config.width_gd_steps = 10
        config.width_batch_size = 32 * 5
        config.width_loss_lambda = 0.1
        config.width_loss_lambda1 = 0.01
        config.width_lr = 0.001 # =0.0015 for layer=6
        config.width_gradient_clip = 5
        config.retrain_interval = 3 # No. of NPG steps per epoch is retrain_interval * n_policy_loops
        config.width_loop = 1000
        config.query_size = 20000
        config.query_batch = 20 # =10 for layer=6
        config.copy = 0 # 0: initialize width_network once; 1: reinit every width_train; 2: copy 'explore' network
        config.online = 0
        config.beta = -1
    
    # hyperparameters for PCPG on Mountaincar
    if config.game == 'MountainCarContinuous-v0' and config.bonus == 'rbf-kernel':
        config.bonus_coeff = 0.01
        config.retrain_interval = 3
        config.beta = -1
        
    # hyperparameters for PPO
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    config.rollout_length = config.horizon
    config.mini_batch_size = 32 * 5
    config.optimization_epochs = 5
    config.ppo_ratio_clip = 0.2
    config.max_steps = 10e9*config.horizon
    config.start_exploit = config.horizon if 'combolock' in config.game else 0
    if 'combolock' in config.game:
        config.start_exploit = config.horizon 
        config.max_epochs = 3*config.horizon
        config.rmax = 5.0
        config.n_rollouts_for_density_est = 50
    else:
        config.start_exploit = 3
        config.max_epochs = 50
        if config.game == 'MountainCarContinuous-v0':
            config.rmax = 100 # hardcoding for now
        config.n_rollouts_for_density_est = 10
        #config.state_normalizer = TwoRescaleNormalizer(1., 10.)
        
    config.ridge = 0.01
    if config.alg == 'ppo-rpg':
        run_steps(RPGAgent(config))
    elif config.alg == 'ppo-rpg2':
        run_steps(RPG2Agent(config))



if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    select_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=int, default=0)
    parser.add_argument('-alg', type=str, default='ppo-rpg')
    parser.add_argument('-env', type=str, default='diabcombolockhallway')
    parser.add_argument('-horizon', type=int, default=100)
    parser.add_argument('-noise', type=str, default='bernoulli')
    parser.add_argument('-eps', type=float, default=0.05)
    parser.add_argument('-lr', type=float, default=0.0005)
    parser.add_argument('-lr2', type=float, default=-1.0)
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-rnd_l2', type=float, default=0.0)
    parser.add_argument('-proll', type=float, default=0.8)
    parser.add_argument('-rnd_bneck', type=int, default=4)
    parser.add_argument('-bonus_coeff', type=float, default=0.01)
    parser.add_argument('-bonus_choice', type=int, default=1)  # 1 is max(bonus, rewards); 2 is rewards += bonus
    parser.add_argument('-bonus_select', type=int, default=2)  # 1 is permutation; 2 is uniform sampling; 3 is sequential 
    parser.add_argument('-bonus', type=str, default=None)
    parser.add_argument('-w_q', type=float, default = 0.85)
    parser.add_argument('-layer', type=int, default=6)
    parser.add_argument('-n_policy_loops', type=int, default=10)
    parser.add_argument('-n_traj_per_loop', type=int, default=50)
    parser.add_argument('-init_new_policy', type=int, default=0)
    parser.add_argument('-norm_rew', type=int, default=0)
    parser.add_argument('-norm_rew_b', type=int, default=0)
    parser.add_argument('-phi_dim', type=int, default=64)
    parser.add_argument('-beta', type=float, default=-1)
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-obs_type', type=int, default=0)
    parser.add_argument('-retrain_interval', type=int, default=3)
    parser.add_argument('-system', type=str, default='gcr')
    parser.add_argument('-save', type=int, default=0)
    config = parser.parse_args()
    select_device(config.device)
    random_seed(config.seed)
    
    if config.alg == 'ppo':
        if config.env == 'MontezumaRevengeNoFrameskip-v4':
            ppo_pixel(game=config.env,
                        lr=config.lr,
                        seed=config.seed,
                        rnd = 0,
                        alg='ppo',
                        system = config.system)
        else:
            ppo_feature(game=config.env,
                        lr=config.lr,
                        horizon=config.horizon,
                        noise = config.noise, 
                        seed=config.seed,
                        eps = config.eps,
                        rnd = 0,
                        alg='ppo',
                        system = config.system)        
    elif config.alg == 'ppo-rnd':
        ppo_feature(game=config.env,
                    lr=config.lr,
                    horizon=config.horizon,
                    noise = config.noise, 
                    seed=config.seed,
                    rnd = 1,
                    rnd_l2 = config.rnd_l2,
                    rnd_bneck = config.rnd_bneck,
                    eps = config.eps,
                    phi_dim = config.phi_dim, 
                    rnd_bonus = config.bonus_coeff,
                    alg='ppo-rnd',
                    norm_rew=config.norm_rew,
                    norm_rew_b=config.norm_rew_b,
                    system = config.system)
    elif config.alg in ['ppo-rpg', 'ppo-rpg2']:
        rpg_feature(game=config.env,
                    lr=config.lr,
                    lr2=config.lr2,
                    horizon=config.horizon,
                    noise = config.noise, 
                    seed=config.seed,
                    eps = config.eps,
                    proll = config.proll,
                    bonus = config.bonus, 
                    bonus_coeff = config.bonus_coeff,
                    bonus_choice = config.bonus_choice,
                    bonus_select = config.bonus_select,
                    beta = config.beta,
                    w_q = config.w_q,
                    phi_dim = config.phi_dim, 
                    alg=config.alg,
                    system = config.system,
                    layer = config.layer,
                    weight_decay = config.weight_decay)
