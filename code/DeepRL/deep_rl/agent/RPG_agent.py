#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *

import math, random, pdb, copy, sys
import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import scipy
from sklearn.kernel_approximation import RBFSampler
from IPython import embed
from time import time
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

    

class RPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.task.action_space.seed(config.seed) # set action random seed manually for reproduction purpose
        self.network, self.optimizer, self.replay_buffer, self.density_model = dict(), dict(), dict(), dict()
        self.replay_buffer_actions = dict()
        self.replay_buffer_infos = dict()
        for mode in ['explore', 'exploit', 'rollin']:
            self.network[mode] = config.network_fn()
            self.replay_buffer[mode] = []
            self.replay_buffer_actions[mode] = []
            self.replay_buffer_infos[mode] = []
            
        self.optimizer['explore'] = config.optimizer_fn(self.network['explore'].parameters())
        self.optimizer['exploit'] = config.optimizer_fn2(self.network['exploit'].parameters())
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.policy_mixture = [copy.deepcopy(self.network['explore'].state_dict())]
        self.policy_mixture_optimizers = [copy.deepcopy(self.optimizer['explore'].state_dict())]
        self.policy_mixture_weights = torch.tensor([1.0])
        self.policy_mixture_returns = []
        self.timestamp = None
        self.bonus_print = []
        self.query_pool = []
        self.query_counter = 0

        if self.config.bonus == 'rnd':
            self.rnd_network = FCBody(self.config.obs_dim).cpu()
            self.rnd_pred_network = FCBody(self.config.obs_dim).cpu()
            self.rnd_optimizer = torch.optim.RMSprop(self.rnd_pred_network.parameters(), 0.001)
        elif self.config.bonus == 'randnet-kernel-s':
            self.kernel = FCBody(self.config.obs_dim, hidden_units=(self.config.phi_dim, self.config.phi_dim)).cpu()
        elif self.config.bonus == 'randnet-kernel-sa':
            self.kernel = FCBody(self.config.obs_dim + self.config.action_dim, hidden_units=(self.config.phi_dim, self.config.phi_dim)).cpu()
        elif self.config.bonus == 'rbf-kernel':
            self.rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=self.config.phi_dim)
            if isinstance(self.task.action_space, Box):
                self.rbf_feature.fit(X = np.random.randn(5, self.config.obs_dim + self.config.action_dim))
            else:
                self.rbf_feature.fit(X = np.random.randn(5, self.config.obs_dim + 1))
        elif self.config.bonus == 'width':
            
            # initialize the trained one and copy the fixed one
            #self.width_network = copy.deepcopy(self.network['explore']).to(Config.DEVICE)
            self.width_network = self.config.network_fn().to(Config.DEVICE)
            self.width_network_fixed = copy.deepcopy(self.width_network).to(Config.DEVICE)
            
        if isinstance(self.task.action_space, Box):
            self.uniform_prob = self.continous_uniform_prob()
        else:
            self.uniform_prob = 1./self.config.action_dim

    # takes as input a minibatch of states, returns exploration reward
    def compute_reward_bonus(self, obs, actions = None):
        if self.config.bonus == 'rnd':
            pbs = torch.from_numpy(obs).float().cpu()
            rnd_target = self.rnd_network(obs).detach()
            rnd_pred = self.rnd_pred_network(obs).detach()
            rnd_loss = F.mse_loss(rnd_pred, rnd_target, reduction='none').mean(1)
            reward_bonus = rnd_loss.cpu().numpy()
        
        # width on critic network
        elif self.config.bonus == 'width':
            reward_bonus = self.get_critic_width('explore', obs).detach()
            
        elif 'randnet-kernel' in self.config.bonus:
            phi = self.compute_kernel(tensor(obs), actions)
            reward_bonus = torch.sqrt((torch.mm(phi, self.density_model) * phi).sum(1)).detach()
            
        elif 'rbf-kernel' in self.config.bonus:
            assert actions is not None
            phi = self.compute_kernel(tensor(obs), tensor(actions))
            reward_bonus = torch.sqrt((torch.mm(phi, self.density_model) * phi).sum(1)).detach()
            
        elif 'id-kernel' in self.config.bonus:
            phi = self.compute_kernel(tensor(obs), actions)
            reward_bonus = torch.sqrt((torch.mm(phi, self.density_model) * phi).sum(1)).detach()

            
        elif 'counts' in self.config.bonus:
            reward_bonus = []
            for s in self.config.state_normalizer(obs):
                s = tuple(s)
                if not s in self.density_model['explore'].keys():
                    cnts = 0
                else:
                    cnts = self.density_model['explore'][s]
                if self.config.bonus == 'counts':
                    reward_bonus.append(1.0/(1.0 + cnts))
                elif self.config.bonus == 'counts-sqrt':
                    reward_bonus.append(1.0/math.sqrt(1.0 + cnts))
                    
            reward_bonus = np.array(reward_bonus)
        
        # use threshold
        if self.config.beta > 0: 
            reward_bonus = np.array([1 if r > self.config.beta else 0 for r in reward_bonus])
        # directly use bonus value
        else: 
            reward_bonus = np.array([r for r in reward_bonus])
        return reward_bonus

    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))

    def width_loss(self, size, output, target):
        
        # let |target - output| be as big as possible
        # lam1 is set nonzero to avoid local optimal
        
        lam = self.config.width_loss_lambda
        lam1 = self.config.width_loss_lambda1
        loss = torch.sum((output[0:size] - target[0:size])**2) / size - lam1 * torch.sum(target[size:] - output[size:])/(len(target)-size)\
               - lam * torch.sum((target[size:] - output[size:])**2)/(len(target)-size)
        return loss

    def retrain_width(self, mode):
        print("retrain width")
        width_max = self.config.width_max
        query_batch = self.config.query_batch
        replay_buffer = torch.cat(sum(self.replay_buffer[mode], [])).cpu().numpy()
        query_pool = self.query_pool
        gd_steps = self.config.width_gd_steps
        width_batch_size = self.config.width_batch_size
        
        
        # reinitialize width_network: 0. void reinit; 1. random reinit; 2. copy current 'explore' network
        if self.config.copy == 1:
            self.width_network = self.config.network_fn().to(Config.DEVICE)
        elif self.config.copy == 2:
            self.width_network = copy.deepcopy(self.network[mode]).to(Config.DEVICE)
            
        
        # reset width_network_fixed: can be randomly initialized or directly copy width_network
        
        #del self.width_network_fixed
        #torch.cuda.empty_cache()
        self.width_network_fixed = copy.deepcopy(self.width_network).to(Config.DEVICE)
        self.width_optimizer = torch.optim.RMSprop(self.width_network.parameters(), lr=self.config.width_lr)
        self.width_network_fixed.train(mode=False)
        
        
        # several query selection methods: 1: permutation; 2: uniform sampling; 3. sequential
        if self.config.bonus_select == 1:
            self.query_pool = np.random.permutation(self.query_pool)
        
        for _ in range(self.config.width_loop):
            
            # sample query
            if self.config.bonus_select == 2:
                in_obs = self.query_pool[np.random.choice(query_pool.shape[0], query_batch, replace=False), :].tolist()
            else:
                in_obs = self.query_pool[_*query_batch: (_+1)*query_batch].tolist()
            
            # plot the bonus map
            if not _ % 1000 and self.config.plot:
                obs = self._to_variable(query_pool.reshape(-1, self.config.obs_dim)).to(Config.DEVICE)
                width = self.width_network(obs)['v']-self.width_network_fixed(obs)['v']
                width = torch.clamp(width, 0, width_max).detach()
                width = np.array([w for w in width])
                print(str(np.count_nonzero(width)) + ' of '+str(len(width))+ 'is nonzero')
        
                # plot width bonus
                width = width.reshape((len(width), 1))
                self.bonus_print = np.copy(query_pool)
                self.bonus_print = np.append(self.bonus_print, width, axis=1)
                self.plot_bonus(mode=_)
            
            
            # optimize over the sampled query  
            for i in range(gd_steps):
                minibatch = replay_buffer[np.random.choice(replay_buffer.shape[0], width_batch_size, replace=False), :].tolist()
                obs = np.vstack(minibatch + in_obs)
                obs = self._to_variable(obs.reshape(-1, self.config.obs_dim)).to(Config.DEVICE)
                self.width_network.train(mode=False)
                v_new = self.width_network(obs)['v'].to(Config.DEVICE)
                v_old = self.width_network_fixed(obs)['v'].to(Config.DEVICE)
                loss = self.width_loss(width_batch_size, v_new, v_old)
                self.width_network.train(mode=True)
                self.width_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.width_network.parameters(), self.config.width_gradient_clip)
                self.width_optimizer.step()
                
        self.width_network.train(mode=False)
        
        # Below: plot bonus map and reset width_max
        obs = self._to_variable(query_pool.reshape(-1, self.config.obs_dim)).to(Config.DEVICE)
        visited = replay_buffer.tolist()
        
        # obtain width as |target - output|
        width = torch.abs(self.width_network_fixed(obs)['v']-self.width_network(obs)['v'])
        width = np.array([w for w in width])
        width_visited = torch.abs(self.width_network_fixed(visited)['v']-self.width_network(visited)['v'])
        width_visited  = np.array([w for w in width_visited]) 
        
        # rescale width
        self.config.width_max = np.amax(width).tolist()[0]
        if self.config.width_max > 0 :
            width /= self.config.width_max
            width *= self.config.bonus_coeff * self.config.horizon
            width_visited /= self.config.width_max
            width_visited *= self.config.bonus_coeff * self.config.horizon
            
        # thresholding width if needed
        if self.config.beta > 0:
            self.config.beta = np.quantile(width, self.config.w_q)
            width = np.array([1 if r > self.config.beta else 0 for r in width])
            width_visited = np.array([1 if r > self.config.beta else 0 for r in width_visited])
            
        # plot bonus map
		if self.config.plot:
			width = width.reshape((len(width), 1))
			width_visited = width_visited.reshape((len(width_visited), 1))
			#print(str(np.count_nonzero(width)) + ' of '+str(len(width))+ 'is nonzero')
			self.bonus_print = np.copy(query_pool)
			self.bonus_print = np.append(self.bonus_print, width, axis=1)
			self.plot_bonus(mode='postwidth')
			self.bonus_print = np.copy(replay_buffer)
			self.bonus_print = np.append(self.bonus_print, width_visited, axis=1)
			self.plot_bonus(mode='postvisited')
    
    def get_critic_width(self, mode, in_obs): 
        
        # obtain width as |target - output|
        width = torch.abs(self.width_network_fixed(in_obs)['v']-self.width_network(in_obs)['v'])
        
        # rescale width
        if self.config.width_max > 0:
            width /= self.config.width_max
        
        return width

    def time(self, tag=''):
        if self.time is None or tag=='reset':
            self.timestamp = time()
        else:
            t = time()
            print(f'{tag} took {t - self.timestamp:.4f}s')
            self.timestamp = t
            
    # gather trajectories following a policy and return them in a buffer.
    # explore mode uses exploration bonus as reward, exploit uses environment reward
    # can specify whether to roll in using policy mixture or not
    def gather_trajectories(self, roll_in=True, add_bonus_reward=True, debug=False, mode=None, record_return=False):
        config = self.config
        states = self.states
        network = self.network[mode]
        roll_in_length = 0 if (debug or not roll_in) else np.random.randint(0, config.horizon - 1)
        roll_out_length = config.horizon - roll_in_length
        storage = Storage(roll_out_length)

        if roll_in_length > 0:
            assert roll_in
            # Sample previous policy to roll in
            #print(self.policy_mixture_weights)
            i = torch.multinomial(self.policy_mixture_weights.cpu(), num_samples=1)
            self.network['rollin'].load_state_dict(self.policy_mixture[i])

            # Roll in
            for _ in range(roll_in_length):
                if self.config.obs_type:
                    states = self.state_to_obs(states)
                prediction = self.network['rollin'](states)
                next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
                next_states = config.state_normalizer(next_states)
                states = next_states
                self.total_steps += config.num_workers
        
        # Roll out
        for i in range(roll_out_length):
            if self.config.obs_type:
                states = self.state_to_obs(states)
            if i == 0 and roll_in: #if roll-in is false, then we ignore epsilon greedy and simply roll-out the current policy
                # we are using \hat{\pi}
                sample_eps_greedy = np.random.rand() < self.config.eps
                if sample_eps_greedy:
                    if isinstance(self.task.action_space, Discrete):
                        actions = torch.randint(self.config.action_dim, (states.shape[0],)).cpu()
                    elif isinstance(self.task.action_space, Box):
                        actions = self.uniform_sample_cont_random_acts(states.shape[0])
                    prediction = network(states, tensor(actions))
                else:
                    prediction = network(states)
                # to-do
                #update the log_prob_a by including the epsilon_greed
                prediction['log_pi_a'] = (prediction['log_pi_a'].exp() * (1.-self.config.eps) + self.config.eps*self.uniform_prob).log()
            else:
                # we are using \pi
                prediction = network(states)

            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
           
            if add_bonus_reward:
                
                s = states
                
                # check if retrain width network
                if self.config.bonus == 'width' and self.query_counter < (self.config.query_size + 1):
                    if self.config.online: 
                        self.query_pool = np.copy(s)
                        self.retrain_width(mode = 'explore')
                    else:
                        if not self.query_counter:
                            self.query_pool = np.copy(s)
                        elif self.query_counter < self.config.query_size:
                            self.query_pool = np.append(self.query_pool, s, axis=0)
                        else: # after cumulating enough queries, retrain width
                            self.retrain_width(mode = 'explore')
                    self.query_counter += 1
                
                # calculate bonus
                reward_bonus = self.config.reward_bonus_normalizer(self.compute_reward_bonus(s,to_np(prediction['a'])))
                
                # increment bonus to rewards
                if self.config.bonus_choice == 1:
                    rewards = np.maximum(self.config.bonus_coeff * self.config.horizon * reward_bonus, rewards)
                    # check if rewards are zero
                    #if self.epoch > 1 and max(rewards) == 0 and self.config.bonus == 'width':
                    #    print("zero")
                elif self.config.bonus_choice == 2:
                    rewards = np.add(self.config.bonus_coeff * self.config.horizon * reward_bonus, rewards)
                
                # plot bonus map for PC-PG
                if self.config.bonus == 'rbf-kernel':
                    reward_bonus = np.reshape(reward_bonus, (len(reward_bonus), 1))
                    s = np.append(s, reward_bonus, axis=1)
                    
                    if not self.query_counter:
                        self.bonus_print = np.copy(s)
                    elif self.query_counter == 10000:
                        self.plot_bonus(mode='linear')
                        
                    elif self.query_counter < 10000:
                        self.bonus_print = np.append(self.bonus_print, s, axis=0)
                        
                    self.query_counter += 1   
                    
                    
                #    if not len(self.bonus_print):
                #        self.bonus_print = np.copy(s)
                #    else:
                #        self.bonus_print = np.append(self.bonus_print, s, axis=0)
                    
                #    self.counter_print += 1
                #    if self.counter_print == 10000:
                #        self.plot_bonus()
                        
                #assert(all(rewards >= 0))

            if record_return:
                self.record_online_return(info)

            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         'i': list(info), 
                         's': tensor(states)})
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        if self.config.obs_type:
            states = self.state_to_obs(states)
        prediction = network(states)
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = prediction['v'].detach()
        for i in reversed(range(roll_out_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        return storage

    def log(self, s):
        logtxt(self.logger.log_dir + '.txt', s, show=True, date=False)


    def compute_kernel(self, states, actions = None):
        if self.config.bonus == 'randnet-kernel-s':
            phi = F.normalize(self.kernel(tensor(states).cpu()), p=2, dim=1)
        elif self.config.bonus == 'randnet-kernel-sa':
            actions_one_hot = tensor(np.eye(self.config.action_dim)[actions])
            state_actions = torch.cat((tensor(states).cpu(), actions_one_hot), dim=1)
            phi = F.normalize(self.kernel(state_actions), p=2, dim=1)
        elif self.config.bonus == 'id-kernel-s':
            phi = states.cpu()
        elif self.config.bonus == 'id-kernel-sa':
            actions_one_hot = tensor(np.eye(self.config.action_dim)[actions])
            state_actions = torch.cat((tensor(states).cpu(), actions_one_hot), dim=1)
            phi = state_actions
        elif self.config.bonus == 'rbf-kernel':
        #elif self.config.bonus == 'width':
            assert actions is not None
            if actions is None:
                phi = self.rbf_feature.transform(states.cpu().numpy())
                phi = torch.tensor(phi).cpu()
            else:
                #concatenate state and act features
                np_states = states.cpu().numpy()
                np_actions = actions.cpu().numpy()
                if isinstance(self.task.action_space, Discrete):
                    np_actions = np.expand_dims(np_actions, axis = 1)
                assert np_actions.ndim == 2 and np_actions.shape[0] == np_states.shape[0] 
                states_acts_cat = np.concatenate((np_states, self.clip_actions(np_actions)), axis = 1)
                phi = self.rbf_feature.transform(states_acts_cat)
                phi = torch.tensor(phi).cpu()
        else:
            raise NotImplementedError
        return phi


    # for visualizing visitations in combolock
    def log_visitations(self, visitations):
        self.log('lock1')
        self.log(np.around(visitations[0], 3))
        self.log('lock2')
        self.log(np.around(visitations[1], 3))

    # turn count-based density model into visitation table
    def compute_state_visitations(self, density_model, use_one_hot=False):
        locks = [np.zeros((3, self.config.horizon-1)), np.zeros((3, self.config.horizon-1))]
        N = sum(list(density_model.values()))
        for state in density_model.keys():
            if use_one_hot:
                k = np.argmax(state)
                (s, l, h) = np.unravel_index(k , (3, 3, self.config.horizon))
                if l in [0, 1]:
                    locks[l][s][h] += float(density_model[state]) / N
            else:
                if not all(np.array(state)==0.0):
                    s = np.argmax(state[:3])
                    l = int(state[-1])
                    h = np.argmax(state[3:-1])
                    locks[l][s][h] += float(density_model[state]) / N
        return locks

    '''
    def compute_state_visitations(self, infos):
        locks = [np.zeros((3, self.config.horizon)), np.zeros((3, self.config.horizon))]
        for info in infos:
            i, h, lock = info['state']
            if lock in [0, 1]:
                locks[lock][i, h] += 1
        locks[0] /= len(infos)
        locks[1] /= len(infos)
        return locks
    ''' 
    # update the density model using data from replay buffer.
    # also computes covariance matrices for kernel case. 
    def update_density_model(self, mode=None):
        replay_buffer = self.replay_buffer[mode]
        replay_buffer_act = self.replay_buffer_actions[mode]
        states = torch.cat(sum(replay_buffer, []))
        actions = torch.cat(sum(replay_buffer_act,[]))
        
        if self.config.bonus == 'rnd':
            states = states.cpu()
            targets = self.rnd_network(states).detach()
            data = DataLoader(TensorDataset(states, targets), batch_size = 100, shuffle=True)

            for i in range(1):
                total_loss = 0
                losses = []
                for j, batch in enumerate(data):
                    self.rnd_optimizer.zero_grad()
                    pred = self.rnd_pred_network(batch[0])
                    loss = F.mse_loss(pred, batch[1], reduction='none')
                    (loss.mean()).backward()
                    self.rnd_optimizer.step()
                    total_loss += loss.mean().item()
                    losses.append(loss)
                print(f'[RND loss: {total_loss / j:.5f}]')
            bonuses = torch.cat(losses).view(-1)
        
        elif self.config.bonus == 'rbf-kernel':
        #elif self.config.bonus == 'width':
            N = states.shape[0]
            ind = np.random.choice(N, min(2000, N), replace=False)
            pdists = scipy.spatial.distance.pdist((states.cpu().numpy())[ind])
            self.rbf_feature.gamma = 1./(np.median(pdists)**2)
            phi = self.compute_kernel(states, actions = actions)
            n, d = phi.shape
            sigma = torch.mm(phi.t(), phi) + self.config.ridge*torch.eye(d).cpu()
            self.density_model = torch.inverse(sigma).detach()

            covariance_matrices = []
            assert len(replay_buffer) == len(replay_buffer_act)
            #for i, states in enumerate(replay_buffer):
            for i in range(len(replay_buffer)):
                states = torch.cat(replay_buffer[i])
                actions = torch.cat(replay_buffer_act[i])
                phi = self.compute_kernel(states,actions)
                n, d = phi.shape
                sigma = torch.mm(phi.t(), phi) + self.config.ridge*torch.eye(d).cpu()
                covariance_matrices.append(sigma.detach())
            m = 0
            for matrix in covariance_matrices:
                m = max(m, matrix.max())
            covariance_matrices = [matrix / m for matrix in covariance_matrices]

        elif 'kernel' in self.config.bonus:
            N = states.shape[0]
            phi = self.compute_kernel(states, actions)
            n, d = phi.shape
            sigma = torch.mm(phi.t(), phi) + self.config.ridge*torch.eye(d).cpu()
            self.density_model = torch.inverse(sigma).detach()

            covariance_matrices = []
            assert len(replay_buffer) == len(replay_buffer_act)
            #for i, states in enumerate(replay_buffer):
            for i in range(len(replay_buffer)):
                states = torch.cat(replay_buffer[i])
                actions = torch.cat(replay_buffer_act[i])
                phi = self.compute_kernel(states, actions)
                n, d = phi.shape
                sigma = torch.mm(phi.t(), phi) + self.config.ridge*torch.eye(d).cpu()
                covariance_matrices.append(sigma.detach().cpu())
            m = 0
            for matrix in covariance_matrices:
                m = max(m, matrix.max())
            covariance_matrices = [matrix / m for matrix in covariance_matrices]

        elif 'counts' in self.config.bonus:
            states = [tuple(s) for s in states.numpy()]
            unique_states = list(set(states))
            self.density_model[mode] = dict(zip(unique_states, [0] * len(unique_states)))
            for s in states: self.density_model[mode][s] += 1
            bonuses = torch.tensor([1.0/self.density_model[mode][s] for s in states])
            covariance_matrices, visitations = [], []
            for i, states in enumerate(replay_buffer):
                states = [tuple(s) for s in torch.cat(states).numpy()]
                density_model = dict(zip(unique_states, [0] * len(unique_states)))
                for s in states: density_model[s] += 1
                sums=torch.tensor([density_model[s] for s in unique_states]).float()
                covariance_matrices.append(torch.diag(sums) + torch.eye(len(unique_states)))
                visitations.append(self.compute_state_visitations(density_model))

            m = 0
            for matrix in covariance_matrices:
                m = max(m, matrix.max())
            covariance_matrices = [matrix / m for matrix in covariance_matrices]

        if mode == 'explore': self.optimize_policy_mixture_weights(covariance_matrices)

        if 'combolock' in self.config.game:

            visitations = []
            states = torch.cat(sum(replay_buffer, []))
            states = [tuple(s) for s in states.numpy()]
            unique_states = list(set(states))
            for i, states in enumerate(self.replay_buffer[mode]):
                states = [tuple(s) for s in torch.cat(states).numpy()]
                density_model = dict(zip(unique_states, [0] * len(unique_states)))
                for s in states: density_model[s] += 1
#                visitations.append(self.compute_state_visitations(self.replay_buffer_infos[mode][i]))
                visitations.append(self.compute_state_visitations(density_model))
                
            if mode == 'explore':
                weighted_visitations = [np.zeros((3, self.config.horizon - 1)), np.zeros((3, self.config.horizon - 1))]
                for i in range(len(visitations)):
                    weighted_visitations[0] += self.policy_mixture_weights[i].item()*visitations[i][0]
                    weighted_visitations[1] += self.policy_mixture_weights[i].item()*visitations[i][1]

                for i in range(len(visitations)):
                    self.log(f'\nstate visitations for policy {i}:')
                    self.log_visitations(visitations[i])

                self.log(f'\nstate visitations for weighted policy mixture:')
                self.log_visitations(weighted_visitations)
                
            elif mode == 'exploit':
                self.log(f'\nstate visitations for exploit policy:')
                self.log_visitations(visitations[-1])

        if self.config.norm_rew_b == 1:
            self.reward_bonus_normalizer = MeanStdNormalizer(read_only=True)
            self.reward_bonus_normalizer.rms = RunningMeanStd(shape=(1,))
            self.reward_bonus_normalizer.rms.mean = torch.mean(bonuses).item()
            self.reward_bonus_normalizer.rms.var = torch.var(bonuses).item()
        else:
            self.reward_bonus_normalizer= RescaleNormalizer()



    # optimize policy mixture weights using log-determinant loss
    def optimize_policy_mixture_weights(self, covariance_matrices):
        d = covariance_matrices[0].shape[0]
        N = len(covariance_matrices)
        if N == 1:
            self.policy_mixture_weights = torch.tensor([1.0])
        else:
            self.log_alphas = nn.Parameter(torch.randn(N))
            opt = torch.optim.Adam([self.log_alphas], lr=0.001)
            for i in range(5000):
                opt.zero_grad()
                sigma_weighted_sum = torch.zeros(d, d)
                for n in range(N):
                    sigma_weighted_sum += F.softmax(self.log_alphas, dim=0)[n]*covariance_matrices[n]
                loss = -torch.logdet(sigma_weighted_sum)
                if math.isnan(loss.item()):
                    pdb.set_trace()
                #if not i % 500:
                    #print(f'optimizing log det, loss={loss.item()}')
                loss.backward()
                opt.step()
            with torch.no_grad():
                self.policy_mixture_weights = F.softmax(self.log_alphas, dim=0)
        self.log(f'\npolicy mixture weights: {self.policy_mixture_weights.numpy()}')


    # roll out using explore/exploit policies and store data in replay buffer
    def update_replay_buffer(self):
        print('[gathering trajectories for replay buffer]')
        for mode in ['explore', 'exploit']:
            states, actions, returns, infos = [], [], [], []
            for _ in range(self.config.n_rollouts_for_density_est):
                new_traj = self.gather_trajectories(roll_in=False, add_bonus_reward=False, mode=mode,
                                                    record_return=(mode=='exploit'))            
                states += new_traj.cat(['s'])
                returns += new_traj.cat(['r'])
                actions += new_traj.cat(['a']) #append actions as well
                infos += new_traj.i

            mean_return = torch.cat(returns).cpu().mean()*self.config.horizon
            if mode == 'explore':
                self.policy_mixture_returns.append(mean_return.item())
                self.log(f'[policy mixture returns: {np.around(self.policy_mixture_returns, 3)}]')
            states = [s.cpu() for s in states]
            #print(f'return ({mode}): {mean_return}')
            self.replay_buffer[mode].append(states)
 
            actions = [a.cpu() for a in actions]
            self.replay_buffer_actions[mode].append(actions)
            self.replay_buffer_infos[mode].append(sum(infos, []))
        
    # optimize explore and/or exploit policies
    def optimize_policy(self):
        if self.config.init_new_policy == 1:
            self.initialize_new_policy('explore')
            
        print(self.policy_mixture_weights)
        for mode in ['explore', 'exploit']:
            if mode == 'exploit' and self.epoch < self.config.start_exploit:
                continue
            for i in range(self.config.n_policy_loops):
                rewards = self.step_optimize_policy(mode=mode)
                if not i % 5: 
                    print(f'[optimizing policy ({mode}), step {i}, mean reward: {rewards.mean():.5f}]')
                    
                self.config.print = 0
                
        if self.config.flag:           
            self.policy_mixture.append(copy.deepcopy(self.network['explore'].state_dict()))
            self.policy_mixture_optimizers.append(copy.deepcopy(self.optimizer['explore'].state_dict()))
            print(f'{len(self.policy_mixture)} policies in mixture')
            if self.config.bonus == 'width':
                self.policy_mixture_weights = torch.ones([len(self.policy_mixture)], dtype = torch.float32)
            
    def initialize_new_policy(self, mode):
        self.network[mode] = self.config.network_fn()
        self.optimizer[mode] = self.config.optimizer_fn(self.network[mode].parameters())
        

        '''
        i = np.argmax(self.policy_mixture_returns)
        self.log(f'[warmstarting exploit policy using policy {i} with return {self.policy_mixture_returns[i]:.4f}]')
        self.network['exploit'].load_state_dict(self.policy_mixture[i])
        self.optimizer['exploit'].load_state_dict(self.policy_mixture_optimizers[i])
        '''


    # gather a batch of data and perform some on-policy optimization steps
    def step_optimize_policy(self, mode=None):
        config = self.config
        network = self.network[mode]
        optimizer = self.optimizer[mode]

        states, actions, rewards, log_probs_old, returns, advantages = [], [], [], [], [], []
        #self.time('reset')
        for i in range(self.config.n_traj_per_loop):

            #with probability (1-proll), we roll-in from the policy itself (so no data is wasted), with probability proll from mixture
            coin = np.random.rand()
            if coin < (1-config.proll):
                traj = self.gather_trajectories(add_bonus_reward=(mode=='explore'), mode=mode, roll_in = False)
            else: #from mixture
                traj = self.gather_trajectories(add_bonus_reward=(mode=='explore'), mode=mode, roll_in = True)
            
            states += traj.cat(['s'])
            actions += traj.cat(['a'])
            log_probs_old += traj.cat(['log_pi_a'])
            returns += traj.cat(['ret'])
            rewards += traj.cat(['r'])
            advantages += traj.cat(['adv'])
            
        #self.time('gathering trajectories')
        states = torch.cat(states, 0)
        actions = torch.cat(actions, 0)
        log_probs_old = torch.cat(log_probs_old, 0)
        returns = torch.cat(returns, 0)
        rewards = torch.cat(rewards, 0)
        advantages = torch.cat(advantages, 0)
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == advantages.shape[0] == returns.shape[0]
        
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        #self.time('reset')
        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_returns = returns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = network(sampled_states, sampled_actions)
 
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                #print(ratio)
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                #print(obj_clipped)
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()
                #print(policy_loss)
                value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()
                #print(value_loss)
                optimizer.zero_grad()
                (policy_loss + value_loss).backward()
                nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)
                optimizer.step()
        #self.time('optimizing policy')
        return rewards.mean()


    #we clip the actions since the policy uses Gaussian distribution to sample actions
    #this avoids the policy generating large actions to maximum the neg log-det.
    def clip_actions(self, actions): 
        #action: numpy
        if isinstance(self.task.action_space, Box):
            #only clip in continuos setting. 
            for i in range(self.config.action_dim):
                actions[:, i] = np.clip(actions[:,i], self.task.action_space.low[i], 
                    self.task.action_space.high[i])
        #embed()
        return actions
        
    
    def eval_step(self, state):
        network = self.network['exploit']
        if self.config.obs_type:
            state = self.state_to_obs(state)
        prediction = network(state)
        action = to_np(prediction['a'])
        return action


    #test function for policy: 
    def test_exploit_policy_performance(self):
        network = self.network['exploit']
        roll_in_length = self.config.horizon
        storage = Storage(roll_in_length)
        num_trajs = 0
        total_rews = 0
        states = self.task.reset() #reset environment, so roll-in from the beignning
        for i in range(roll_in_length+1):
            if self.config.obs_type:
                states = self.state_to_obs(states)
            prediction = network(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            num_trajs += terminals.sum()
            total_rews += rewards.sum()
            states = next_states
            
        #assert num_trajs > 0
        #print(total_rews / num_trajs) #this may overestimates rewards...but fair for all baselines as well..
        print(f'[episodes {self.total_steps / self.config.horizon}, mean reward: {total_rews / self.config.num_workers:.5f}]')
    
    def state_to_obs(self, states):
        assert self.config.obs_type > 0
        # cast states to tensor
        states = torch.FloatTensor(states).detach().to(self.config.DEVICE)
        
        # output (x_1, 10*x_1*x_2)
        if self.config.obs_type == 1:
            states[:,1] = 10 * states[:,0] * states[:,1]
        # output (1/(x_1+1), 1/(x_2+1))
        elif self.config.obs_type == 2:
            states[:,:] = 1./(states[:,:]+1.)
        # padding with noise
        elif self.config.obs_type == 3:
            low = self.config.noise_low
            high = self.config.noise_high
            noise = torch.empty(len(states), self.config.noise_dim).uniform_(low, high).detach().to(self.config.DEVICE)
            states = torch.cat((states,noise), dim=1).detach().to(self.config.DEVICE)
            
        return states
    
    def plot_bonus(self, mode):
        print("start plotting bonus")
        config = self.config
        x = self.bonus_print[:, 0]
        y = self.bonus_print[:, 1]
        value = self.bonus_print[:, 2]
        
        min_index = value.argmin()
        print(x[min_index], y[min_index], value[min_index])
        max_index = value.argmax()
        print(x[max_index], y[max_index], value[max_index])
        
        cm = plt.cm.get_cmap('RdYlBu')
        sc = plt.scatter(x, y, c=value, vmin=value[min_index], vmax=value[max_index], s=2, cmap=cm)
        plt.colorbar(sc)
       
        plt.xlim(-1.3, 0.7)
        plt.ylim(-0.08, 0.08)
        #plt.ylim(-0.8, 0.8)
        plt.xlabel('position', fontsize = 20)
        plt.ylabel('10*velocity', fontsize = 20)
        plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
        if config.bonus == 'width':
            plt.savefig(str(mode)+'_rollout_'+str(config.n_rollouts_for_density_est)+'_loop_'+str(config.width_loop)
                    +'_lam_'+str(config.width_loss_lambda)+'_lam1_'+str(config.width_loss_lambda1)+'_proll_'+str(config.proll)
                    +'_select_'+str(config.bonus_select)+'_query_'+str(config.query_batch)+"_gd_"+str(config.width_gd_steps)
                    + '_clip_'+str(config.gradient_clip)+'_layer_'+str(config.layer)+"_coeff_"+str(config.bonus_coeff)
                    +'_copy_'+str(config.copy)+'_horizon_'+str(config.horizon)+'_decay_'+str(config.weight_decay)
                    +'_lr_'+str(config.width_lr)+'_seed_'+str(config.seed)+'_epoch_'+str(self.epoch)+'.png', transparent=False, 
                    bbox_inches='tight', pad_inches=0.1) 
        if config.bonus == 'rbf-kernel':
            plt.savefig(str(mode)+'_rollout_'+str(config.n_rollouts_for_density_est)+'_proll_'+str(config.proll)
                    + '_clip_'+str(config.gradient_clip)+'_layer_'+str(config.layer)+'_coeff_'+str(config.bonus_coeff)
                    +'_horizon_'+str(config.horizon)+'_decay_'+str(config.weight_decay)
                    +'_seed_'+str(config.seed)+'_epoch_'+str(self.epoch)+'.png', transparent=False, bbox_inches='tight', pad_inches=0.1) 
        print("finish plotting bonus")
        plt.close()
        
    def plot_visitation(self, mode):
        print("start plotting")
        config = self.config
        replay_buffer = self.replay_buffer[mode]
        states = torch.cat(sum(replay_buffer, []))
        #plt.figure(figsize=(7,5))
        x = states[:, 0]
        y = states[:, 1]
        if len(self.policy_mixture) == 1:
            plt.title('1 policy', fontsize=20)
        else:
            plt.title(str(len(self.policy_mixture))+' policies', fontsize=15)
        if self.config.bonus == 'width':
            COLOR = 'red'
        else:
            COLOR = 'blue'
        plt.plot(x, y, '.', color=COLOR, markersize=0.5, alpha=0.7)
        plt.plot(0.45, 0, '*', color='black', markersize = 10)
        plt.plot(x[0], y[0], 'o', color='yellow', markersize = 10)
        plt.xlim(-1.3, 0.7)
        plt.ylim(-0.08, 0.08)
        plt.xlabel('position', fontsize = 20)
        plt.ylabel('10*velocity', fontsize = 20)
        plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
        plt.savefig(str(mode)+'_layer_'+str(config.layer)+'_coeff_'+str(config.bonus_coeff)+'_clip_'+str(config.gradient_clip)
                    +'_interval_'+str(config.retrain_interval)
                    +'_lr_'+str(config.lr)+'_seed_'+str(config.seed)+'_epoch_'+str(self.epoch)+'.png', transparent=False, 
                    bbox_inches='tight', pad_inches=0.1) 
        print("finish plotting bonus")
        plt.close()