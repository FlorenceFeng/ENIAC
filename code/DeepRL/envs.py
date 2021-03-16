import gym 



class LongHallway(object):

    def __init__(self, horizon):
        self.config = config
        self.action_space = gym.spaces.Discrete(2)
        self.length = 2*self.horizon + 1
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.length,), dtype=np.float)
        self.reset()
        self.optimal_reward = 10


    def make_obs(self):
        obs = np.zeros(self.length)
        obs[self.pos] = 1
        return obs
    
    def reset(self):
        self.pos = self.horizon + 1

    def step(self, action):
        if action == 0:
            self.pos += 1
        elif action == 1:
            self.pos -= 1

        obs = self.make_obs()
        reward = self.optimal_reward if self.pos == 0 else 0
        done = (self.pos in [0, self.length])
        info = {}
        return obs, reward, done, info
        
        
        
