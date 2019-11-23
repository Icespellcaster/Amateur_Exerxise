"""
尝试使用强化学习求解一维横场ising模型的基态能

H = -J * sigma_z^i sigma_z^{i+1} - h * sigma_x^i 的基态能
"""

import os 
os.chdir('./test')


import numpy as np
import pandas as pd
import time

np.random.seed(2)

L = 10 # The length of chain
CHANNEL = 2
ACTIONS = ['up', 'down']
MAX_EPISODES = 130
FRESH_TIME = 0.3
ITER = 0



class TFI_1d:
    def __init__(self, L, channel, iter):
        self.L = L
        self.channel = channel
        self.iter = iter
        self.init_config()

    def init_config(self):
        self.config = np.zeros((self.L, self.channel))
        x = np.random.randint(2, size=(self.L))
        self.config[:,0] = x
        self.config[:,1] = 1 - x

    def local_E_Ising(self, h=1):
        """
        H = -J sz_i sz_j - h sx_i
        """
        L, channel = self.config.shape
        localE = 0.
        for i in range(L-1):
            temp = self.config[i, :].dot(self.config[i+1, :])
            localE -= 2 * (temp - 0.5)

        # Periodic Boundary condition
        temp = self.config[0, :].dot(self.config[-1, :])
        localE -= 2 * (temp - 0.5)
        #########################################

        return localE

    def init_env(self):
        env_list = []
        for i in range(self.config.shape[0]):
            if self.config[i, 0] == 1:
                env_list += ['↑'] 
            else:
                env_list += ['↓']
        return env_list

    def choose_action(self, state):
        # This is how to choose an action
        
        if self.config[state, 0] == 1:
            action_name = ACTIONS[1]
        else:
            action_name = ACTIONS[0]
        
        return action_name

    def get_env_feedback(self, S, A):
        S_ = ((S + 1) % self.L)
        oldlocalE = self.local_E_Ising(self.config)
        if A == 'up':
            self.config[S, 0] = 1
            self.config[S, 1] = 0
            localE = self.local_E_Ising(self.config)
            if localE < oldlocalE:
                pass
            elif localE == oldlocalE:
                if self.iter % 2 == 1:
                    self.config[S, 0] = 0
                    self.config[S, 1] = 1
                    localE = oldlocalE
                self.iter += 1
            else:
                self.config[S, 0] = 0
                self.config[S, 1] = 1
                localE = oldlocalE
        else:
            self.config[S, 0] = 0
            self.config[S, 1] = 1
            localE = self.local_E_Ising(self.config)
            if localE < oldlocalE:
                pass
            else:
                self.config[S, 0] = 1
                self.config[S, 1] = 0
                localE = oldlocalE
        return S_, localE

    def update_env(self, episode):
        # This is how environment be updated
        env_list = self.init_env()

        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def main():
    ising = TFI_1d(L, CHANNEL, ITER)

    
    localE = ising.local_E_Ising()
    print('init localE: ', localE)
    
    # main part of RL loop
    init_env_list = ising.init_env()
    print(''.join(init_env_list))
    S = 0
    for episode in range(MAX_EPISODES):
        
        A = ising.choose_action(S)
        
        S_, localE = ising.get_env_feedback(S, A)
        
        S = S_
        
        print('##########################')
        print('episode: %d' % episode)
        ising.update_env(episode)
        print()
        print('localE: ', localE)

        if episode %10 == 0:
            input()


if __name__ == '__main__':