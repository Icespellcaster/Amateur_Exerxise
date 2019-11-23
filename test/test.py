"""
尝试使用强化学习求解一维ising模型的基态能

H = -J * sigma_z^i sigma_z^{i+1} 的基态能
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
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3
ITER = 0

def build_q_table(L, actions):
    table = pd.DataFrame(
        np.zeros((L, len(actions))),
        columns=actions,
    )
    # for i in range(config.shape[0]):
    #     table.iloc[i, 0] = config[i, 0]
    #     table.iloc[i, 1] = config[i, 1]
    # print(config)
    # print(list(table))
    # print(table)
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    
    if config[state, 0] == 1:
        action_name = ACTIONS[1]
    else:
        action_name = ACTIONS[0]
    
    return action_name


def get_env_feedback(S, A, config, iter, L):
    S_ = ((S + 1) % L)
    oldlocalE = local_E_Ising(config)
    if A == 'up':
        config[S, 0] = 1
        config[S, 1] = 0
        localE = local_E_Ising(config)
        if localE < oldlocalE:
            R = 1
        elif localE == oldlocalE:
            if iter % 2 == 1:
                config[S, 0] = 0
                config[S, 1] = 1
                localE = oldlocalE
                R = 0
            iter += 1
            R = 1
        else:
            config[S, 0] = 0
            config[S, 1] = 1
            localE = oldlocalE
            R = 0
    else:
        config[S, 0] = 0
        config[S, 1] = 1
        localE = local_E_Ising(config)
        if localE < oldlocalE:
            R = 1
        else:
            config[S, 0] = 1
            config[S, 1] = 0
            localE = oldlocalE
            R = 0
    return S_, R, localE, iter

def init_env(config):
    env_list = []
    for i in range(config.shape[0]):
        if config[i, 0] == 1:
            env_list += ['↑'] 
        else:
            env_list += ['↓']
    return env_list


def update_env(episode, step_counter, config):
    # This is how environment be updated
    env_list = init_env(config)

    interaction = ''.join(env_list)
    print('\r{}'.format(interaction), end='')
    time.sleep(FRESH_TIME)


# 假定格点数为10
def init_config(L, channel):
    config = np.zeros((L, channel))
    x = np.random.randint(2, size=(L))
    config[:,0] = x
    config[:,1] = 1 - x
    return config


def local_E_Ising(config, h=1):
    """
    H = -J sz_i sz_j - h sx_i
    """
    L, channel = config.shape
    localE = 0.
    for i in range(L-1):
        temp = config[i, :].dot(config[i+1, :])
        localE -= 2 * (temp - 0.5)

    # Periodic Boundary condition
    temp = config[0, :].dot(config[-1, :])
    localE -= 2 * (temp - 0.5)
    #########################################

    return localE



if __name__ == '__main__':
    config = init_config(L, CHANNEL)
    # print(config)
    
    localE = local_E_Ising(config)
    print('init localE: ', localE)
    
    # main part of RL loop
    init_env_list = init_env(config)
    print(''.join(init_env_list))

    q_table = build_q_table(L, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        for i in range(100):
            A = choose_action(S, q_table)
            
            S_, R, localE, ITER = get_env_feedback(S, A, config, ITER, L)
            q_predict = q_table.loc[S, A]
            q_target = R 

            q_table.loc[S, A] += ALPHA * (q_target - q_target)
            S = S_
            step_counter += 1
            print('##########################')
            print('episode: %d, step: %d' % (episode, step_counter))
            update_env(episode, step_counter+1, config)
            print()
            print('localE: ', localE)

            if i %10 == 0:
                input()