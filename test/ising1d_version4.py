"""
尝试使用强化学习求解一维横场ising模型的基态能

H = -J * sigma_z^i sigma_z^{i+1} - h * sigma_x^i 的基态能

存在问题1：自旋分块同向，导致系统陷入局部最小。
问题1已解决
"""

import os 
os.chdir('./test')

import tensorflow as tf
import numpy as np
import pandas as pd
import time

np.random.seed(3)
tf.set_random_seed(1)

L = 10 # The length of chain
CHANNEL = 2
# ACTIONS = ['up', 'down']
# [000, 001, 010, 011, 100, 101, 110, 111]
# [0,1,2,3,4,5,6,7]
# ACTIONS = ['ddd', 'ddu', 'dud', 'duu', 'udd', 'udu', 'uud', 'uuu']
ACTIONS = 8
MAX_EPISODES = 130
FRESH_TIME = 0.1
LR = 1e-2
EPSILON = 0.9
ALPHA = 0.1


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, actions)),
    )

    return table


class TFI_1d:
    def __init__(self, L, channel, learning_rate=1e-3, momentum=0.9, h=1):
        self.L = L
        self.channel = channel
        self.h = h
        self.init_config()
        self.x = tf.placeholder(tf.float32, [self.L, self.channel])
        self.amp = self.build_TFI_1d(self.x)
        
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, 
                                                    momentum=momentum)

        self.local_E_Ising()
        self.train_op = self.optimizer.minimize(self.loss)
        

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth=True
        # config.allow_soft_placement = True
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # print(self.sess.run(self.localE, feed_dict={self.x: self.config}))
        # print(self.sess.run(self.loss, feed_dict={self.x: self.config}))
        # input()

    def init_config(self):
        self.config = np.zeros((self.L, self.channel))
        x = np.random.randint(2, size=(self.L))
        # x = np.ones((10))
        self.config[:,0] = x
        self.config[:,1] = 1 - x
        self.config.astype(np.float32)

    def A2list(self, A):
        A_list = []
        A_list.append(A//2//2)
        A_list.append((A-4*A_list[0])//2)
        A_list.append((A-4*A_list[0]-2*A_list[1]))
        return A_list

    def update_config(self, S, A_list):
        self.config[S, 0] = A_list[0]
        self.config[S, 1] = 1 - A_list[0]
        self.config[(S+1) % self.L, 0] = A_list[1]
        self.config[(S+1) % self.L, 1] = 1 - A_list[1]
        self.config[(S+2) % self.L, 0] = A_list[2]
        self.config[(S+2) % self.L, 1] = 1 - A_list[2]

    def local_E_Ising(self):
        """
        H = -J sz_i sz_j - h sx_i
        """
        L, channel = self.x.shape
        self.localE = tf.constant(0.)
        self.loss = tf.constant(0.)
        for i in range(L-1):
            temp = tf.reduce_sum(tf.multiply(self.x[i, :], (self.x[i+1, :])))
            self.localE -= 2 * (temp - 0.5)
            
        # Periodic Boundary condition
        temp = tf.reduce_sum(tf.multiply(self.x[0, :], (self.x[-1, :])))
        self.localE -= 2 * (temp - 0.5)
        #########################################
        
        oldAmp = self.amp
        num_oldAmp = oldAmp[0][0]
        
        self.loss = tf.constant(0.)
        
        tempAmp = []
        for i in range(self.L):
            # tempAmp = self.sess.run(self.amp, feed_dict={self.x: tempConfig})
            tempAmp.append(self.build_TFI_1d(tf.constant(1., shape=[self.L, self.channel]) - self.x[i, :]))
            num_tempAmp = tempAmp[i][0][0]
            self.localE -= self.h * num_tempAmp / num_oldAmp
            self.loss -= (tempAmp[i] - oldAmp)

    def init_env(self):
        env_list = []
        for i in range(self.config.shape[0]):
            if self.config[i, 0] == 1.:
                env_list += ['↑'] 
            else:
                env_list += ['↓']
        return env_list

    '''
    def choose_action(self, state):
        # This is how to choose an action
        
        if self.config[state, 0] == 1:
            action_name = ACTIONS[1] # dwon
        else:
            action_name = ACTIONS[0] # up
        
        return action_name
    '''

    def choose_action(self, state, q_table):
        state_actions = q_table.iloc[0, :]
        if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
            action_name = np.random.choice(ACTIONS)
        else:
            action_name = state_actions.idxmax()
        return action_name

    def get_env_feedback(self, S, A, Loss):
        S_ = ((S + 1) % self.L)
        oldlocalE = self.sess.run(self.localE, feed_dict={self.x: self.config})
        oldconfig = self.config.copy()
        A_list = self.A2list(A)
        self.update_config(S, A_list)
        localE = self.sess.run(self.localE, feed_dict={self.x: self.config})
        if localE < oldlocalE:
            R = 2
        else:
            R = -1
            localE = oldlocalE
            self.config = oldconfig
            self.sess.run(self.train_op, feed_dict={self.x: self.config})
        Loss.append(self.sess.run(self.loss, feed_dict={self.x: self.config}))
        return S_, Loss, localE, R

    def update_env(self, episode):
        # This is how environment be updated
        env_list = self.init_env()

        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

    def build_TFI_1d(self, x):
        x = x[:,0]
        x = tf.reshape(x, [-1,self.L])
        x_shape = x.get_shape()

        # fc1
        layer_name = 'fc1'; dim = x_shape[1]; hiddens = 256
        self.w1 = tf.Variable(tf.random_normal([dim, hiddens]), name = layer_name + '_weights')
        self.b1 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name = layer_name + '_biases')
        self.ip1 = tf.add(tf.matmul(x, self.w1), self.b1, name = layer_name + '_ip1')
        self.out1 = tf.nn.relu(self.ip1, name = layer_name + '_activations')

        # fc2 
        layer_name = 'fc2'; dim = 256; hiddens = 64
        self.w2 = tf.Variable(tf.random_normal([dim, hiddens]), name = layer_name + '_weights')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name = layer_name + '_biases')
        self.ip2 = tf.add(tf.matmul(self.out1, self.w2), self.b2, name = layer_name + '_ip1')
        self.out2 = tf.nn.relu(self.ip2, name = layer_name + '_activations')

        # fc3
        layer_name = 'fc3'; dim = 64; hiddens = 1
        self.w3 = tf.Variable(tf.random_normal([dim, hiddens]), name = layer_name + '_weights')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name = layer_name + '_biases')
        self.ip3 = tf.add(tf.matmul(self.out2, self.w3), self.b3, name = layer_name + '_ip1')
        self.out3 = tf.nn.sigmoid(self.ip3, name = layer_name + '_outputs')
        self.y = tf.exp(self.out3)

        return self.y



def main():
    ising = TFI_1d(L=L, channel=CHANNEL, learning_rate=LR)
    
    q_table = build_q_table(1, ACTIONS)
    
    
    # main part of RL loop
    init_env_list = ising.init_env()
    print(''.join(init_env_list))
    S = 0
    Loss = []
    for episode in range(MAX_EPISODES):
        A = ising.choose_action(S, q_table)
        print('ACTION: ', A)
        print(q_table)
        S_, Loss, localE, R = ising.get_env_feedback(S, A, Loss)
        
        q_target = R
        q_table.iloc[0, A] += ALPHA * q_target 
        
        S = S_
        print('##########################')
        print('episode: %d' % episode)
        ising.update_env(episode)
        print()
        print('localE: ', localE/L)
        print('loss: ', Loss[-1])
        if episode %10 == 0:
            input()
        # input()


if __name__ == '__main__':
    main()



