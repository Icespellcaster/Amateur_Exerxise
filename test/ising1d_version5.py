"""
尝试使用强化学习求解一维横场ising模型的基态能

H = -J * sigma_z^i sigma_z^{i+1} - h * sigma_x^i 的基态能

存在问题1：自旋分块同向，导致系统陷入局部最小。
问题1已解决
存在问题2：该方法求出的基态能小于精确解。
问题2未解决，尝试其他模型

尝试使用强化学习求解一维反铁磁海森堡模型的基态能

H = J sigma_i sigma_{i+1}
"""

import os 
os.chdir('./test')

import tensorflow as tf
import numpy as np
import pandas as pd
import time

np.random.seed(3)
# tf.set_random_seed(2)

L = 4 # The length of chain
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
J = 1/4
H = 1


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, actions)),
    )

    return table


class TFI_1d:
    def __init__(self, L, channel, learning_rate=1e-3, model_name='ising', momentum=0.9, h=1, j=1):
        self.L = L
        self.channel = channel
        self.h = h
        self.j = j
        self.init_config()
        self.x = tf.placeholder(tf.float32, [self.L, self.channel])
        self.amp = self.build_TFI_1d(self.x)
        
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, 
                                                    momentum=momentum)

        if model_name == 'ising':
            self.local_E_Ising()
        elif model_name == 'heisenberg':
            self.local_E_heisenberg()
        self.train_op = self.optimizer.minimize(self.loss)
        

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth=True
        # config.allow_soft_placement = True
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # print(self.sess.run(self.localE, feed_dict={self.x: self.config}))
        # print(self.sess.run(self.loss, feed_dict={self.x: self.config}))
        # input()
        # print(self.sess.run(tf.concat([self.x[:1], 
        #                             (((self.x[1]*2-1)*(-1)+1)/2)[tf.newaxis, :],
        #                             self.x[1+1:]], axis=0),
        #                             feed_dict={self.x: self.config}))
        # print(self.sess.run(self.x, feed_dict={self.x: self.config}))
        # input()

    def init_config(self):
        self.config = np.zeros((self.L, self.channel))
        x = np.random.randint(2, size=(self.L))
        # x = np.ones((self.L))
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
    
    
    # def local_E_Ising(self):
    #     """
    #     H = -J sz_i sz_j - h sx_i
    #     """
    #     L, channel = self.x.shape
    #     self.localE = tf.constant(0.)
    #     self.loss = tf.constant(0.)
    #     # for i in range(self.L-1):
    #     #     temp = tf.reduce_sum(tf.multiply(self.x[i, :], (self.x[i+1, :])))
    #     #     self.localE -= 2 * (temp - 0.5)
            
    #     # # Periodic Boundary condition
    #     # temp = tf.reduce_sum(tf.multiply(self.x[0, :], (self.x[-1, :])))
    #     # self.localE -= 2 * (temp - 0.5)

    #     oldAmp = self.amp
    #     num_oldAmp = oldAmp[0][0]
        
        
    #     '''
    #     temp = []

    #     for i in range(self.L-1):
    #         temp.append(self.build_TFI_1d(tf.concat([self.x[:i], 
    #                 tf.transpose(tf.matmul(tf.convert_to_tensor([[1.,0.], [0., -1.]]) ,(self.x[i])[:, tf.newaxis])),
    #                 tf.transpose(tf.matmul(tf.convert_to_tensor([[1.,0.], [0., -1.]]) ,(self.x[i+1])[:, tf.newaxis])),
    #                 self.x[i+2:]], axis=0)))
            
    #         num_temp = temp[i][0][0]
    #         self.localE -= self.h * num_temp / num_oldAmp
    #         self.loss -= (temp[i] - oldAmp)

    #     temp.append(self.build_TFI_1d(tf.concat([
    #             tf.transpose(tf.matmul(tf.convert_to_tensor([[1.,0.], [0., -1.]]) ,(self.x[0])[:, tf.newaxis])),
    #             self.x[1:(self.L-1)], 
    #             tf.transpose(tf.matmul(tf.convert_to_tensor([[1.,0.], [0., -1.]]) ,(self.x[self.L-1])[:, tf.newaxis])),
    #             ], axis=0)))
        
    #     num_temp = temp[self.L-1][0][0]
    #     self.localE -= self.h * num_temp / num_oldAmp
    #     self.loss -= (temp[i] - oldAmp)
    #     #########################################
        
        
        
    #     tempAmp = []
    #     for i in range(self.L):
    #         # tempAmp = self.sess.run(self.amp, feed_dict={self.x: tempConfig})
    #         # self.x[i] = ((self.x[i]*2-1)*(-1)+1)/2
    #         tempAmp.append(self.build_TFI_1d(tf.concat([self.x[:i], 
    #                                                   (1.-self.x[i])[tf.newaxis, :],
    #                                                   self.x[i+1:]], axis=0)))
            
    #         num_tempAmp = tempAmp[i][0][0]
    #         self.localE -= self.h * num_tempAmp / num_oldAmp
    #         self.loss -= (tempAmp[i] - oldAmp)
    #     '''
    #     temp = []

    #     for i in range(self.L-1):
    #         temp.append(self.build_TFI_1d(self.j * tf.concat([self.x[:i], 
    #                 tf.transpose(tf.matmul(tf.convert_to_tensor([[1.,0.], [0., -1.]]) ,(self.x[i])[:, tf.newaxis])),
    #                 tf.transpose(tf.matmul(tf.convert_to_tensor([[1.,0.], [0., -1.]]) ,(self.x[i+1])[:, tf.newaxis])),
    #                 self.x[i+2:]], axis=0) + 
    #                                     self.h * tf.concat([self.x[:i], 
    #                                             (1.-self.x[i])[tf.newaxis, :],
    #                                             self.x[i+1:]], axis=0)))
            
    #         num_temp = temp[i][0][0]
    #         self.localE -= num_temp / num_oldAmp
    #         self.loss -= (temp[i] - oldAmp)

    #     temp.append(self.build_TFI_1d(tf.concat([
    #             tf.transpose(tf.matmul(tf.convert_to_tensor([[1.,0.], [0., -1.]]) ,(self.x[0])[:, tf.newaxis])),
    #             self.x[1:(self.L-1)], 
    #             tf.transpose(tf.matmul(tf.convert_to_tensor([[1.,0.], [0., -1.]]) ,(self.x[self.L-1])[:, tf.newaxis])),
    #             ], axis=0) +
    #                                 tf.concat([self.x[:self.L-1], 
    #                                         (1.-self.x[self.L-1])[tf.newaxis, :],
    #                                         self.x[self.L:]], axis=0)))
        
    #     num_temp = temp[self.L-1][0][0]
    #     self.localE -= num_temp / num_oldAmp
    #     self.loss -= (temp[i] - oldAmp)
    

    def local_E_heisenberg(self):
        """
        H = J sigma_i sigma_{i+1}
        """
        L, channel = self.x.shape
        self.localE = tf.constant(0.)
        self.loss = tf.constant(0.)

        oldAmp = self.amp
        num_oldAmp = oldAmp[0][0]
        
        temp = []

        for i in range(self.L-1):
            temp.append(self.build_TFI_1d(tf.concat([self.x[:i], 
                    tf.transpose(tf.matmul(tf.convert_to_tensor([[1.,0.], [0., -1.]]) ,(self.x[i])[:, tf.newaxis])),
                    tf.transpose(tf.matmul(tf.convert_to_tensor([[1.,0.], [0., -1.]]) ,(self.x[i+1])[:, tf.newaxis])),
                    self.x[i+2:]], axis=0) + 
                                          tf.concat([self.x[:i], 
                    tf.transpose(tf.matmul(tf.convert_to_tensor([[0.,1.], [1., 0.]]) ,(self.x[i])[:, tf.newaxis])),
                    tf.transpose(tf.matmul(tf.convert_to_tensor([[0.,1.], [1., 0.]]) ,(self.x[i+1])[:, tf.newaxis])),
                    self.x[i+2:]], axis=0) -
                                          tf.concat([self.x[:i], 
                    tf.transpose(tf.matmul(tf.convert_to_tensor([[0.,-1.], [1., 0.]]) ,(self.x[i])[:, tf.newaxis])),
                    tf.transpose(tf.matmul(tf.convert_to_tensor([[0.,-1.], [1., 0.]]) ,(self.x[i+1])[:, tf.newaxis])),
                    self.x[i+2:]], axis=0)
                                        ))
            
            num_temp = temp[i][0][0]
            self.localE -= self.j * num_temp / num_oldAmp
            self.loss -= (temp[i] - oldAmp)

        temp.append(self.build_TFI_1d( tf.concat([
                    tf.transpose(tf.matmul(tf.convert_to_tensor([[1.,0.], [0., -1.]]) ,(self.x[0])[:, tf.newaxis])),
                    self.x[1:self.L-1], 
                    tf.transpose(tf.matmul(tf.convert_to_tensor([[1.,0.], [0., -1.]]) ,(self.x[self.L-1])[:, tf.newaxis])),
                    ], axis=0) + 
                                          tf.concat([
                    tf.transpose(tf.matmul(tf.convert_to_tensor([[0.,1.], [1., 0.]]) ,(self.x[0])[:, tf.newaxis])),                          
                    self.x[1:self.L-1], 
                    tf.transpose(tf.matmul(tf.convert_to_tensor([[0.,1.], [1., 0.]]) ,(self.x[self.L-1])[:, tf.newaxis])),
                    ], axis=0) -
                                          tf.concat([
                    tf.transpose(tf.matmul(tf.convert_to_tensor([[0.,-1.], [1., 0.]]) ,(self.x[0])[:, tf.newaxis])),                          
                    self.x[1:self.L-1], 
                    tf.transpose(tf.matmul(tf.convert_to_tensor([[0.,-1.], [1., 0.]]) ,(self.x[self.L-1])[:, tf.newaxis])),
                    ], axis=0)
                    ))
        
        num_temp = temp[self.L-1][0][0]
        self.localE -= self.j * num_temp / num_oldAmp
        self.loss -= (temp[i] - oldAmp)

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
            R = self.L
        else:
            R = -1
            # localE = oldlocalE
            # self.config = oldconfig
            self.sess.run(self.train_op, feed_dict={self.x: oldconfig})
        Loss.append(self.sess.run(self.loss, feed_dict={self.x: oldconfig}))
        return Loss, localE, R

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
        self.out1 = tf.nn.sigmoid(self.ip1, name = layer_name + '_activations')

        # fc2 
        layer_name = 'fc2'; dim = 256; hiddens = 64
        self.w2 = tf.Variable(tf.random_normal([dim, hiddens]), name = layer_name + '_weights')
        self.b2 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name = layer_name + '_biases')
        self.ip2 = tf.add(tf.matmul(self.out1, self.w2), self.b2, name = layer_name + '_ip1')
        self.out2 = tf.nn.sigmoid(self.ip2, name = layer_name + '_activations')

        # fc3
        layer_name = 'fc3'; dim = 64; hiddens = 1
        self.w3 = tf.Variable(tf.random_normal([dim, hiddens]), name = layer_name + '_weights')
        self.b3 = tf.Variable(tf.constant(0.1, shape=[hiddens]), name = layer_name + '_biases')
        self.ip3 = tf.add(tf.matmul(self.out2, self.w3), self.b3, name = layer_name + '_ip1')
        self.out3 = tf.nn.sigmoid(self.ip3, name = layer_name + '_outputs')
        self.y = tf.exp(self.out3)

        return self.y


def main():
    ising = TFI_1d(L=L, channel=CHANNEL, learning_rate=LR, model_name='heisenberg', j=J, h=H)
    
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
        Loss, localE, R = ising.get_env_feedback(S, A, Loss)
        
        q_target = R
        q_table.iloc[0, A] += ALPHA * q_target 
        # for _ in range(L):
        #     S = (S+1) % L
        print('##########################')
        print('episode: %d' % episode)
        ising.update_env(episode)
        print()
        print('localE: ', localE/L)
        print('loss: ', Loss[-1])
        
        if episode %10 == 0:
            input()
        # input()


def exact_solution():
    from itertools import product
    N = 5
    basis = list(product([-1,1],repeat=N))

    print('Generated %d basis functions' % (len(basis)))
    #print(len(basis_functions))

    #list(permutations([0,1,0,0]))
    H = np.zeros((2**N,2**N))
    for H_i in range(2**N):
        for H_j in range(2**N):
            H_sum = 0
            for i in range(N):
                if H_i == H_j:
                    if i == N-1:
                        H_sum -= basis[H_j][i]*basis[H_j][0]
                    else:
                        H_sum -= basis[H_j][i]*basis[H_j][i+1]
                        
                sj = list(basis[H_j])
                sj[i] *= -1
                if H_i == basis.index(tuple(sj)):
                    H_sum -= 2

            H[H_i,H_j] = H_sum
                
    print('Ground state energy:', np.min(np.linalg.eigvals(H))/N)


if __name__ == '__main__':
    main()
    # exact_solution()



