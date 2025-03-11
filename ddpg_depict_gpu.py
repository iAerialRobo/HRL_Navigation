#!/usr/bin/env python
# Authors: Junior Costa de Jesus #

import rospy
import os
import json
import numpy as np
import random
import numpy.random as nr
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32
from scripts.environment_stage_1 import Env
import torch
import torch.nn.functional as F
import gc
import torch.nn as nn
import math
from collections import deque
import copy
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#---Directory Path---#
dir = os.path.dirname(os.path.realpath(__file__))
#---Functions to make network updates---#
 
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data*(1.0 - tau)+ param.data*tau)

def hard_update(target,source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

#---Ornstein-Uhlenbeck Noise for action---#

# class OUNoise(object):
    # def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.99, min_sigma=0.01, decay_period= 600000):
    #     self.mu           = mu
    #     self.theta        = theta
    #     self.sigma        = max_sigma
    #     self.max_sigma    = max_sigma
    #     self.min_sigma    = min_sigma
    #     self.decay_period = decay_period
    #     self.action_dim   = action_space
    #     self.reset()
        
    # def reset(self):
    #     self.state = np.ones(self.action_dim) * self.mu
        
    # def evolve_state(self):
    #     x  = self.state
    #     dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
    #     self.state = x + dx
    #     return self.state
    
    # def get_noise(self, t=0): 
    #     ou_state = self.evolve_state()
    #     # print('noise' + str(ou_state))
    #     decaying = float(float(t)/ self.decay_period)
    #     self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
    #     return ou_state
class OUNoise (object):
    # def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.99, min_sigma=0.01, decay_period= 600000):
    #     self.mu           = mu
    #     self.theta        = theta
    #     self.sigma        = max_sigma
    #     self.max_sigma    = max_sigma
    #     self.min_sigma    = min_sigma
    #     self.decay_period = decay_period
    #     self.action_dim   = action_space
    #     self.reset()
        
    # def reset(self):
    #     self.state = np.ones(self.action_dim) * self.mu
        
    # def evolve_state(self):
    #     x  = self.state
    #     dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
    #     self.state = x + dx
    #     return self.state
    
    # def get_noise(self, t=0): 
    #     ou_state = self.evolve_state()
    #     # print('noise' + str(ou_state))
    #     decaying = float(float(t)/ self.decay_period)
    #     self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
    #     return ou_state
    
        def __init__(self,action_dimension,mu=0, theta=0.15, sigma=0.1):
            self.action_dimension = action_dimension
            self.mu = mu
            self.theta = theta
            self.sigma = sigma
            self.state = np.ones(self.action_dimension) * self.mu
            self.reset()

        def reset(self):
            self.state = np.ones(self.action_dimension) * self.mu

        def get_noise(self):
            x = self.state
            dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
            self.state = x + dx
            return self.state
#---Critic--#

EPS = 0.003
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1./np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v,v)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(state_dim, 125)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        
        self.fa1 = nn.Linear(action_dim, 125)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())
        
        self.fca1 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fca1.weight)
        self.fca1.bias.data.fill_(0.01)
        # self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        
        self.fca2 = nn.Linear(250, 1)
        nn.init.xavier_uniform_(self.fca2.weight)
        self.fca2.bias.data.fill_(0.01)
        # self.fca2.weight.data.uniform_(-EPS, EPS)
        
    def forward(self, state, action):
        xs = torch.relu(self.fc1(state))
        xa = torch.relu(self.fa1(action))
        x = torch.cat((xs,xa), dim=1)
        x = torch.relu(self.fca1(x))
        vs = self.fca2(x)
        return vs

#---Actor---#

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w):
        super(Actor, self).__init__()
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        
        self.fa1 = nn.Linear(state_dim, 250)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())
        
        self.fa2 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0.01)
        # self.fa2.weight.data = fanin_init(self.fa2.weight.data.size())
        
        self.fa3 = nn.Linear(250, action_dim)
        nn.init.xavier_uniform_(self.fa3.weight)
        self.fa3.bias.data.fill_(0.01)
        # self.fa3.weight.data.uniform_(-EPS,EPS)
        
    def forward(self, state):
        x = torch.relu(self.fa1(state))
        x = torch.relu(self.fa2(x))
        action = self.fa3(x)
        if state.shape <= torch.Size([self.state_dim]):
            action[0] = torch.sigmoid(action[0])*self.action_limit_v
            action[1] = torch.tanh(action[1])*self.action_limit_w
        else:
            action[:,0] = torch.sigmoid(action[:,0])*self.action_limit_v
            action[:,1] = torch.tanh(action[:,1])*self.action_limit_w
        return action

#---Memory Buffer---#

class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0
        
    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)
        
        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])
        done_array = np.float32([array[4] for array in batch])
        
        return s_array, a_array, r_array, new_s_array, done_array
    
    def len(self):
        return self.len
    
    def add(self, s, a, r, new_s, done):
        transition = (s, a, r, new_s, done)
        self.len += 1 
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)

#---Where the train is made---#

BATCH_SIZE = 256
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.005
load_models=True
class Trainer:
    
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w, ram):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        #print('w',self.action_limit_w)
        self.ram = ram
        #self.iter = 0 

        
        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w).to(device)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)
        
        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.target_critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)
        self.pub_qvalue = rospy.Publisher('qvalue', Float32, queue_size=5)
        self.qvalue = Float32()
        
        if load_models:
                checkpoint = torch.load("/home/wendy/RLprojects/ros_ddpg/src/turtlebot3_ddpg/scripts/Models/stage2/ddpg/730a.pt")
                print(checkpoint.keys())
                print(checkpoint['epoch'])
                print(checkpoint)
                self.target_critic.load_state_dict(checkpoint['target_critic'])
                self.critic.load_state_dict(checkpoint['critic'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                self.target_actor.load_state_dict(checkpoint['target_actor'])
                self.actor.load_state_dict(checkpoint['actor'])
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                # self.start_epoch = checkpoint['epoch']
                # self.sigma=checkpoint["sigma"]
                # self.noise= OUNoise(action_dim,max_sigma=self.sigma)
                rospy.loginfo("loadmodel")
        else:
                hard_update(self.target_actor, self.actor)
                hard_update(self.target_critic, self.critic)
        
    def get_exploitation_action(self,state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        #print('actionploi', action)
        return action.data.numpy()
        
    def get_exploration_action(self, state):
        state = torch.from_numpy(state).to(device)
        action = self.actor.forward(state).detach().cpu()
        #noise = self.noise.sample()
        #print('noisea', noise)
        #noise[0] = noise[0]*self.action_limit_v
        #noise[1] = noise[1]*self.action_limit_w
        #print('noise', noise)
        new_action = action.data.numpy() #+ noise
        #print('action_no', new_action)
        return new_action
    
    
    def optimizer(self):
        s_sample, a_sample, r_sample, new_s_sample, done_sample = ram.sample(BATCH_SIZE)
        
        s_sample = torch.from_numpy(s_sample).to(device)
        a_sample = torch.from_numpy(a_sample).to(device)
        r_sample = torch.from_numpy(r_sample)
        new_s_sample = torch.from_numpy(new_s_sample).to(device)
        done_sample = torch.from_numpy(done_sample)
        
        #-------------- optimize critic
        
        a_target = self.target_actor.forward(new_s_sample).detach()
        next_value = torch.squeeze(self.target_critic.forward(new_s_sample, a_target).detach().cpu())
        # y_exp = r _ gamma*Q'(s', P'(s'))
        y_expected = r_sample + (1 - done_sample)*GAMMA*next_value
        # y_pred = Q(s,a)
        y_predicted = torch.squeeze(self.critic.forward(s_sample, a_sample)).cpu()
        #-------Publisher of Vs------
        self.qvalue = y_predicted.detach()
        self.pub_qvalue.publish(torch.max(self.qvalue))
        #print(self.qvalue, torch.max(self.qvalue))
        #----------------------------
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected).to(device)

        
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()
        
        #------------ optimize actor
        pred_a_sample = self.actor.forward(s_sample)
        loss_actor = -1*torch.sum(self.critic.forward(s_sample, pred_a_sample)).to(device)
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        
        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)
    
    # def save_models(self, episode_count):

    #     torch.save(self.target_actor.state_dict(), dirPath +'/Models/' + world + '/' + str(episode_count)+ '_actor.pt')
    #     torch.save(self.target_critic.state_dict(), dirPath + '/Models/' + world + '/'+str(episode_count)+ '_critic.pt')
    #     print('****Models saved***')
    def save_model(self,dir,ep,sigma,rewards,total_sum_count,total_success_count,method):
        state = {'target_critic':trainer.target_critic.state_dict(),'critic':trainer.critic.state_dict(), 'critic_optimizer':trainer.critic_optimizer.state_dict(), 
                 'target_actor':trainer.target_actor.state_dict(),'actor':trainer.actor.state_dict(), 'actor_optimizer':trainer.actor_optimizer.state_dict(), 
                 'epoch':ep,'sigma':sigma,'rewards':rewards,'total_sum_count':total_sum_count,'total_success_count':total_success_count}
        torch.save(state,dir + '/Models/' + world + '/'+method+ '/'+str(ep)+"a.pt")
        
        
    def load_models(self, episode):
        self.actor.load_state_dict(torch.load(dir + '/Models/' + world + '/'+str(episode)+ '_actor.pt'))
        self.critic.load_state_dict(torch.load(dir + '/Models/' + world + '/'+str(episode)+ '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('***Models load***')

#---Mish Activation Function---#
def mish(x):
    '''
        Mish: A Self Regularized Non-Monotonic Neural Activation Function
        https://arxiv.org/abs/1908.08681v1
        implemented for PyTorch / FastAI by lessw2020
        https://github.com/lessw2020/mish
        param:
            x: output of a layer of a neural network
        return: mish activation function
    '''
    return x*(torch.tanh(F.softplus(x)))

#---Run agent---#

is_training = True

exploration_decay_rate = 0.001

MAX_EPISODES = 10001
MAX_STEPS = 300
MAX_BUFFER = 200000
rewards_all_episodes = []

STATE_DIMENSION = 14
ACTION_DIMENSION = 2
ACTION_V_MAX = 0.22 # m/s
ACTION_W_MAX = 2. # rad/s
world = 'stage2'

if is_training:
    var_v = ACTION_V_MAX*.5
    var_w = ACTION_W_MAX*2*.5
else:
    var_v = ACTION_V_MAX*0.10
    var_w = ACTION_W_MAX*0.10

print('State Dimensions: ' + str(STATE_DIMENSION))
print('Action Dimensions: ' + str(ACTION_DIMENSION))
print('Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad/s')
ram = MemoryBuffer(MAX_BUFFER)
trainer = Trainer(STATE_DIMENSION, ACTION_DIMENSION, ACTION_V_MAX, ACTION_W_MAX, ram)
noise = OUNoise(ACTION_DIMENSION)
#trainer.load_models(4880)



if __name__ == '__main__':
    rospy.init_node('ddpg_depict_gpu')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env(action_dim=ACTION_DIMENSION)
    before_training = 4

    past_action = np.zeros(ACTION_DIMENSION)
    stage_name="stage2"
    total_sum_count=0
    total_success_count=0
    method ="ddpg"

    start_epoch =0
    past_action = np.zeros(ACTION_DIMENSION)
    start_time =time.time()
    rewards=[]
    writer = SummaryWriter("/home/wendy/RLprojects/ros_ddpg/src/turtlebot3_ddpg/logs/ddpg")
    if  load_models:
        checkpoint = torch.load("/home/wendy/RLprojects/ros_ddpg/src/turtlebot3_ddpg/scripts/Models/stage2/ddpg/730a.pt")
        rewards=checkpoint['rewards']
        total_sum_count=checkpoint['total_sum_count']
        total_success_count=checkpoint['total_success_count']
        start_epoch=checkpoint['epoch']
        rospy.loginfo("loadmodel")
        noise.sigma= checkpoint['sigma'] 
    for ep in range(start_epoch+1,MAX_EPISODES):
        done = False
        state = env.reset()
        # if is_training and not ep%10 == 0 and ram.len >= before_training*MAX_STEPS:
        #     print('---------------------------------')
        #     print('Episode: ' + str(ep) + ' training')
        #     print('---------------------------------')
        # else:
        #     if ram.len >= before_training*MAX_STEPS:
        #         print('---------------------------------')
        #         print('Episode: ' + str(ep) + ' evaluating')
        #         print('---------------------------------')
        #     else:
        #         print('---------------------------------')
        #         print('Episode: ' + str(ep) + ' adding to memory')
        #         print('---------------------------------')

        rewards_current_episode = 0.

        for step in range(MAX_STEPS):
            state = np.float32(state)

            # if is_training and not ep%10 == 0 and ram.len >= before_training*MAX_STEPS:
            if is_training and not ep%10 == 0:
                action = trainer.get_exploration_action(state)
                # action[0] = np.clip(
                #     np.random.normal(action[0], var_v), 0., ACTION_V_MAX)
                # action[0] = np.clip(np.clip(
                #     action[0] + np.random.uniform(-var_v, var_v), action[0] - var_v, action[0] + var_v), 0., ACTION_V_MAX)
                # action[1] = np.clip(
                #     np.random.normal(action[1], var_w), -ACTION_W_MAX, ACTION_W_MAX)
                N = copy.deepcopy(noise.get_noise(t=step))
                N[0] = N[0]*ACTION_V_MAX/2
                N[1] = N[1]*ACTION_W_MAX
                action[0] = np.clip(action[0] + N[0], 0, ACTION_V_MAX)
                action[1] = np.clip(action[1] + N[1], -ACTION_W_MAX, ACTION_W_MAX)
            else:
                action = trainer.get_exploration_action(state)

            if not is_training:
                action = trainer.get_exploitation_action(state)
            next_state, reward, done,goal = env.step(action, past_action)
            # print('action', action,'r',reward)
            past_action = copy.deepcopy(action)
            
            rewards_current_episode += reward
            next_state = np.float32(next_state)
            if not ep%10 == 0 or not ram.len >= before_training*MAX_STEPS:
                if reward == 1000.:
                    print('***\n-------- Maximum Reward ----------\n****')
                    for _ in range(3):
                        ram.add(state, action, reward, next_state, done)
                else:
                    ram.add(state, action, reward, next_state, done)
            state = copy.deepcopy(next_state)
            
            if (done or goal):   
                # sum_count_episode=sum_count_episode+1
                total_sum_count=total_sum_count+1
                if goal:
                #   success_count_episode=success_count_episode+1
                    total_success_count=total_success_count+1
            

            # if ram.len >= before_training*MAX_STEPS and is_training and not ep%10 == 0:
            if ram.len>=BATCH_SIZE:

                trainer.optimizer()
            

            if done or step == MAX_STEPS-1: 
                if(total_sum_count==0):
                    total_success_rate=0
                else:
                    total_success_rate=total_success_count/total_sum_count
                m,s =divmod(int(time.time()- start_time),60)
                h,m =divmod(m,60)
                rewards.append(rewards_current_episode)
                # success_rates.append(success_rate_episode)
                writer.add_scalar("Total reward per episode_"+stage_name,rewards_current_episode,ep)
                rospy.loginfo('回合: %d 得分: %.2f 记忆量: %d sigma: %.2f 花费时间: %d:%02d:%02d,总成功率：%.3f',ep, rewards_current_episode, ram.len, noise.sigma, h, m, s,total_success_rate)
        
                break
        average_reward = np.average(rewards)
        writer.add_scalar("average Reward:"+stage_name,average_reward,ep)
        writer.add_scalar("total succes rate:"+stage_name,total_success_rate,ep) 
        if ep%10 == 0:
            trainer.save_model(dir,ep,noise.sigma,rewards,total_sum_count,total_success_count,method)


print('Completed Training')