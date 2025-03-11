


import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

#---Ornstein-Uhlenbeck Noise for action---#

# class OUNoise(object):
#     def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.99, min_sigma=0.01, decay_period=1000000):
#         self.mu           = mu
#         self.theta        = theta
#         self.sigma        = max_sigma
#         self.max_sigma    = max_sigma
#         self.min_sigma    = min_sigma
#         self.decay_period = decay_period
#         self.action_dim   = action_space
#         self.reset()
        
#     def reset(self):
#         self.state = np.ones(self.action_dim) * self.mu
        
#     def evolve_state(self):
#         x  = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
#         self.state = x + dx
#         return self.state
    
#     def get_noise(self, t=0): 
#         ou_state = self.evolve_state()
#         decaying = float(float(t)/ self.decay_period)
#         self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
#         return ou_state
class OUNoise (object):
    
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

# EPS = 0.01
# def fanin_init(size, fanin=None):
#     fanin = fanin or size[0]
#     v = 1./np.sqrt(fanin)
#     return torch.Tensor(size).uniform_(-v,v)

    
    
    
    
    
    #connect state,goal, action information at the first layer of the network.
    
# class Critic(nn.Module):
#     def __init__(self, state_dim, goal_dim, action_dim):
#         super(Critic, self).__init__()
        
#         # self.state_dim = state_dim = state_dim
#         # self.action_dim = action_dim
#         # self.goal_dim=goal_dim
        
#         self.fc1 = nn.Linear(state_dim+goal_dim+action_dim, 250)
#         nn.init.xavier_uniform_(self.fc1.weight)
#         self.fc1.bias.data.fill_(0.01)
        
#         self.fc2 = nn.Linear(250,750)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         self.fc2.bias.data.fill_(0.01)

        
#         self.fc3 = nn.Linear(750, 1)
#         nn.init.xavier_uniform_(self.fc3.weight)
#         self.fc3.bias.data.fill_(0.01)

#     def forward(self, state, goal, action):
#         sa = torch.cat([state, goal, action], 1)
#         q = F.relu(self.fc1(sa))
#         q = F.relu(self.fc2(q))
#         q = self.fc3(q)

#         return q

class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim):
        super(Critic, self).__init__()
        
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim=goal_dim
        
        self.fc1 = nn.Linear(state_dim, 125)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        # self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        
        self.fs1 = nn.Linear(goal_dim, 125)
        nn.init.xavier_uniform_(self.fs1.weight)
        self.fs1.bias.data.fill_(0.01)
        
        self.fa1 = nn.Linear(action_dim, 125)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())
        
        self.fca1 = nn.Linear(375, 375)
        nn.init.xavier_uniform_(self.fca1.weight)
        self.fca1.bias.data.fill_(0.01)
        # self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        
        self.fca2 = nn.Linear(375, 1)
        nn.init.xavier_uniform_(self.fca2.weight)
        self.fca2.bias.data.fill_(0.01)
        # self.fca2.weight.data.uniform_(-EPS, EPS)
        
    def forward(self, state, sub_goal,action):
        xs = torch.relu(self.fc1(state))
        xg = torch.relu(self.fs1(sub_goal))
        xa = torch.relu(self.fa1(action))
        x = torch.cat((xs,xg,xa), dim=1)
        x = torch.relu(self.fca1(x))
        vs = self.fca2(x)
        return vs
#---Actor---#

# class TD3Critic(nn.Module):
#     def __init__(self, state_dim, goal_dim, action_dim):
#         super(TD3Critic, self).__init__()
#         # Q1
#         self.l1 = nn.Linear(state_dim + goal_dim + action_dim, 300)
#         self.l2 = nn.Linear(300, 300)
#         self.l3 = nn.Linear(300, 1)
#         # Q2
#         self.l4 = nn.Linear(state_dim + goal_dim + action_dim, 300)
#         self.l5 = nn.Linear(300, 300)
#         self.l6 = nn.Linear(300, 1)

#     def forward(self, state, goal, action):
#         sa = torch.cat([state, goal, action], 1)

#         q = F.relu(self.l1(sa))
#         q = F.relu(self.l2(q))
#         q = self.l3(q)

#         return q

# class TD3Actor(nn.Module):
#     def __init__(self, state_dim, goal_dim, action_dim, scale=None):
#         super(TD3Actor, self).__init__()
#         if scale is None:
#             scale = torch.ones(state_dim)
#         else:
#             scale = get_tensor(scale)
#         self.scale = nn.Parameter(scale.clone().detach().float(), requires_grad=False)

#         self.l1 = nn.Linear(state_dim + goal_dim, 300)
#         self.l2 = nn.Linear(300, 300)
#         self.l3 = nn.Linear(300, action_dim)

#     def forward(self, state, goal):
#         a = F.relu(self.l1(torch.cat([state, goal], 1)))
#         a = F.relu(self.l2(a))
#         return self.scale * torch.tanh(self.l3(a))






class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim,action_dim,action_limit_v, action_limit_w):
        super(Actor, self).__init__()
        self.state_dim = state_dim 
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        
        self.fa1 = nn.Linear(state_dim + goal_dim, 250)
        nn.init.xavier_uniform_(self.fa1.weight)
        self.fa1.bias.data.fill_(0.01)
        # self.fa1.weight.data.uniform_(-EPS, EPS)
        # self.fa1.bias.data.uniform_(-EPS, EPS)
        
        self.fa2 = nn.Linear(250, 250)
        nn.init.xavier_uniform_(self.fa2.weight)
        self.fa2.bias.data.fill_(0.01)
        # self.fa2.weight.data.uniform_(-EPS, EPS)
        # self.fa2.bias.data.uniform_(-EPS, EPS)
        
        self.fa3 = nn.Linear(250, action_dim)
        nn.init.xavier_uniform_(self.fa3.weight)
        self.fa3.bias.data.fill_(0.01)
        # self.fa3.weight.data.uniform_(-EPS, EPS)
        # self.fa3.bias.data.uniform_(-EPS, EPS)
        
    def forward(self, state,goal):
        

        x = torch.relu(self.fa1(torch.cat([state, goal], 1)))
        x = torch.relu(self.fa2(x))
        action = self.fa3(x)
        if len(action.shape)>=2:
            action[:,0] = torch.sigmoid(action[:,0])*self.action_limit_v
            action[:,1] = torch.tanh(action[:,1])*self.action_limit_w
        elif len(action.shape)==1 :
            action[0] = torch.sigmoid(action[0])*self.action_limit_v
            action[1] = torch.tanh(action[1])*self.action_limit_w
        return action
    
# class HighActor(nn.Module):
#     #input:robot position+final goal,output: subgoal?
#         def __init__(self, current_position_dim, final_goal_dim,subgoal_dim,scale):
#             super(HighActor, self).__init__()
#             self.input_dim = current_position_dim+final_goal_dim
#             self.output_dim = subgoal_dim
#             self.scale=scale
#             self.fa1 = nn.Linear(self.input_dim, 250)
#             nn.init.xavier_uniform_(self.fa1.weight)
#             self.fa1.bias.data.fill_(0.01)
#             # self.fa1.weight.data.uniform_(-EPS, EPS)
#             # self.fa1.bias.data.uniform_(-EPS, EPS)
            
#             self.fa2 = nn.Linear(250, 250)
#             nn.init.xavier_uniform_(self.fa2.weight)
#             self.fa2.bias.data.fill_(0.01)
#             # self.fa2.weight.data.uniform_(-EPS, EPS)
#             # self.fa2.bias.data.uniform_(-EPS, EPS)
            
#             self.fa3 = nn.Linear(250, self.output_dim)
#             nn.init.xavier_uniform_(self.fa3.weight)
#             self.fa3.bias.data.fill_(0.01)
#             # self.fa3.weight.data.uniform_(-EPS, EPS)
#             # self.fa3.bias.data.uniform_(-EPS, EPS)
            
#         def forward(self, current_pos, final_goal_pos):
#             x = torch.cat([current_pos, final_goal_pos], 1)  # Concatenate inputs
#             # x = torch.relu(self.fa1(state))
#             x = torch.relu(self.fa1(x))
#             x = torch.relu(self.fa2(x))
#             subgoal = self.fa3(x)
#             if len(current_pos.shape)>=2:
#                 subgoal[:,0] = torch.tanh(subgoal[:,0])*self.scale
#                 subgoal[:,1] = torch.tanh(subgoal[:,1])*self.scale
#             else:
#                 subgoal[0] = torch.tanh(subgoal[0])*self.scale
#                 subgoal[1] = torch.tanh(subgoal[1])*self.scale
#             return subgoal
        
class HighActor(nn.Module):
    #input:robot position+final goal,output: subgoal?
        def __init__(self, current_position_dim, final_goal_dim,subgoal_dim):
            super(HighActor, self).__init__()
            self.input_dim = current_position_dim+final_goal_dim
            self.output_dim = subgoal_dim
            self.scale=0.5
            self.fa1 = nn.Linear(self.input_dim, 250)
            nn.init.xavier_uniform_(self.fa1.weight)
            self.fa1.bias.data.fill_(0.01)
            # self.fa1.weight.data.uniform_(-EPS, EPS)
            # self.fa1.bias.data.uniform_(-EPS, EPS)
            
            self.fa2 = nn.Linear(250, 250)
            nn.init.xavier_uniform_(self.fa2.weight)
            self.fa2.bias.data.fill_(0.01)
            # self.fa2.weight.data.uniform_(-EPS, EPS)
            # self.fa2.bias.data.uniform_(-EPS, EPS)
            
            self.fa3 = nn.Linear(250, self.output_dim)
            nn.init.xavier_uniform_(self.fa3.weight)
            self.fa3.bias.data.fill_(0.01)
            # self.fa3.weight.data.uniform_(-EPS, EPS)
            # self.fa3.bias.data.uniform_(-EPS, EPS)
            
        def forward(self, current_pos, final_goal_pos):
            x = torch.cat([current_pos, final_goal_pos], 1)  # Concatenate inputs
            # x = torch.relu(self.fa1(state))
            x = torch.relu(self.fa1(x))
            x = torch.relu(self.fa2(x))
            subgoal = self.fa3(x)
            # x_position=torch.full((256,),current_pos[0])
            # y_position=torch.full((256,),current_pos[1])
            if len(current_pos.shape)>=2:
                
                subgoal[:,0] = torch.tanh(subgoal[:,0])*self.scale+current_pos[:,0]
                subgoal[:,1] = torch.tanh(subgoal[:,1])*self.scale+current_pos[:,1]
                
            else:
                subgoal[0] = torch.tanh(subgoal[0])*self.scale+current_pos[:,0]
                subgoal[1] = torch.tanh(subgoal[1])*self.scale+current_pos[:,1]
            subgoal=torch.clamp(subgoal,min=-4.5, max=4.5)

            return subgoal
               
class HighCritic(nn.Module):
    #evaluate the value of state and subgoal
    def __init__(self, current_positon_dim, final_goal_dim, sub_goal_dim):
        super(HighCritic, self).__init__()
        
        # self.state_dim = state_dim = state_dim
        # self.action_dim = action_dim
        # self.goal_dim=goal_dim
        
        self.fc1 = nn.Linear(current_positon_dim+final_goal_dim+sub_goal_dim, 250)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        
        self.fc2 = nn.Linear(250,250)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)

        
        self.fc3 = nn.Linear(250, 1)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

    def forward(self, current_positon, final_goal_position, sub_goal):
        sa = torch.cat([current_positon, final_goal_position, sub_goal], 1)
        q = F.relu(self.fc1(sa))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)

        return q       
        


# class HighActor(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(HighActor, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, current_pos, goal_pos):
#         x = torch.cat((current_pos, goal_pos), dim=-1)  # Concatenate inputs
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         subgoal = self.fc3(x)
#         return subgoal

# # Example usage
# input_dim = 4  # e.g., 2 for current_pos (x, y) and 2 for goal_pos (x, y)
# hidden_dim = 128
# output_dim = 2  # e.g., 2 for subgoal (x, y)

# model = HighLevelPolicyNetwork(input_dim, hidden_dim, output_dim)

# # Example input
# current_pos = torch.tensor([[1.0, 2.0]])
# goal_pos = torch.tensor([[5.0, 6.0]])

# subgoal = model(current_pos, goal_pos)
# print(subgoal)
       
        
        
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.99, min_sigma=0.2, decay_period=1000000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_noise(self, t=0): 
        ou_state = self.evolve_state()
        decaying = float(float(t)/ self.decay_period)
        self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
        return ou_state