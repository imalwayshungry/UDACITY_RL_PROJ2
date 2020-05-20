import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from learn_model import *
from utils import *


class DDPGagent:            #**well trim the learning rate as we get better!
    def __init__(self, load_modelz, model_list, env, hidden_size=48, hidden_size2=32, actor_learning_rate=.0005, critic_learning_rate=.0005, gamma=0.99, tau=1e-2,
                 max_memory_size=50000):
        # Params
        self.num_states = env.vector_observations.size
        self.num_states = 33
        #self.num_actions = env.previous_vector_actions.shape[1]
        self.num_actions = env.previous_vector_actions.size
        self.num_actions = 4
        self.gamma = gamma
        self.tau = tau
        self.alr = actor_learning_rate
        self.clr = critic_learning_rate

        # Networks
        self.actor = Actor(self.num_states, hidden_size, hidden_size2, self.num_actions)      #**WE HAVE 4 TOTAL MODELS WE NEED TO SAVE
        self.actor_target = Actor(self.num_states, hidden_size, hidden_size2, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, hidden_size2, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, hidden_size2, self.num_actions)

        if load_modelz:         #***LOAD STORED MODELS!!
            self.load_model_inner(model_list)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)


    def get_learning_rate(self): #**main module can get current learning ratez!
        return self.alr, self.clr

    def update_learning_rate(self, alr, clr): #**well adjust the learning rate as we learn more stuff!
        self.alr = alr
        self.clr = clr
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = self.alr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = self.clr
        print("UPDATED LEARNING RATE: GREAT SUCCESS!")


    def load_model_inner(self, name_list):
        self.actor.load_state_dict(torch.load(name_list[0]))
        self.actor_target.load_state_dict(torch.load(name_list[1]))
        self.critic.load_state_dict(torch.load(name_list[2]))
        self.critic_target.load_state_dict(torch.load(name_list[3]))
        print("LOADED MODELS FROM DISK")

    def save_models(self):
        rand_ext = str(random.randint(0, 9999999))
        self.save_models_inner(self.actor, "actor", rand_ext)
        self.save_models_inner(self.actor_target, "actor_target", rand_ext)
        self.save_models_inner(self.critic, "critic", rand_ext)
        self.save_models_inner(self.critic_target, "critic_target", rand_ext)

    def save_models_inner(self, model_reference, model_type, rand_ext):
        """Initialize an Agent object.

                        Params
                        ======
                            Save:

                                torch.save(model.state_dict(), PATH)
                            Load:

                                model = TheModelClass(*args, **kwargs)
                                model.load_state_dict(torch.load(PATH))
                                model.eval()
                        """
        import random
        file_name = "MODEL_CHECKPOINT."
        file_name = file_name + rand_ext + "." + model_type + ".pt"
        torch.save(model_reference.state_dict(), file_name)
        print("Model Saved: " + file_name)

    def get_action(self, state, train_model):
        if train_model == False:
            with torch.no_grad():
                state = Variable(torch.from_numpy(state).float().unsqueeze(0))
                action = self.actor.forward(state)
        if train_model:
            with torch.no_grad():
                state = Variable(torch.from_numpy(state).float().unsqueeze(0))
                action = self.actor.forward(state)
        #action = action.detach().numpy()[0, 0]
        action = action.detach().numpy()
        return action

    def update(self, batch_size, train_model):
        if not train_model:  #***Want to make sure we have the ability to play without trianing
            return
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime.detach())

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))