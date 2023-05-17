
import numpy as np
import torch as T
import torch.nn.functional as F
from Classes.networks import ActorNetwork, CriticNetwork

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma, C_fc1_dims, C_fc2_dims, A_fc1_dims,
                 A_fc2_dims, batch_size, n_agents, agent_name, noise):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.number_agents = n_agents
        self.number_actions = n_actions
        self.number_states = input_dims
        self.agent_name = agent_name
        self.noise = noise

        self.actor = ActorNetwork(alpha, input_dims, A_fc1_dims, A_fc2_dims, n_agents,
                                n_actions=n_actions, name='actor', agent_label=agent_name)
        self.critic_task1 = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, n_agents,
                                n_actions=n_actions, name='critic_task1', agent_label=agent_name)

        self.critic_task2 = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, n_agents,
                                          n_actions=n_actions, name='critic_task2', agent_label=agent_name)

        self.target_actor = ActorNetwork(alpha, input_dims, A_fc1_dims, A_fc2_dims, n_agents,
                                n_actions=n_actions, name='target_actor', agent_label=agent_name)

        self.target_critic_task1 = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, n_agents,
                                n_actions=n_actions, name='target_critic_task1', agent_label=agent_name)

        self.target_critic_task2 = CriticNetwork(beta, input_dims, C_fc1_dims, C_fc2_dims, n_agents,
                                                 n_actions=n_actions, name='target_critic_task2', agent_label=agent_name)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        # print('check this variable for convergence!!! : ', mu)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise, size=self.number_actions),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_task1.save_checkpoint()
        self.critic_task2.save_checkpoint()
        self.target_critic_task1.save_checkpoint()
        self.target_critic_task2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_task1.load_checkpoint()
        self.critic_task2.load_checkpoint()
        self.target_critic_task1.load_checkpoint()
        self.target_critic_task2.load_checkpoint()

    def local_learn(self, global_loss, state, action, reward_t1, reward_t2, state_, terminal):

        states = state
        states_ = state_
        actions = action
        rewards_t1 = reward_t1
        rewards_t2 = reward_t2
        done = terminal
        self.global_loss = global_loss

        self.target_actor.eval()
        self.target_critic_task1.eval()
        self.target_critic_task2.eval()
        self.critic_task1.eval()
        self.critic_task2.eval()

        target_actions = self.target_actor.forward(states_).clone().detach()
        critic_value_task1_ = self.target_critic_task1.forward(states_, target_actions)
        critic_value_task1 = self.critic_task1.forward(states, actions)

        critic_value_task1_[done] = 0.0
        critic_value_task1_ = critic_value_task1_.view(-1)

        target_task1 = rewards_t1 + self.gamma*critic_value_task1_
        target_task1 = target_task1.view(self.batch_size, 1)
        self.critic_task1.train()
        self.critic_task1.optimizer.zero_grad()
        critic_loss_task1 = F.mse_loss(target_task1, critic_value_task1)
        critic_loss_task1.backward()
        self.critic_task1.optimizer.step()
        self.critic_task1.eval()

        critic_value_task2_ = self.target_critic_task2.forward(states_, target_actions)
        critic_value_task2 = self.critic_task2.forward(states, actions)

        critic_value_task2_[done] = 0.0
        critic_value_task2_ = critic_value_task2_.view(-1)

        target_task2 = rewards_t2 + self.gamma*critic_value_task2_
        target_task2 = target_task2.view(self.batch_size, 1)

        self.critic_task2.train()
        self.critic_task2.optimizer.zero_grad()
        critic_loss_task2 = F.mse_loss(target_task2, critic_value_task2)
        critic_loss_task2.backward()
        self.critic_task2.optimizer.step()
        self.critic_task2.eval()
        # print('loss for global critic: ',self.global_loss.mean())
        # print('loss for critic 1: ',-self.critic_task1.forward(states, self.actor.forward(states)).mean())
        # print('loss for critic 2: ',-self.critic_task2.forward(states, self.actor.forward(states)).mean())

        self.actor.optimizer.zero_grad()
        self.actor.train()
        actor_loss = -self.critic_task1.forward(states, self.actor.forward(states))-self.critic_task2.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss) + (T.mean(self.global_loss)*2)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params_task1 = self.critic_task1.named_parameters()
        critic_params_task2 = self.critic_task2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params_task1 = self.target_critic_task1.named_parameters()
        target_critic_params_task2 = self.target_critic_task2.named_parameters()

        critic_state_dict_task1 = dict(critic_params_task1)
        critic_state_dict_task2 = dict(critic_params_task2)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict_task1 = dict(target_critic_params_task1)
        target_critic_state_dict_task2 = dict(target_critic_params_task2)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict_task2:
            critic_state_dict_task2[name] = tau*critic_state_dict_task2[name].clone() + \
                                (1-tau)*target_critic_state_dict_task2[name].clone()

        for name in critic_state_dict_task1:
            critic_state_dict_task1[name] = tau*critic_state_dict_task1[name].clone() + \
                                (1-tau)*target_critic_state_dict_task1[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic_task1.load_state_dict(critic_state_dict_task1)
        self.target_critic_task2.load_state_dict(critic_state_dict_task2)
        self.target_actor.load_state_dict(actor_state_dict)