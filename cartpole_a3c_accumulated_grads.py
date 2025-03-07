import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import torch.multiprocessing as mp


# local net - specific copy for each agent/worker
class LocalNet(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super().__init__()
        self.fc0 = nn.Linear(s_dim, h_dim)
        self.fc1_p = nn.Linear(h_dim, a_dim)
        self.fc1_v = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        state = torch.tensor(state)
        h1 = self.relu(self.fc0(state))
        out_p = self.softmax(self.fc1_p(h1))
        out_v = self.fc1_v(h1)
        return out_p, out_v


# local net - specific copy for each agent/worker
class LocalNet_deeper(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super().__init__()
        self.fc0 = nn.Linear(s_dim, h_dim)
        self.fc1_p = nn.Linear(h_dim, h_dim)
        self.fc2_p = nn.Linear(h_dim, a_dim)
        self.fc1_v = nn.Linear(h_dim, h_dim)
        self.fc2_v = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        state = torch.tensor(state)
        h1 = self.relu(self.fc0(state))
        h2_p = self.relu(self.fc1_p(h1))
        out_p = self.softmax(self.fc2_p(h2_p))
        h2_v = self.relu(self.fc1_v(h1))
        out_v = self.fc2_v(h2_v)
        return out_p, out_v


# global net - shared
class GlobalNet(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super().__init__()
        self.fc0 = nn.Linear(s_dim, h_dim)
        self.fc1_p = nn.Linear(h_dim, a_dim)
        self.fc1_v = nn.Linear(h_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        state = torch.tensor(state)
        h1 = self.relu(self.fc0(state))
        out_p = self.softmax(self.fc1_p(h1))
        out_v = self.fc1_v(h1)
        return out_p, out_v


# shared adam optimizer
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super().__init__(params, lr=lr)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


# worker
class Worker(mp.Process):
    def __init__(self, env_name, nS, nA, nH, lr, eps, gnet, optimizer, ep_return_list, episode_counter, name):
        super().__init__()
        # local members
        self.env = gym.make(env_name)
        self.lnet = LocalNet_deeper(nS, nH, nA)
        self.l_optimizer = torch.optim.Adam(lr=lr, params=self.lnet.parameters())
        self.eps = eps
        self.name = str(name)

        # global members
        self.gnet = gnet
        self.g_optimizer = optimizer
        self.ep_return_list = ep_return_list
        self.episode_counter = episode_counter

    def run(self):
        while self.episode_counter.value < max_episodes:

            with self.episode_counter.get_lock():
                self.episode_counter.value += 1

            # eps decay
            if self.eps % (max_episodes//10) == 0:
                if self.eps > 0:
                    self.eps -= 0.1

            ep_return = 0
            ep_steps = 0

            state = self.env.reset()
            done = False

            update_steps_trace = []

            # initialize dict to accumulate param gradients - {key=param_name: value=param.grad}
            # accumulated_gradients = {}
            # for param_name, param in self.lnet.named_parameters():
            #     accumulated_gradients[param_name] = None
            # self.l_optimizer.zero_grad()

            while not done:

                # if render_final_episodes and ep + 10 > max_episodes:
                #     env.render()

                action_probs = self.lnet(state)[0].clone().detach().numpy()
                greedy_action = np.random.choice(nA, p=action_probs)
                action = eps_greedy_action(greedy_action, nA, eps)
                next_state, reward, done, _ = self.env.step(action)
                ep_return += (df ** ep_steps) * reward
                update_steps_trace.append([state, action, reward])

                ## time to train
                if done or (ep_steps % ep_update_steps == 0):
                    # calculate losses - have to do O(n^2) iterations to prevent circular graph in autograd
                    N = len(update_steps_trace)
                    for j in range(N-1, -1, -1):

                        # calculate discounted return
                        discounted_return = 0
                        if not done:
                            discounted_return = self.lnet(next_state)[1]
                        for k in range(N-1, j, -1):
                            discounted_return = update_steps_trace[k][2] + df * discounted_return

                        # calculate loss
                        state, action, reward = update_steps_trace[j]
                        delta = discounted_return - self.lnet(state)[1]
                        delta_scalar = delta.clone().detach()
                        p_loss = -torch.log(self.lnet(state)[0][action]) * delta_scalar
                        v_loss = delta**2
                        # accumulate gradients
                        v_loss.backward()
                        p_loss.backward()
                    # take the gradient step
                    self.l_optimizer.step()
                    # for next update step
                    self.l_optimizer.zero_grad()


                # for next step
                state = next_state
                ep_steps += 1

            # store ep stats
            self.ep_return_list.put(ep_return)
            if self.episode_counter.value % (max_episodes//10) == 0:
                print('ep:{} \t ep_return:{} \t worker:{}'.format(self.episode_counter.value, ep_return, self.name))




# function to get epsilon-greedy action
def eps_greedy_action(greedy_action, nA, eps):
    action = greedy_action
    r = np.random.uniform(0, 1)
    if eps > r:
        action = np.random.choice(nA)
    return action


## main

if __name__ == "__main__":
    # hyperparams
    lr = 1e-3
    eps = .8
    df = 0.99
    beta = 0 # entropy regularization
    ep_update_steps = 5
    max_episodes = 2000
    # render_final_episodes = True
    random_seed = 42

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    env_name = 'CartPole-v1'
    nS = 4
    nA = 2
    nH = 8

    gnet = GlobalNet(nS, nH, nA)
    gnet.share_memory()

    optimizer = SharedAdam(lr=lr, params=gnet.parameters())

    ep_return_list = mp.Queue()
    episode_counter = mp.Value('i', 0)

    # workers = [Worker(env_name, nS, nA, nH, lr, eps, gnet, optimizer, ep_return_list, episode_counter, i) for i in range(mp.cpu_count())]
    workers = [Worker(env_name, nS, nA, nH, lr, eps, gnet, optimizer, ep_return_list, episode_counter, i) for i in range(1)]
    [w.start() for w in workers]

    ep_returns_moving_mean = [0]
    while episode_counter.value < max_episodes:
        ep_return = ep_return_list.get()
        n = len(ep_returns_moving_mean)
        prev_mean = ep_returns_moving_mean[n-1]
        new_mean = prev_mean + ((ep_return - prev_mean)/n)
        ep_returns_moving_mean.append(new_mean)

    [w.join() for w in workers]


    # plot results
    plt.plot(ep_returns_moving_mean[1:], label='ep_return')
    plt.legend()
    plt.xlabel('episode')
    plt.show()
