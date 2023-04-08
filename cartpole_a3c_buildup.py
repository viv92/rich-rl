import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import sys
import torch.multiprocessing as mp


# Local policy net - specific for each worker
class LocalNet_p(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim, num_layers):
        super().__init__()
        self.fc_in = nn.Linear(s_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, a_dim)

        self.n_layers = num_layers
        self.fc_n = nn.ModuleList()
        for _ in range(num_layers):
            self.fc_n.append(nn.Linear(h_dim, h_dim))

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        state = torch.tensor(state)
        h = self.relu(self.fc_in(state))

        for i in range(self.n_layers):
            h = self.relu(self.fc_n[i](h))

        out = self.softmax(self.fc_out(h))
        return out


# Local value net - specific for each worker
class LocalNet_v(nn.Module):
    def __init__(self, s_dim, h_dim, num_layers):
        super().__init__()
        self.fc_in = nn.Linear(s_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, 1)

        self.n_layers = num_layers
        self.fc_n = nn.ModuleList()
        for _ in range(num_layers):
            self.fc_n.append(nn.Linear(h_dim, h_dim))

        self.relu = nn.ReLU()

    def forward(self, state):
        state = torch.tensor(state)
        h = self.relu(self.fc_in(state))

        for i in range(self.n_layers):
            h = self.relu(self.fc_n[i](h))

        out = self.fc_out(h)
        return out


# Global policy net - single instance shared across workers
class GlobalNet_p(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim, num_layers):
        super().__init__()
        self.fc_in = nn.Linear(s_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, a_dim)

        self.n_layers = num_layers
        self.fc_n = nn.ModuleList()
        for _ in range(num_layers):
            self.fc_n.append(nn.Linear(h_dim, h_dim))

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, state):
        state = torch.tensor(state)
        h = self.relu(self.fc_in(state))

        for i in range(self.n_layers):
            h = self.relu(self.fc_n[i](h))

        out = self.softmax(self.fc_out(h))
        return out


# Global value net - single instance shared across workers
class GlobalNet_v(nn.Module):
    def __init__(self, s_dim, h_dim, num_layers):
        super().__init__()
        self.fc_in = nn.Linear(s_dim, h_dim)
        self.fc_out = nn.Linear(h_dim, 1)

        self.n_layers = num_layers
        self.fc_n = nn.ModuleList()
        for _ in range(num_layers):
            self.fc_n.append(nn.Linear(h_dim, h_dim))

        self.relu = nn.ReLU()

    def forward(self, state):
        state = torch.tensor(state)
        h = self.relu(self.fc_in(state))

        for i in range(self.n_layers):
            h = self.relu(self.fc_n[i](h))

        out = self.fc_out(h)
        return out


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
    def __init__(self, env_name, nS, nA, nH, num_layers, lr, eps, gnet_p, gnet_v, g_optimizer_p, g_optimizer_v, optimization_lock, ep_return_list, episode_counter, worker_seed, worker_name):
        super().__init__()
        # local members
        self.env = gym.make(env_name)
        self.lnet_p = LocalNet_p(nS, nH, nA, num_layers)
        self.lnet_v = LocalNet_v(nS, nH, num_layers)
        self.l_optimizer_p = torch.optim.Adam(lr=lr, params=self.lnet_p.parameters())
        self.l_optimizer_v = torch.optim.Adam(lr=lr, params=self.lnet_v.parameters())
        self.nA = nA
        self.eps = eps
        self.name = str(worker_name)
        self.seed = int(worker_seed)

        # global members
        self.gnet_p = gnet_p
        self.gnet_v = gnet_v
        self.g_optimizer_p = g_optimizer_p
        self.g_optimizer_v = g_optimizer_v
        self.optimization_lock = optimization_lock
        self.ep_return_list = ep_return_list
        self.episode_counter = episode_counter

    def run(self):
        # set random seed for worker instance
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        while self.episode_counter.value < max_episodes:
            # with self.episode_counter.get_lock():
            with self.optimization_lock:
                self.episode_counter.value += 1

            # eps decay
            if self.episode_counter.value % (max_episodes//10) == 0:
                self.eps -= (1/7)

            ep_return = 0
            ep_steps = 0
            update_steps_trace = []

            state = self.env.reset(seed=self.seed)
            done = False

            while not done:

                # if render_final_episodes and ep + 10 > max_episodes:
                #     env.render()

                # # sync lnet params with gnet params
                # with self.optimization_lock:
                #     self.lnet_v.load_state_dict(self.gnet_v.state_dict())
                #     self.lnet_p.load_state_dict(self.gnet_p.state_dict())

                action_probs = self.lnet_p(state).clone().detach().numpy()
                # check for nan
                if np.isnan(np.sum(action_probs)):
                    # take random action
                    action = np.random.choice(self.nA)
                else:
                    greedy_action = np.random.choice(self.nA, p=action_probs)
                    # get eps greedy action
                    action = greedy_action
                    r = np.random.uniform(0, 1)
                    if self.eps > r:
                        action = np.random.choice(self.nA)

                next_state, reward, done, _ = self.env.step(action)
                ep_return += (df ** ep_steps) * reward
                update_steps_trace.append([state, action, reward])

                ## time to train
                if done or (ep_steps % update_steps == 0):
                    # calculate losses - have to do O(n^2) iterations to prevent circular graph in autograd
                    N = len(update_steps_trace)
                    for j in range(N-1, -1, -1):

                        # calculate discounted return
                        discounted_return = 0
                        if not done:
                            discounted_return = self.lnet_v(next_state)
                        for k in range(N-1, j, -1):
                            discounted_return = update_steps_trace[k][2] + df * discounted_return

                        # calculate loss
                        state, action, reward = update_steps_trace[j]
                        delta = discounted_return - self.lnet_v(state)
                        delta_scalar = delta.clone().detach()
                        p_loss = -torch.log(self.lnet_p(state)[action]) * delta_scalar
                        entropy = -torch.dot(torch.log(self.lnet_p(state)), self.lnet_p(state))
                        p_loss_total = p_loss - beta * entropy
                        v_loss = delta**2
                        # accumulate gradients
                        v_loss.backward()
                        p_loss_total.backward()

                    # following optimization operations to be done by acquiring a lock
                    with self.optimization_lock:

                        self.g_optimizer_v.zero_grad()
                        self.g_optimizer_p.zero_grad()

                        # copy lnet grads to gnet grads
                        for lp, gp in zip(self.lnet_p.parameters(), self.gnet_p.parameters()):
                            gp._grad = lp.grad.clone()
                        for lp, gp in zip(self.lnet_v.parameters(), self.gnet_v.parameters()):
                            gp._grad = lp.grad.clone()

                        # take the gradient step on gnet
                        self.g_optimizer_v.step()
                        self.g_optimizer_p.step()

                        # for next update step
                        update_steps_trace = []
                        self.l_optimizer_v.zero_grad()
                        self.l_optimizer_p.zero_grad()

                        # sync lnet params with gnet params
                        self.lnet_v.load_state_dict(self.gnet_v.state_dict())
                        self.lnet_p.load_state_dict(self.gnet_p.state_dict())



                # for next step
                state = next_state
                ep_steps += 1

            # store ep stats
            with self.optimization_lock:
                self.ep_return_list.put(ep_return)

            if self.episode_counter.value % (max_episodes//10) == 0:
                print('ep:{} \t ep_return:{} \t worker:{}'.format(self.episode_counter.value, ep_return, self.name))




### main

if __name__ == '__main__':
    # hyperparams
    lr = 1e-4
    eps = 1.
    beta = 0
    df = 0.99
    update_steps = 5
    max_episodes = 9000
    render_final_episodes = False
    random_seed = 42

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # load environment
    env_name = 'CartPole-v1'
    nS = 4
    nA = 2
    nH = 24
    num_layers = 2

    # num_workers = mp.cpu_count()
    num_workers = 2
    worker_seeds = np.arange(num_workers) * random_seed

    gnet_p = GlobalNet_p(nS, nH, nA, num_layers)
    gnet_p.share_memory()
    gnet_v = GlobalNet_v(nS, nH, num_layers)
    gnet_v.share_memory()

    g_optimizer_p = SharedAdam(lr=lr, params=gnet_p.parameters())
    g_optimizer_v = SharedAdam(lr=lr, params=gnet_v.parameters())

    optimization_lock = mp.Lock()

    ep_return_list = mp.Queue()
    episode_counter = mp.Value('i', 0)

    workers = [Worker(env_name, nS, nA, nH, num_layers, lr, eps, gnet_p, gnet_v, g_optimizer_p, g_optimizer_v, optimization_lock, ep_return_list, episode_counter, worker_seeds[i], i) for i in range(num_workers)]

    [w.start() for w in workers]

    # resuts to plot
    ep_returns_100moving_mean = [0]
    n = 0
    while episode_counter.value < max_episodes:
        n += 1
        n = n % (max_episodes//20)
        ep_return = ep_return_list.get()
        prev_mean = ep_returns_100moving_mean[-1]
        new_mean = prev_mean + ((ep_return - prev_mean)/(n+1))
        ep_returns_100moving_mean.append(new_mean)

    [w.join() for w in workers]

    # hyperparam dict
    hyperparam_dict = {}
    hyperparam_dict['env'] = 'CartPole-v1'
    hyperparam_dict['algo'] = 'a3c-buildup'
    hyperparam_dict['update-steps'] = str(update_steps)
    hyperparam_dict['lr'] = str(lr)
    hyperparam_dict['num-workers'] = str(num_workers)
    hyperparam_dict['beta'] = str(beta)
    hyperparam_dict['df'] = str(df)
    hyperparam_dict['max-ep'] = str(max_episodes)
    hyperparam_dict['nH'] = str(nH)
    hyperparam_dict['num-layers'] = str(num_layers)
    hyperparam_dict['random-seed'] = str(random_seed)

    # hyperparam string
    hyperstr = ""
    for k,v in hyperparam_dict.items():
        hyperstr += k + ':' + v + "__"

    # plot results
    plt.plot(ep_returns_100moving_mean, label='ep_return')
    plt.legend()
    plt.xlabel('episode')
    plt.savefig('plots/' + hyperstr + '.png')
