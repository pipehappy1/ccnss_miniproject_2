import numpy as np
import pandas as pd
from pylab import *

df=pd.read_pickle('df.pickle')

start_index = np.array([1062, 1360, 1388, 1409, 1507, 1656, 1695, 1733, 1734, 1755, 1796,
                        1856, 1894, 1921, 1922, 1928, 1932, 1935, 1942, 1943, 1948, 1953,
                        2058, 2061, 2064, 2065, 2066, 2068, 2070, 2071, 2072])

def softmax(state, q, beta):
    """
    Softmax policy: selects action probabilistically depending on the value.
    Args:
        state: an integer corresponding to the current state.
        q: a matrix indexed by state and action.
        params: a dictionary containing the default parameters.
    Returns:
        an integer corresponding to the action chosen according to the policy.
    """
    
    value = q[state,:]
    prob = np.exp(value * beta) # beta is the inverse temperature parameter
    prob = prob / sum(prob)  # normalize
    cum_prob = cumsum(prob)  # cummulation summation
    action = where(cum_prob > rand())[0][0]
    return action


class world(object):
    def __init__(self):
        return

    def get_outcome(self):
        return
    
    def get_all_outcomes(self):
        
        outcomes = {}
        for state in range(self.n_states):
            for action in range(1 if self.n_actions == 0 else self.n_actions):
                next_state, reward = self.get_outcome(state, action)
                outcomes[state, action] = [(1, next_state, reward)]
        return outcomes



class drifting_probabilitic_bandit(world):
    """
    World: 2-Armed bandit.
    Each arm returns reward with a different probability.
    The probability of returning rewards for all arms follow Gaussian random walks.
    """
    
    def __init__(self, arm_number, drift):
        self.name = "n_armed_bandit"
        self.n_states = 1
        self.n_actions = arm_number
        self.dim_x = 1
        self.dim_y = 1
        
        self.mu_min = 0.25
        self.mu_max = 0.75
        self.drift = drift
        
        self.reward_mag = 1
        
        self.mu = [np.random.uniform(self.mu_min, self.mu_max) for a in range(self.n_actions)]
        
    def update_mu(self):
        self.mu += np.random.normal(0, self.drift, self.n_actions)
        self.mu[self.mu > self.mu_max] = self.mu_max
        self.mu[self.mu < self.mu_min] = self.mu_min
            
    def get_outcome(self, state, action):
        
        self.update_mu()
        self.rewards = [self.reward_mag if np.random.uniform(0,1) < self.mu[a] else 0 for a in range(self.n_actions) ]
        next_state = None
        
        reward = self.rewards[action]
        return int(next_state) if next_state is not None else None, reward



class contextual_bandits(drifting_probabilitic_bandit):
  
    def __init__(self):
        
        self.name = "contextual_bandit"
        self.n_states = 3
        self.n_actions = 2
        self.dim_x = 1
        self.dim_y = 1
        
        self.n_arms = self.n_actions
        self.n_of_bandits = self.n_states
        self.drift = 0.02
        self.bandits = [drifting_probabilitic_bandit(self.n_arms, self.drift) for n in range(self.n_of_bandits)]

    def get_outcome(self, state, action):
        
        _, reward = self.bandits[state].get_outcome(0,action)
        available_states = [s for s in range(self.n_of_bandits) if s != state]
        next_state = np.random.choice(available_states)
        
        return int(next_state) if next_state is not None else None, reward



def e_greedy_policy(q, epsilon):
    if np.random.uniform(0,1) > epsilon:
        return np.argmax(q)
    else:
        # choose randomly from all but the highest q value action
        a_list = np.arange(len(q))
        arg_max = np.argmax(q)
        choices = np.delete(a_list, arg_max)
        return np.random.choice(np.delete(a_list, arg_max))


class Daw_two_step_task(drifting_probabilitic_bandit):
  
    def __init__(self):
        
        self.name = "Daw_two_step_task"
        self.n_states = 3
        self.n_actions = 2
        self.dim_x = 1
        self.dim_y = 1
        
        self.n_arms = self.n_actions
        self.n_of_bandits = 2
        self.drift = 0.02
        
        self.context_transition_prob = 0.7
        self.bandits = [drifting_probabilitic_bandit(self.n_arms, self.drift) for n in range(self.n_of_bandits)]
        
    def get_outcome(self, state, action):
        
        if state == 0:
            reward = 0
            if action == 0:
                if np.random.uniform(0,1) < self.context_transition_prob:
                    next_state = 1
                else:
                    next_state = 2
            elif action == 1:
                if np.random.uniform(0,1) < self.context_transition_prob:
                    next_state = 2
                else:
                    next_state = 1
            else:
                print('No valid action specified')
                
        if state == 1:
            _, reward = self.bandits[0].get_outcome(0, action)
            next_state = 0
            
        if state == 2:
            _, reward = self.bandits[1].get_outcome(0, action)
            next_state = 0
        
        return int(next_state) if next_state is not None else None, reward


def softmax_score(state, q, beta, real_action): # returns the probability score
    """
        Softmax policy: selects action probabilistically depending on the value.
        Args:
            state: an integer corresponding to the current state.
            q: a matrix indexed by state and action.
            params: a dictionary containing the default parameters.
        Returns:
            an integer corresponding to the action chosen according to the policy.
    """

    value = q[state,:]
    prob = exp(value * beta) # beta is the inverse temperature parameter
    prob = prob / sum(prob)  # normalize
    cum_prob = cumsum(prob)  # cummulation summation
    action = where(cum_prob > rand())[0][0]
    return prob[real_action]




def MB1_nll(alphap,betap,sub_indx):
    df_sub = df.loc[df['sub']==sub_indx]
    df_sub = df_sub.reset_index(drop=True)
    n_trials = len(df_sub)
    alpha = alphap
    beta = betap
    trace = 0.9
    gamma = 0.8

    tst = Daw_two_step_task()
    n_bandit = tst.n_of_bandits
    n_arms = tst.n_actions
    n_states = tst.n_states

    T = np.zeros((n_states-1, n_arms))
    count = np.zeros((n_states-1, n_arms))

    for i in range(n_states-1):
        for j in range(n_arms):
            T[i, j] = np.random.uniform(0,1)

    ev_nexts = np.zeros(n_trials)
    ev_s = np.zeros(n_trials)
    ev_ac = np.zeros(n_trials)

    state = 0
    actions_ev = np.zeros((n_trials, 2))
    q = np.zeros((tst.n_states, tst.n_actions))
    # q_ev_1 = np.zeros((n_trials*2, tst.n_states, tst.n_actions))
    # q_ev_0 = np.zeros((n_trials, tst.n_actions))
    mu_ev = np.zeros((tst.n_of_bandits, tst.n_actions, n_trials))
    muvec = np.zeros((tst.n_of_bandits, tst.n_actions))

    action_ev = np.zeros((n_trials, 2))

    ac_nextS_nextA_rew_ev = np.zeros((4, n_trials)) #[action, next_state, next_action, reward]
    ac_rare_rew_ev = np.zeros((3, n_trials))

    q_ev = np.zeros((tst.n_states, tst.n_actions, n_trials))
    # mu_ev = np.zeros((tst.n_of_bandits, tst.n_actions, n_trials))
    # muvec = np.zeros((tst.n_of_bandits, tst.n_actions))
    qMB = np.zeros((1, n_arms))

    ll = 0
    for t in range(n_trials):
        state=0
        # choose the spaceship
        action = softmax(state, q, beta)
        real_action = df_sub.iloc[t,1] - 1
        ll += np.log(softmax_score(state, q, beta,real_action))
        # travel to planet A or B
        next_state, _ = tst.get_outcome(state, action)

        # decide on an arm 1 or 2
        next_action = softmax(next_state, q, beta)

        # choose between arm 1 or 2, get reward
        _, reward = tst.get_outcome(next_state, next_action)

        # update q for the 0 state
        q[state, action] = q[state, action]  + alpha * (0 + gamma * q[next_state, next_action] - q[state, action])

        # update q for the 2nd step states
        q[next_state, next_action] = q[next_state, next_action] + alpha * (reward - q[next_state, next_action])

        q[state, action] = q[state, action] + alpha * trace * (reward - q[next_state, next_action])

        ####  Q-MB model based q function update
        ####
        ns_idx = next_state-1
        count[ns_idx, action] += 1
        T[ns_idx, action] = count[ns_idx, action]/np.sum(count[:,action])

        qtmp = 0
        for sc in range(n_states-1):
            qtmp += T[sc, action] * np.max(q[sc+1, :])

        q[state, action] = qtmp
        #     #####
        #     #### update for the other 1st stage action as well, to be more model-based
        #     qtmp = 0
        #     for sc in range(n_states-1):
        #         qtmp += T[sc, 1-action] * np.max(q[sc+1, :])

        #     q[state, 1-action] = qtmp


        ##################xx SAVING stuff ##############x
        for i in range(tst.n_of_bandits):
            muvec[i, :] = tst.bandits[i].mu

        q_ev[:, :, t] = q
        mu_ev[:, :, t] = muvec

        ac_nextS_nextA_rew_ev[:, t] = [action, next_state, next_action, reward]

        #q_ev_0[t, action] = q[state, action]
        #q_ev_1[t, next_state, next_action] = q[next_state, next_action]
        action_ev[t, :] = [action, next_action]

        ac_nextS_nextA_rew_ev[ :, t] = [action, next_state, next_action, reward]

        tranzp_01 = tst.context_transition_prob
        if tranzp_01 >= 0.5:
            if ((action == 0) & (next_state == 1)) | ((action == 1) & (next_state == 2)):
                ac_rare_rew_ev[:, t] = [action, 0, reward]
            else:
                ac_rare_rew_ev[:, t] = [action, 1, reward]
        else:
            if ((action == 0) & (next_state == 1)) | ((action == 1) & (next_state == 2)):
                ac_rare_rew_ev[:, t] = [action, 1, reward]
            else:
                ac_rare_rew_ev[:, t] = [action, 0, reward]
                
    return -ll

    

#print(nll_MF(0.5, 5, 1062))

n_subs = 31
model_free_history = np.zeros((n_subs, 3))
for sub in range(n_subs):

    n_try = 100
    alpha_min = 0
    alpha_max = 1
    beta_min = 0.5
    beta_max = 10

    best_alpha = None
    best_beta = None
    best_value = 1000
    for i in range(n_try):
        alpha = np.random.random()*(alpha_max - alpha_min) + alpha_min
        beta = np.random.random()*(beta_max - beta_min) + beta_min
        value = MB1_nll(alpha, beta, start_index[sub])
        if value < best_value:
            best_alpha = alpha
            best_beta = beta
            best_value = value
        #print(value, alpha, beta)

    print('best case for {}:'.format(sub))
    print("[{}, {}, {}],".format(best_value, best_alpha, best_beta))
    model_free_history[sub, 0] = best_value
    model_free_history[sub, 1] = best_alpha
    model_free_history[sub, 2] = best_beta
print('all result')

