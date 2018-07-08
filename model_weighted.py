import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import copy


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
    prob = exp(value * beta) # beta is the inverse temperature parameter
    prob = prob / sum(prob)  # normalize
    cum_prob = cumsum(prob)  # cummulation summation
    action = where(cum_prob > rand())[0][0]
    return action


real_data = np.array([[0.94252874 ,0.92       ,0.81538462 ,0.72222222],
[0.82191781 ,0.74193548 ,0.48484848 ,0.56666667],
[0.859375   ,0.75       ,0.58461538 ,0.6       ],
[0.95876289 ,0.91304348 ,0.82142857 ,0.91666667],
[0.95061728 ,0.83333333 ,0.76363636 ,0.88235294],
[1.         ,0.89473684 ,0.87878788 ,0.88372093],
[0.88571429 ,0.91428571 ,0.484375   ,0.8       ],
[0.88311688 ,0.9375     ,0.32758621 ,0.66666667],
[0.85333333 ,0.64516129 ,0.77272727 ,0.57142857],
[0.81927711 ,0.7826087  ,0.77777778 ,0.76190476],
[0.48717949 ,0.58333333 ,0.61666667 ,0.56      ],
[0.875      ,0.84210526 ,0.59016393 ,0.71428571],
[0.70666667 ,0.85185185 ,0.61764706 ,0.48      ],
[0.91139241 ,0.94444444 ,0.90384615 ,0.90909091],
[0.8875     ,0.92857143 ,0.67647059 ,0.63636364],
[0.52631579 ,0.25       ,0.57142857 ,0.35714286],
[0.98684211 ,0.93333333 ,0.85714286 ,0.86206897],
[0.98648649 ,0.96875    ,0.94642857 ,0.92592593],
[0.90566038 ,0.9        ,0.90909091 ,1.        ],
[0.80769231 ,0.76923077 ,0.45454545 ,0.65517241],
[0.65       ,0.52631579 ,0.51666667 ,0.65      ],
[0.98717949 ,1.         ,0.86111111 ,0.72727273],
[0.54320988 ,0.37142857 ,0.66666667 ,0.5       ],
[0.44444444 ,0.64516129 ,0.39726027 ,0.33333333],
[0.84415584 ,0.76       ,0.6119403  ,0.67741935],
[0.87837838 ,0.96875    ,0.68055556 ,0.77272727],
[0.42424242 ,0.65517241 ,0.46575342 ,0.53571429],
[0.9375     ,0.875      ,0.77966102 ,0.79310345],
[0.90410959 ,0.91428571 ,0.69230769 ,0.84      ],
[0.1884058  ,0.72727273 ,0.42857143 ,0.75      ],
[0.67567568 ,0.45454545 ,0.52173913 ,0.58333333],])


def model_free_sample(alpha=0.5, beta=5, weight=0.3, gamma=0.98, n_trials=3000,trace=0.6):

    # Weighted version, combining model free and model based 2 updates
    #n_trials = 5000
    #alpha = 0.6
    #beta = 5
    #trace = 0.6
    #gamma = 0.98
    w1 = weight
    w2 = 1 - w1
    n_states = 3
    n_arms = 2

    tst = Daw_two_step_task()
    state = 0

    q = np.zeros((tst.n_states, tst.n_actions)) # model free
    qb = np.zeros((tst.n_states, tst.n_actions)) # MB full
    #qcomb = np.zeros((tst.n_states, tst.n_actions)) # combined q
    q_ev_1 = np.zeros((n_trials*2, tst.n_states, tst.n_actions))
    q_ev_0 = np.zeros((n_trials, tst.n_actions))

    ac_nextS_nextA_rew_ev = np.zeros((4, n_trials)) #[action, next_state, next_action, reward]
    ac_rare_rew_ev = np.zeros((3, n_trials))
    action_ev = np.zeros((n_trials, 2))

    T = np.zeros((n_states-1, n_arms))
    count = np.zeros((n_states-1, n_arms))
    for i in range(n_states-1):
        for j in range(n_arms):
            T[i, j] = np.random.uniform(0,1)

    q_ev = np.zeros((tst.n_states, tst.n_actions, n_trials))
    mu_ev = np.zeros((tst.n_of_bandits, tst.n_actions, n_trials))
    muvec = np.zeros((tst.n_of_bandits, tst.n_actions))

    ac_nextS_nextA_rew_ev = np.zeros((4, n_trials)) #[action, next_state, next_action, reward]
    ac_rare_rew_ev = np.zeros((3, n_trials))

    qMB = np.zeros((1, n_arms))
    for t in range(n_trials):
        state=0
        # choose the spaceship
        action = softmax(state, w1*q+w2*qb, beta)

        # travel to planet A or B
        next_state, _ = tst.get_outcome(state, action)

        # decide on an arm 1 or 2
        # next_action = softmax(next_state, q, beta)
        next_action = softmax(next_state, w1*q+w2*qb, beta)

        # choose between arm 1 or 2, get reward
        _, reward = tst.get_outcome(next_state, next_action)

        # update q for the 0 state

        q[state, action] = q[state, action]  + alpha * (0 + gamma * q[next_state, next_action] - q[state, action])

        qb[state, action] = qb[state, action]  + alpha * (0 + gamma * qb[next_state, next_action] - qb[state, action])
        #q_ev_0[t, action] = q[state, action]

        # update q for the 2nd step states
        q[next_state, next_action] = q[next_state, next_action] + alpha * (reward - q[next_state, next_action])
        q[state, action] = q[state, action] + alpha * trace * (reward - q[state, action])
        
        qb[next_state, next_action] = qb[next_state, next_action] + alpha * (reward - qb[next_state, next_action])
        qb[state, action] = qb[state, action] + alpha * trace * (reward - qb[next_state, next_action])


        ####  Q-MB model based qb function update
        ####
        ns_idx = next_state-1
        count[ns_idx, action] += 1
        T[ns_idx, action] = count[ns_idx, action]/np.sum(count[:,action])

        qtmp = 0
        for sc in range(n_states-1):
            qtmp += T[sc, action] * np.max(qb[sc+1, :])

        qb[state, action] = qtmp
        #####
        #### update for the other 1st stage action as well, to be more model-based
        qtmp = 0
        for sc in range(n_states-1):
            qtmp += T[sc, 1-action] * np.max(qb[sc+1, :])

        qb[state, 1-action] = qtmp

        ##################xx SAVING stuff (not yet mixed for this part)  ##############x
        for i in range(tst.n_of_bandits):
            muvec[i, :] = tst.bandits[i].mu

        q_ev[:, :, t] = q
        mu_ev[:, :, t] = muvec

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


        #q_ev_0[t, action] = q[state, action]
        #q_ev_1[t, next_state, next_action] = q[next_state, next_action]
        action_ev[t, :] = [action, next_action]
        
    actions = ac_rare_rew_ev[0, :]
    actions_next_trial = np.roll(actions, -1)


    # for i in range(len(actions)):
    #     (actions[i], actions_next_trial[i])

    type_rare_rew = np.sum((ac_rare_rew_ev[0, :] == actions_next_trial[:]) & \
                            (ac_rare_rew_ev[1, :] == 1 ) & \
                            (ac_rare_rew_ev[2, :] == 1 )) / \
                    np.sum((ac_rare_rew_ev[1, :] == 1 ) & \
                            (ac_rare_rew_ev[2, :] == 1 ))

    type_rare_norew = np.sum((ac_rare_rew_ev[0, :] == actions_next_trial[:]) & \
                            (ac_rare_rew_ev[1, :] == 1 ) & \
                            (ac_rare_rew_ev[2, :] == 0 )) / \
                      np.sum((ac_rare_rew_ev[1, :] == 1 ) & \
                            (ac_rare_rew_ev[2, :] == 0 ))


    type_comm_rew = np.sum((ac_rare_rew_ev[0, :] == actions_next_trial[:]) & \
                            (ac_rare_rew_ev[1, :] == 0 ) & \
                            (ac_rare_rew_ev[2, :] == 1 )) / \
                    np.sum((ac_rare_rew_ev[1, :] == 0 ) & \
                            (ac_rare_rew_ev[2, :] == 1 ))


    type_comm_norew = np.sum((ac_rare_rew_ev[0, :] == actions_next_trial[:]) & \
                            (ac_rare_rew_ev[1, :] == 0 ) & \
                            (ac_rare_rew_ev[2, :] == 0 )) / \
                     np.sum((ac_rare_rew_ev[1, :] == 0 ) & \
                            (ac_rare_rew_ev[2, :] == 0))
    
    return type_comm_rew, type_rare_rew, type_comm_norew, type_rare_norew

def model_free_fitness_MSE(real_data, alpha=0.5, beta=5, weight=0.3, gamma=0.9, n_trials=10000,trace=0.8):
    cr, rr, cn, rn = model_free_sample(alpha, beta, weight, gamma, n_trials,trace)
    n = real_data.shape[0]

    return np.sum((real_data - np.array([cr, rr, cn, rn]))*(real_data - np.array([cr, rr, cn, rn])))

    
model_free_history = np.zeros((real_data.shape[0], 4))
for sub in range(real_data.shape[0]):

    n_try = 100
    alpha_min = 0
    alpha_max = 1
    beta_min = 0.5
    beta_max = 10
    weight_min = 0.05
    weight_max = 0.95

    best_alpha = None
    best_beta = None
    best_weight = None
    best_value = 1000
    for i in range(n_try):
        alpha = np.random.random()*(alpha_max - alpha_min) + alpha_min
        beta = np.random.random()*(beta_max - beta_min) + beta_min
        weight = np.random.random()*(weight_max - weight_min) + weight_min
        value = model_free_fitness_MSE(real_data[sub,:], alpha, beta, weight)
        if value < best_value:
            best_alpha = alpha
            best_beta = beta
            best_weight = weight
            best_value = value
        #print(value, alpha, beta)

    print('best case for {}:'.format(sub))
    print(best_value, best_alpha, best_beta, best_weight)
    model_free_history[sub, 0] = value
    model_free_history[sub, 1] = alpha
    model_free_history[sub, 2] = beta
    model_free_history[sub, 3] = weight
print('all result')
print(model_free_history)