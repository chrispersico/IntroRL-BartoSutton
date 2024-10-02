import numpy as np
from scipy.stats import norm

class BanditEnv:
    """
    Pag. 47 - 68
    Repeated choice between k options. After a choice you get a numerical reward from a stationary probability
    distribution, dependent on the action selected.
    We want to maximise the expected total reward over a given period of time.
    We need to estimate the value of every action q(a) to know which one to take.
    All possible k machines are all gaussian distribution so we assign to each of them a random mean, all with sd=1, and in
    the test, we tell it to the user, so he knows if the strategy is learned correctly.

    We have to know at the start which are the means of the N distributions and which is the best.

    option with epsilon that is reduced over time, user can put the possibility, we can
    generate a start and a finish according to his needs, opposed to the single term that
    is constant.

    if problem stationary 1/n otherwise we can use, maybe not constant, value alpha in (0,1]

    add initial Q states, default at 0 but the user can add them;
    they can add useful prior knowledge. Suggest a wildly positive initial state to make initial
    big exploration.
    "optimistic initial values"
    test with Q1 = 5
    remember to the user that is not useful if we are treating a non-stationary problem

    How can we avoid bias of constant step size for non-stationary problems?
    step size = beta = alpha / o
    o_n = trace = o_(n-1) + alpha (1 - o_(n-1))
    o_0 = 0 as a rule; consider o_1 = alpha but not hard coded
    beta_0 do not exist, no choice at the starting
    tell the user that making alpha/(his evolution) is a way to debias the parameter

    UCB Upper Confidence Interval as an alternative to eps greedy:
    choose from the non-non greedy the one more similar or with the best potential
    A_t = argmax(a) [Q_t(a) + c * sqrt((ln t)/(N_t(a)))]
    N(a) number of time a selected
    c controls the degree of exploration, if N(a) = 0 then is always optimal to choose from
    difficult ot expand over the bandit problems but solve them really well
    the more a lever is pulled and less is considered, also if pulled a lot and low reward
    will be ignored. c regulates how much this part is important in respect to the value of state

    NewEstimate <- OldEstimate + StepSize [Target - OldEstimate]

    StepSize = alpha

    Q_n+1 = Q_n + 1/n (R_n - Q_n)

    SIMPLE BANDIT PSEUDOCODE

    Initialize, for a = 1 to k:
        Q(a) <- 0
        N(a) <- 0
    Loop forever:
        A <- {argmax Q(a) with P(1-eps) or random a with P(eps)}
        R <- bandit(A)
        N(A) <- N(A) + 1
        Q(A) <- Q(A) + 1/(N(A)) [R - Q(A)]

    Inform user that he can choose a different approach, we can use also the numerical preference
    for an action H_t(a), and has no interpretation in terms of reward. The only thing that
    count is the relative preference of an action relative to the others

    Action probabilities are soft-max distribution
    P(A=a)=(e^{H_t(a)})/(sum^k_{b=1} e^{H_t(a)}) = pi_t(a)
    pi is the probability of taking action a at time t. Initially all H_1(a)=0
    on each step after selecting A_t and receiving R_t:
    H_{t+1}(A_t)=H_t(A_t)+alpha(R_t-mean(R_t))(1-pi_t(A_t))
    H_{t+1}(a)=H_t(a)-alpha(R_t-mean(R_t))(pi_t(a)) for all a != A_t # watch the sign!
    mean(R_t) include the present one
    if H(.)=0 you see in the equation that all actions have same probability of being picked.
    mean(R_t) is called baseline, if R higher than probability increases otherwise it decreases.
    The gradient bandit is a stochastic approximation to gradient ascent and follows the same
    logic as this implementation making it logical.

    TEST
    make an artificial test where distributions are modified with time with a random walk, so the
    plot shows that sample average has difficulty to solve non-stationary problems.

    add personalized means and sd for the distributions so that we can see where they are going

    show the confrontation between all methods used over a sample test of defined slots
    PARAMETER STUDY
    plot with average reward over 1000 steps and on x values of parameters
every curve has the parameters chosen see page 63 for the example.
    """
    def __init__(self, n: int = 1):
        self.arms_number = n

        self.rand_mean_vector = np.random.random_sample()

        reward = 0


    def lever_pull(self, arm: int = None):

        mean, sd = 0, 1
        r = norm.rvs(loc=mean, scale=sd, size=1)

    def reset(self):
        pass

class BanditOptimization:
    def __init__(self):
        self.action_value_table = None

    def action_value_update(self):
        pass

if __name__ == "__main__":
    """
    Each action has an expected reward given that the action is selected: value of that action.
    Q_t(a) is the estimate of q_*(a) real value.
    Exploiting : take the action with max Q_t(a)
    Exploring : take a random action
    
    epsilon = probability to pick random action each turn
    
    Sample average method for estimating action values
    by the law of large numbers, the estimator converge to the true distribution.
    
    if a taken at least one:
        Q_t(a) = (sum of rewards from action a) / (number of time action a is taken)
    else if never taken:
        Q_t(a) = 0
    
    In the example test we have k = 10, 1000 steps of learning, 2000 of these independent games
    eps_1, eps_2, eps_3 = 0, 0.1, 0.01
    plot, for every step, mean of the 2000 independent games.
    """
    # k number of levers to pull
    k = 10
    # Number of action selections / time periods
    N = 1000
    # Epsilon parameter for exploration and exploitation trade-off
    eps = 0
    Q_ta = np.zeros(k)
