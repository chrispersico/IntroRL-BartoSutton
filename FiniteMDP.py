class FiniteMDPenv:
    """
    over discrete time steps
    state -> action -> reward -> new state ... -> trajectory
    dynamic of MDP: p(s',r|s,a) = Pr(S_t=s',R_t=r|S_t-1=s,A_t-1=a)
    so a state is only dependent on his previous state and not the time before that, markov prop.
    state transition probabilities: p(s'|s,a)=Pr(S_t=s'|S_t-1=s,A_t-1=a)=sum(r) p(s',r|s,a)
    and others at page 71 that can directly derive from the first one

    check that reward must be a scalar signal
    each episode stop in the terminal state

    the return need to keep the discounting effect inside of it so that infinite episodes can
    be considered.
    An early stopping parameter need to be introduced

    the agent tries to select actions so that the sum of the discounted rewards it receives
    over the future is maximised
    max expected discounted return: Gt = sum_k gamma^k R_(t+k+1)
    so about all future rewards from a particular state
    gamma is the discount parameter in [0, 1]
    so the more is in the future and smaller is gamma, so smaller the value
    inform the user that gamma=0 means only present, gamma=1 means great future view
    If the actions do not influence future rewards, an optimization of single actions could
    in theory work (gamma=0). But it's not guaranteed.

    G_t = R_(t+1) + gamma G_(t+1)
    G is actually the sequence of the rewards AFTER t!!!

    generic formula that for now unifies continuous and discrete tasks:
    G_t = sum^T_k=(t+1) gamma^(k-t-1) R_k; this can contain the case fot T=inf and gamma=1

    note that for the property of geometric means, for infinite episodes, G still converges <inf


    almost all reinforcement learning algorithms involve the estimation of a value function, so
    estimate how good is for the agent to be in a given state (or perform an action in a given
    state) so, how good is the expected return

    value function are defined according to a particular way of acting, called policy. And is a
    mapping from states ot probabilities of selecting each possible action.
    pi(a|s) is the probability to pick a in state s. so A_t=s given S_t=s

    the value function under policy pi
    v_pi(s)=E_pi[G_t|S_t=s]=E_pi[sum(k) gamma^kR_t+k+1|S_t=s] for all s in S
    the expected because there is a probability os picking being a stochastic policy
    also adding a we get action value function
    q_pi(s,a)=E_pi[G_t|S_t=s,A_t=a]=E_pi[sum(k) gamma^kR_t+k+1|S_t=s,A_t=a]

    value function and action value function can be estimated from experience, if agent follows
    policy pi, and maintains an average of the acual returns that have followed the state, then
    the average will converge to the state's value as the number of times that state is
    encountered approach infinity.
    if separate averages are kept for each action from each state, it will converge to the
    action value of pi.

    averaging over many random samples of actual returns: monte carlo methods.
    if there are much state, v_pi and q_pi are treated as parametric functions with less param
    than states to reduce the complexity.

    pag.81 bellman equation 3.14

    the methods are the one to approximate these functions

    """
    def __init__(self):
        """
        env variables to keep during the episode
        """
        self.state = None

    def reset(self):
        """
        reset env variables
        :return: the initial values for the state for the user and the agent and the bool
        false for the "terminal" space.
        """
        self.state = None

        return self.state, False

    def step(self, action):
        """
        time step, from the current state, and action take reward, modify state
        :param action: action taken by the agent
        :return: return the new state and additional information for the user
        """
        reward = 0 + action, self.state

        self.state = None, action

        terminal = False

        return self.state, reward, terminal


if __name__ == "__main__":

    max_ep_length = 100

    obs, terminal = FiniteMDPenv.reset()

    while True:

        if terminal:
            break