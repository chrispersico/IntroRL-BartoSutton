class MonteCarlo:
    """
    estimating value function and discovering optimal policies.
    require only experience
    sample states, actions and rewards from actual simulated environment
    no prior knowledge needed
    even learn from simulation is better because you still sample from env and not need the
    complete distributions of all env dynamics, so you use sample transitions to learn.

    the method is based on averaging sample returns, and to ensure well-defined rewards
    the process needs to be episodic. (average complete returns, the opposite of TD with partial returns)
    only at the end value is estimated and policy changed
    mc sample and average returns for each state-action pair as in bandit problems
    the problem considering this inter-dependence between actions is non-stationary
    GPI
    learn value functions from sample returns with the MDP
    prediction problem -> policy improvement -> control problem -> solution by GPI

    PREDICTION:
    learn state-value function for a given policy: expected return from that state
    more return observed -> converge to the average
    you can decide if you want to calculate after the first visit of a state or all the visit
    in an episode.

    First visit MC prediction for V = v_pi
    Input: a policy pi to be evaluated
    Initialize:
        V(s) in real, arbitrarily, for all s in S
        Returns(s) <- an empty list, for all s in S
    Loop forever (for each episode):
        Generate an episode following pi: S0,A0,R1,S1,A1,R2...S(T-1),A(T-1),RT
        G <- 0
        Loop for each step of episode, t=T-1,T-2,...,0:
            G <- gamma G + R(t+1)
            Unless S_t appears in S0,S1,...,S(t-1):
                append G to Returns(St)
                V(St) <- average(Returns(St))

    Every visit MC would be the same except without the check for St having occurred in the episode

    Example env pag. 115

    it is best for montecarlo to approximate Q(a,s) and not V(s) so that is easier the choice of
    action when talking about policies.
    So, for the state in question we have the values of all actions possible from that state.

    exploring starts clause, prevent the non exploration of possible actions.
    every starting action-state pair has non 0 probability of being selected at the start

    monte carlo can do control too, and it's inside the GPI container
    we perform alternating complete steps of policy evaluation and policy improvement
    policy improvement is made by making policy greedy, in respect to q_pi_k

    MONTE CARLO ES (Exploring Starts)
    Initialize:
        pi(s) in A(s) arbitrarily for all s in S
        Q(s,a) in real arbitrarily for all s in S and a in A(s)
        Returns(s,a) <- empty list, for all s in S and a in A(s)
    Loop forever (for each episode):
        Choose S0 in S, A0 in A(S0) randomly such that all pairs gave probability > 0
        Generate an episode from S0,A0, following pi: S0,A0,R1,...S(T-1),A(T-1),RT
        G <- 0
        Loop for each step of episode t = T-1, T-2,...,0:
            G <- gamma G + R(t+1)
            Unless the pair St,At appears in S0,A0,S1,A1,...S(t-1),A(t-1):
                Append G to Returns(St,At)
                Q(St,At) <- average(Returns(St,At))
                pi(St) <- argmax(a) Q(St,a)

    More efficient version as in section 2.4: (pag. 121)
    Initialize:
        pi(s) in A(s) arbitrarily for all s in S
        Q(s,a) in real arbitrarily for all s in S and a in A(s)
        N(s,a) <- 0 # the avg is done on multiple episodes obviously! And each pair has his one
    Loop forever (for each episode):
        Choose S0 in S, A0 in A(S0) randomly such that all pairs gave probability > 0
        Generate an episode from S0,A0, following pi: S0,A0,R1,...S(T-1),A(T-1),RT
        G <- 0
        Loop for each step of episode t = T-1, T-2,...,0:
            G <- gamma G + R(t+1)
            Unless the pair St,At appears in S0,A0,S1,A1,...S(t-1),A(t-1):
                Q(St,At) <- (N(s,a)-1)/N(s,a) Q(St,At) + 1/N(s,a) G
                pi(St) <- argmax(a) Q(St,a)

    Apply also this to the poker env.


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

    obs, terminal = MonteCarlo.reset()

    while True:

        if terminal:
            break