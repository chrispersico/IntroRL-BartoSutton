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

    Now we want to avoid the unlikely assumption of exploring starts

    This following will be ON-policy: attempt to evaluate or improve the policy that is used to
    make decisions. Like Monte Carlo ES
    The policy is soft as pi(a|s)>0 for all s in S and a in A.
    now we present the eps-greedy policy: most of the time they choose an action that has max
    estimated action value, but with probability eps they instead select a random action.
    the minimal probability of a non-greedy action is eps/(|A(s)|) and the remaining bulk of the
    probability is (1 - eps + eps/(|A(s)|)), given to the greedy action.
    eps-greedy are eps-soft policies.
    On-policy first-visit MC control (for eps-soft policies)
    Algorithm parameter: small eps > 0
    Initialize:
        pi <- an arbitrary eps-soft policy
        Q(s,a) in real (arbitrarily), for all s in S and a in A(s)
        Returns(s,a) <- empty list, for all s in S and a in A(s)
    Repeat forever (for each episode):
        Generate an episode following pi: S0,A0,R1,...,S(T-1),A(T-1),RT
        G <- 0
        Loop for each step of the episode, t=T-1,T-2,...0:
            G <- gamma G + R(t+1)
            Unless the pair St,At appears in S0,A0,S1,A1...S(t-1),A(t-1):
                Append G to Returns(St,At)
                Q(St,At) <- average(Returns(St,At))
                A* <- argmax(a) Q(St,a)
                For all a in A(s):
                    pi(a|St) <- {(1 - eps + eps/(|A(s)|)) if a=A* OR eps/(|A(s)|) if a!=A*}

    Now Off-policy prediction via importance sampling
    We use two policies, one that is learned about and that becomes the optimal policy and one
    that is more exploratory and is used to generate behaviour.
    Target policy and behaviour policy -> so is off the target policy.
    They can also be used in different context like learning from human experience or
    non-learning controllers, or for multi step predictive models of world's dynamic.

    Now we consider the prediction problem, with both target and behavior policies are fixed.
    So, estimate v_pi and q_pi, with episodes following another policy b, with b != pi
    pi target
    b behaviour
    so if pi(a|s)>0 implies b(a|s)>0 -> assumption of coverage
    so for the coverage it implies that b needs to be stochastic in states where it is not
    identical to pi.
    In control problem for example pi is deterministic.

    all off-policy methods utilize importance sampling a general technique for estimating
    expected values under one distribution given samples form another.
    Apply importance sampling to off-policy learning by weighting returns according to the
    relative probability of their trajectories occurring under pi and b -> importance sampling ratio
    Basically the v_pi is computed adding a ratio of pi/b to the expectation computed over b.

    if importance sampling is a simple average as in 5.5, is ordinary importance sampling
    if at denominator there is sum of weights, is weighted importance sampling.

    the weighted is biased, the simple is unbiased
    the variance of weighted converges to 0 even if the variance of the ratios is inf, the
    variance or simple is unbounded as the variance of ratios.

    these assumptions where for the first time visit. in the every-visit are both biased, but
    in both the bias falls asymptotically to zero as the number of samples increases.

    every visit approach is usually preferred.

    Example 5.4 pag 127 on blackjack and off policy learning.

    Off-policy MC prediction, policy evaluation for estimating Q=q*
    Initialize, for all s in S, a in A(s):
        Q(s,a) in real
        C(s,a) = 0
    Loop forever (for each episode):
        b <- any policy with coverage of pi
        Generate an episode following b: S0,A0,R1,...,S(T-1),A(T-1),RT
        G <- 0
        W <- 1
        Loop for each step of episode, t=T-1,T-2,...,0 while W != 0:
            G <- gamma G + R(t+1)
            C(St,At) <- C(St,At) + W
            Q(St,At) <- Q(St,At) + W/(C(St,At)) [G - Q(St,At)]
            W <- W pi(At|St)/b(At|St)

    It also works for the on-policy case just by choosing the target and pehaviour policy
    as the same.

    MANCA LA PARTE FINALE

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