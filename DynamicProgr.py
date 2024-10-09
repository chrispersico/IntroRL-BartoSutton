class DynamicProgramming:
    """
    Compute a optimal policies given a perfect model of the environment as a Markov Decision
    Process.
    All the other algos are computationally efficient DPs.
    The environment is finite MDP so S, A, R and the dynamic given P(s',r|s,a)
    Concept: use of value functions to organize and structure the search for good policies.
    Once we have the optimal value function, it's easy to find the optimal policy,
    which satisfy the Bellman optimality equation.
    95
    DP algorithms are obtained by turning bellman equations such as these into assignments, that
    is , into update rules for improving approximations of the desired value functions.
    POLICY EVALUATION
    policy evaluation is the computation of the state-value function v_pi for an arbitrary
    policy pi. (Also called prediction problem)
    v_pi = sum(a)pi(a|s)sum(s',r)P(s',r|s,a)[r+gamma v_pi(s')]

    if the environment dynamics are completely known, there is a system of |S| linear equations
    with |S| unknowns.

    iterative policy evaluation: given a set of polycies {v0, v1 ... vk}
    for k -> inf than it converges just for the existence criterion.
    to produce each successive approximation v(k+1), it applies the same operations to each state
    s.
    replace the old value of s with the new one from old from successor state and the expected
    immediate rewards, along all the one-step transitions possible under the policy being eval.
    this is an expected update.
    each iteration of iterative policy evaluation updates the value of every state once to produce
    the new approximate value function v(k+1)

    various kinds of expected updates, depending on whether a state of state action pair is being
    updated, and how the estimated values of successor states are combined.
    in DP are all called expected updates because they are based on an expectation over all
    possible next states rather than on a sample next state.

    sequential computer program for iterative policy evaluation as given by
    input pi, policy to be evaluated
    theta > 0 small threshold
    initialize V(s) for all s in S+, arbitrarily except V(terminal)=0
    Loop:
        delta <- 0
        Loop for each s in S:
            v <- V(s)
            V(s) <- sum(a)pi(a|s)sum(s',r)P(s',r|s,a)[r + gamma V(s')]
            delta <- max(delta, |v -V(s)|)
    until delta < theta

    env to test pag. 98

    evaluation and improvement follow each others
    we know that each policy is a strict improvement over the other, unless already optimal
    finite MDP only finite number of policies, in the end ensured convergence.

    POLICY ITERATION
    1. initialization:
    V(s) in Real and pi(s) in A(s) arbitrarily for all s in S
    2. policy evaluation:
    Loop:
    delta <- 0
    Loop for each s in S:
        v <- V(s)
        V(s) <- sum(s',r) P(s',r|s,pi(s))[r + gamma V(s')]
        delta <- max(delta, |v -V(s)|)
    until delta < theta
    3. policy improvement:
    policy-stable <- true
    for each s in S:
        old-action <- pi(s)
        pi(s) <- argmax(a) sum(s',r) P(s',r|s,a)[r + gamma V(s')]
        if old-action different pi(s), then policy-stable <- false
    if policy-stable, then stop and return V = v* and pi = pi*; else go 2
    also, if policy evaluated equal to the old, keep this and stop iteration.
    otherwise infinite loop, ex.4.4 pag. 104.

    env to test pag. 103

    POLICY ITERATION FOR ACTION VALUES
    1. initialization:
    Q(a,s) in Real and pi(s) in A(s) arbitrarily for all s in S
    2. policy evaluation:
    Loop:
    delta <- 0
    Loop for each s in S:
        q <- Q(a,s)
        Q(a,s) <- sum(s',r) P(s',r|s,a)[r + gamma [sum(a') pi(a'|s') Q(s',a')] ]
        delta <- max(delta, |q - Q(s,a)|)
    until delta < theta
    3. policy improvement:
    policy-stable <- true
    for each s in S:
        old-action <- pi(s)
        pi(s) <- argmax(a) Q(s,a)
        if old-action different pi(s), then policy-stable <- false
    if policy-stable, then stop and return Q = q* and pi = pi*; else go 2
    also, if policy evaluated equal to the old, keep this and stop iteration.

    it's possible to combine the two passages of policy iteration in just one, and it's called
    value iteration. We just take the max action in every step.
    VALUE ITERATION
    parameter: theta threshold >0 determining accuracy of estimation
    initialize: V(s) for all s in S+, arbitrarily except V(terminal)=0
    Loop:
        delta <- 0
        Loop for each s in S:
            v <- V(s)
            V(s) <- max(a) sum(s',r) P(s',r|s,a)[r + gamma V(s')]
            delta <- max(delta, |v - V(s)|)
    until delta < 0
    Output: a deterministic policy, pi = pi* such that pi(s)=argmax(a)sum(s',r) P(s',r|s,a)[r + gamma V(s')]
    page 106 there are more indications
    Faster convergence is often achieved by interposing multiple policy evaluation sweeps
    between each policy improvement sweep.

    env to test pag. 106

    Asynchronous DP are in place iterative DP, they do not systematically sweep the entire state
    set. Update values for states in a whatsoever order using the available values.
    But to converge correctly must update the values of all states.

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

    obs, terminal = DynamicProgramming.reset()

    while True:

        if terminal:
            break