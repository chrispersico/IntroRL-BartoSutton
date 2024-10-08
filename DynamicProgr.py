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