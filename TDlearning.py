class TDlearning:
    """
    It is basically a combination of MC and DP ideas.
    TD methods update estimates based in part on other learned estimates, without waiting for a
    final outcome (bootstrap).
    First policy evaluation and then GPI as before.


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

    obs, terminal = TDlearning.reset()

    while True:

        if terminal:
            break