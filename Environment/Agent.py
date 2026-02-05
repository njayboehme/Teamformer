class Agent:
    def __init__(self, label, **kwargs):
        self.label = label
        # Capability of the agent, just the index of the capability
        self.cap = kwargs.get('cap', 0)
        # Has an agent failed or not
        self.failed = False

