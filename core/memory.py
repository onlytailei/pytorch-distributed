class Memory(object):
    def __init__(self, args):

        # params
        self.state_shape = args.state_shape
        self.action_shape = args.action_shape
        self.reward_shape = args.reward_shape       # num_tasks
        self.gamma_shape = args.gamma_shape         # num_tasks
        self.terminal_shape = args.terminal_shape   # num_tasks

        # memory_params
        self.memory_size = args.memory_size
        self.tensortype = args.tensortype           # for states

        # setup
        self.pos = 0
        self.full = False

    @property
    def size(self):
        if self.full:
            return self.memory_size
        return self.pos

    def append(self, experience):
        raise NotImplementedError("not implemented in base calss")

    def sample(self, batch_size):
        raise NotImplementedError("not implemented in base calss")
