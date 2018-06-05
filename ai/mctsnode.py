#!/bin/python3

class MCTSNode(object):
    def __init__(self, state, parent=None):
        self.visits = 0
        self.value = [0,0,0]
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        child = MCTSNode(child_state, self)
        self.children.append(child)

    def update(self, reward):
        self.value = list([x+y for x, y in zip(self.value, reward)])
        self.visits += 1

    def fully_expanded(self):
        if len(self.children) == len(self.state.empty):
            return True
        return False

    def __repr__(self):
        s = "Node; children %d; visits %d; reward %d" % (len(self.children), self.visits, self.reward)
        return s
