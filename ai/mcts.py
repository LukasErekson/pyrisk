#!/usr/bin/python3
import random
import math
from itertools import islice, chain
from ai.mctsnode import MCTSNode
from ai.mctsstate import MCTSState


class MCTS(object):
    CONSTANT = 0.25
    NUM_SIMULATIONS = 3000

    def __init__(self):
        self.root_node = None

    def UCTSearch(self, state0):
        self.UpdateRoot(state0)
        node0 = MCTSNode(state0)
        for i in range(3000):
            node1 = self.TreePolicy(node0)
            reward = self.DefaultPolicy(node1.state)
            self.Backup(node1, reward)
            # à revoir ce qu'on renvoie
        x = self.BestChild(node0, 0).state.action
        return x

    def TreePolicy(self, node):
        while not node.state.terminal():
            if not node.fully_expanded():
                return self.Expand(node)
            else:
                node = self.BestChild(node, MCTS.CONSTANT)
        return node

    def Expand(self, node):
        tried_children = [c.state for c in node.children]
        new_state = node.state.next_random_state()
        while new_state in tried_children:
            new_state = node.state.next_random_state()
        node.add_child(new_state)
        return node.children[-1]

    def BestChild(self, node, coefficient):
        best_score = 0
        best_children = []
        for c in node.children:
            exploit = c.value[node.state.play_order] / c.visits
            explore = math.sqrt(math.log(node.visits) / float(c.visits))
            score = exploit + coefficient * explore
            if score == best_score:
                best_children.append(c)
            if score > best_score:
                best_children = [c]
                best_score = score
        assert len(best_children) > 0, "Error : no children found"
        return random.choice(best_children)

    def Backup(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

    def UpdateRoot(self, state):
        # ?
        for player in chain(islice(state.players.values(), state.play_order + 1, None, 1),
                            islice(state.players.values(), 0, state.play_order, 1)):
            if self.root_node is None:
                self.root_node = MCTSNode(state, None)
            else:
                for c in self.root_node.children:
                    if c.state == state:
                        self.root_node = c
                        break

    def DefaultPolicy(self, state):
        while not state.terminal():
            state = state.next_random_state()
        return state.values()
