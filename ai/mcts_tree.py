#!/usr/bin/python3
import random
import math

class MCTSNode(object):
    def __init__(self,state,parent=None):
        #to avoid dividing by zero
        self.visits=1
        self.reward=0.0
        self.state=state
        self.children=[]
        self.parent=parent

    def add_child(self,child_state):
        child = MCTSNode(child_state,self)
        self.children.append(child)

    def update(self,reward):
        self.reward+=reward
        self.visits+=1

    def fully_expanded(self):
        if len(self.children)==self.state.num_moves:
            return True
        return False
        
class MCTSState(object):
    def __init__(self):

def UCTsearch(state0):
    #we set 100 prediction loop by default, TODO
    node0 = MCTSNode(state0)
    for i in range(100):
        node1 = TreePolicy(node0)
        reward = DefaultPolicy(node1.state)
        Backup(node1,reward)
    return BestChild(node0,0).action

def TreePolicy(node):
    while not node.terminal():
        if node.canexpand():
            return Expand(node)
        else:
            #Constant CP d'explo : 1/sqrt(2)
            node = BestChild(Node,1/math.sqrt(2))
    return node

def Expand(node):
    tried_children=[c.state for c in node.children]
    new_state=node.state.next_random_state()
    while new_state in tried_children:
        new_state=node.state.next_random_state()
    node.add_child(new_state)    


def BestChild(node, coefficient):
    bestscore=0
    bestchildren=[]
    for c in node.children:
        exploit=c.reward/c.visits
        explore=math.sqrt(2.0*math.log(node.visits)/float(c.visits))
        score=exploit+coefficient*explore
        if score == bestscore:
            bestchildren.append(c)
        if score>bestscore:
            bestchildren=[c]
            bestscore=score
    if len(bestchildren)==0:
        #TODO
        return -1
    #TODO ?
    return random.choice(bestchildren)

def DefaultPolicy(state):
    while not state.terminal():
        state=state.next_random_state()
    return state.reward()

def Backup(node,reward):
    while node!=None:
        node.update(reward)
        node=node.parent
