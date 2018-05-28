#!/usr/bin/python3
import random
import time
import math
import hashlib
from copy import deepcopy
from itertools import islice

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
        if len(self.children)==len(self.state.empty):
            return True
        return False

    def __repr__(self):
        s="Node; children %d; visits %d; reward %d"%(len(self.children),self.visits,self.reward)
        return s
        
class MCTSState(object):
    def __init__(self,player,territories,action=None):
        # on va copier l'etat du monde, remplir aleatoirement pour les autres joueurs (? ou appeller leur IA ? triche)
        # puis on cree un noeud avec l'etat d'apres et on continue
        #todo clean, et repasser sur gym
        self.territories = territories
        self.player = player
        self.players = player.ai.game.players
        self.empty = [ t for t in self.territories if t.owner == None ]
        self.value = 0
        self.action=action
        #0 if first, 1 if second etc
        self.play_order=player.ai.game.turn_order.index(self.player.name)
        #taken from "au automated technique for drafting territories in the board game risk
        # "An Automated Technique for Drafting Territories in the Board Game RIsk" FOR 3 PLAYERS
        self.territory_weights = { "Australia" : [2.97,0,8.45,9.99,10.71],
                                   "South America" : [0.69,1.23,3.90,0,17.72],
                                   "Africa" : [14.40,12.87,10.72,7.16,1.23,0,29.80],
                                   "North America" : [3.11,0.98,0,2.17,7.15,19.35,24.82,24.10,36.15,48.20],
                                   "Europe" : [42.33,45.11,43.11,43.77,41.35,50.77,43.85,36.93],
                                   "Asia" : [27.10,23.90,23.61,23.10,23.61,23.68,19.32,15.63,17.43,13.84,10.25,6.66,3.07]
                                   }
        self.unique_enemy_weight = -0.07
        self.pair_friendly_weight = 0.96
        #dropped the coefficient "first to play" cause we dont have an impact on this

    def next_random_state(self):
        #travailler sur noms pour rapidité
        terri = deepcopy(self.territories)
        empt = [ t for t in terri if t.owner == None ]
        tt = random.choice(empt)
        empt.remove(tt)
        tt.owner = self.player
        action = tt
        #in case we run out of territories in the middle : todo
        for i in islice(self.players.values(),self.play_order+1,None,1):
            if len(empt)==0:
                break
            t = random.choice(empt)
            empt.remove(t)
            t.owner = i
        if self.play_order > 0:
            for i in islice(self.players.values(),0,self.play_order,1):
                if len(empt)==0:
                    break
                t = random.choice(empt)
                empt.remove(t)
                t.owner = i
        return MCTSState(self.player,terri,action)
                
    def reward(self):
        player_scores = {}
        for player in self.players.values():
            score = 0
            unique_enemy = set()
            allied_pairs = 0
            for t in player.territories:
                for u in t.adjacent(None,None):
                    if u.owner != None and u.owner != player:
                        unique_enemy.add(u)
                    elif u.owner == player:
                        allied_pairs = allied_pairs + 0.5
            score = len(unique_enemy)*self.unique_enemy_weight + allied_pairs *  self.pair_friendly_weight
            for area in self.player.world.areas.keys():
                count = 0
                for t in self.player.territories:
                    if t.area.name == area:
                        count = count + 1
                score = score + self.territory_weights[area][count]
            player_scores[player.name]=score
        return player_scores[self.player.name]/sum(player_scores.values())

    def terminal(self):
        if(len(self.empty)==0):
            return True
        return False

    def __hash__(self):
        return int(hashlib.md5(str(self.territories).encode('utf-8')).hexdigest(),16)

    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False

def UCTSearch(state0):
    #we set 100 prediction loop by default, TODO
    node0 = MCTSNode(state0)
    for i in range(2):
        node1 = TreePolicy(node0)
        reward = DefaultPolicy(node1.state)
        Backup(node1,reward)
    x = BestChild(node0,0).state.action
    return x
    
def TreePolicy(node):
    while not node.state.terminal():
        if not node.fully_expanded():
            #todo balance
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
    return node.children[-1]

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
        return "Error"
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
