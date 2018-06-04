#!/usr/bin/python3
import random
import math
import hashlib
from itertools import islice
from world import AREAS


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
        #does it even work?
    def fully_expanded(self):
        if len(self.children)==len(self.state.empty):
            return True
        return False

    def __repr__(self):
        s="Node; children %d; visits %d; reward %d"%(len(self.children),self.visits,self.reward)
        return s

class MCTSState(object):
    """on a oublié la value"""
    def __init__(self,player,territories,action=None):
        # on va copier l'etat du monde, remplir aleatoirement pour les autres joueurs (? ou appeller leur IA ? triche)
        # puis on cree un noeud avec l'etat d'apres et on continue
        #todo clean, et repasser sur gym
        self.territories = territories
        self.player = player
        self.players = player.ai.game.players
        self.empty = [ name for name,owner in self.territories.items() if owner == None ]
        self.value = 0
        self.action=action
        #0 if first, 1 if second etc
        self.play_order=player.ai.game.turn_order.index(self.player.name)
        #taken from "au automated technique for drafting territories in the board game risk
        # "An Automated Technique for Drafting Territories in the Board Game RIsk" FOR 3 PLAYERS
        
        #dropped the coefficient "first to play" cause we dont have an impact on this

    def next_random_state(self):
        #travailler sur noms pour rapidité
        terri = self.territories.copy()
        empt = [ name for name,owner in self.territories.items() if owner == None ]
        tt = random.choice(empt)
        empt.remove(tt)
        terri[tt] = self.player.name
        action = tt
        #in case we run out of territories in the middle : todo
        for i in islice(self.players.values(),self.play_order+1,None,1):
            if len(empt)==0:
                break
            t = random.choice(empt)
            empt.remove(t)
            terri[t] = i
        if self.play_order > 0:
            for i in islice(self.players.values(),0,self.play_order,1):
                if len(empt)==0:
                    break
                t = random.choice(empt)
                empt.remove(t)
                terri[t] = i
        return MCTSState(self.player,terri,tt)
                
    def reward(self):
        player_scores = {}
        for player in self.players.values():
            score = 0
            unique_enemy = set()
            allied_pairs = 0
            for t in self.territories.keys():
                for u in TERRITORIES_NEIGHBOUR[t]:
                    if self.territories[u] != None and self.territories[u] != player.name:
                        unique_enemy.add(u)
                    elif self.territories[u] == player.name:
                        allied_pairs = allied_pairs + 0.5
            score = len(unique_enemy)*UNIQUE_ENEMY_WEIGHT + allied_pairs *  PAIR_FRIENDLY_WEIGHT
            for area,list_terri in AREA_TERRITORIES.items():
                count = 0
                for terri in list_terri:
                    if self.territories[terri] == self.player.name:
                        count = count + 1
                score = score + AREA_WEIGHT[area][count]
            #just for 3 players
            if self.play_order == 0:
                score = score + 13.38
            elif self.play_order == 1:
                score = score + 5.35
            player_scores[player.name]=max(score,0)
        player_rewards = {}
        for player in self.players.values():
            player_rewards[player.name]=player_scores[player.name]/sum(player_scores.values())
        self.player_rewards = player_rewards
        return player_rewards

    def terminal(self):
        if(len(self.empty)==0):
            return True
        return False

    def __hash__(self):
        # à ameliorer
        return int(hashlib.md5((str(self.territories)+str(self.action)).encode('utf-8')).hexdigest(),16)

    def __eq__(self,other):
        if hash(self)==hash(other):
            return True
        return False

    def __repr__(self):
        s="Empty=%s;Action=%s"%(str(len(self.empty)),str(self.action))
        return s




