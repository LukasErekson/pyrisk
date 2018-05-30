from ai import AI
import random
import math
import collections
from ai.mcts_tree import MCTSState, UCTSearch
from copy import deepcopy

class SD_BetterAI(AI):
    """
    Smart drafting : plays randomly like the stupid AI, except for the drafting (placement) phase where it uses MCTS with UCT to find the best countries.
    """

    def start(self):
        #0 if first, 1 if second etc
        self.play_order=self.game.turn_order.index(self.player.name)
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
        self.previous_action = []
        #dropped the coefficient "first to play" cause we dont have an impact on this

    def initial_placement(self, empty, remaining):
        if empty:
            action = UCTSearch(MCTSState(self.player,deepcopy(list(self.world.territories.values())),self.previous_action))
            self.previous_action = action
            return action[-1]
        else:
            t = random.choice(list(self.player.territories))
            return t

    def priority(self):
        priority = sorted([t for t in self.player.territories if t.border], 
                          key=lambda x: self.area_priority.index(x.area.name))
        priority = [t for t in priority if t.area == priority[0].area]
        return priority if priority else list(self.player.territories)
   

        
    def reinforce(self, available):
        priority = self.priority()
        result = collections.defaultdict(int)
        while available:
            result[random.choice(priority)] += 1
            available -= 1
        return result

    def attack(self):
        for t in self.player.territories:
            if t.forces > 1:
                adjacent = [a for a in t.connect if a.owner != t.owner and t.forces >= a.forces + 3]
                if len(adjacent) == 1:
                        yield (t.name, adjacent[0].name, 
                               lambda a, d: a > d, None)
                else:
                    total = sum(a.forces for a in adjacent)
                    for adj in adjacent:
                        yield (t, adj, lambda a, d: a > d + total - adj.forces + 3, 
                               lambda a: 1)
    
    def freemove(self):
        srcs = sorted([t for t in self.player.territories if not t.border], 
                      key=lambda x: x.forces)
        if srcs:
            src = srcs[-1]
            n = src.forces - 1
            return (src, self.priority()[0], n)
        return None
