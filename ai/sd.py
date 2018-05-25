from ai import AI
import random
import math
import collections

class SDAI(AI):
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
        #dropped the coefficient "first to play" cause we dont have an impact on this

    def value_of_land(self):
        player_scores = {}
        for player in self.game.players:
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
        
    def initial_placement(self, empty, remaining):
        if empty:
            return random.choice(empty)
        else:
            t = random.choice(list(self.player.territories))
            return t
        
    def attack(self):
        for t in self.player.territories:
            for a in t.connect:
                if a.owner != self.player:
                    if t.forces > a.forces:
                        yield (t, a, None, None)

    def reinforce(self, available):
        border = [t for t in self.player.territories if t.border]
        result = collections.defaultdict(int)
        for i in range(available):
            t = random.choice(border)
            result[t] += 1
        return result
