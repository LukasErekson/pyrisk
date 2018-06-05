from ai import AI
import random
import math
import collections
from ai.mctsstate import MCTSState, define_neighbours
from ai.mcts import MCTS

class SD_BetterAI(AI):
    """
    Smart drafting : plays randomly like the stupid AI, except for the drafting (placement) phase where it uses MCTS with UCT to find the best countries.
    """

    def start(self):
        define_neighbours(self.player.world)
        self.mcts = MCTS()
        self.area_priority = list(self.world.areas)
        random.shuffle(self.area_priority)


    def initial_placement(self, empty, remaining):
        if empty:
            terri = {t.name: (None if t.owner == None else t.owner.name) for t in self.world.territories.values()}
            action = self.mcts.UCTSearch(MCTSState(self.player, terri))
            return action
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
