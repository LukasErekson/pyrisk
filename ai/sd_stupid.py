from ai import AI
import random
import math
import collections
from ai.mctsstate import MCTSState, define_neighbours
from ai.mcts import MCTS

class SD_StupidAI(AI):
    """
    Smart drafting : plays randomly like the stupid AI, except for the drafting (placement) phase where it uses MCTS with UCT to find the best countries.
    """

    def start(self):
        define_neighbours(self.player.world)
        self.mcts = MCTS()

    def initial_placement(self, empty, remaining):
        if empty:
            terri = {t.name: (None if t.owner == None else t.owner.name) for t in self.world.territories.values()}
            action = self.mcts.UCTSearch(MCTSState(self.player, terri))
            return action
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
