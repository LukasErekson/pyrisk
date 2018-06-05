#!/bin/python3

import random
from itertools import islice
from world import AREAS
import hashlib


def define_neighbours(world):
    for te in list(world.territories.values()):
        MCTSState.TERRITORIES_NEIGHBOUR[te.name] = list([t.name for t in list(te.adjacent(None, None))])


class MCTSState(object):
    # CONSTS
    AREA_WEIGHT = {"Australia": [2.97, 0, 8.45, 9.99, 10.71],
                   "South America": [0.69, 1.23, 3.90, 0, 17.72],
                   "Africa": [14.40, 12.87, 10.72, 7.16, 1.23, 0, 29.80],
                   "North America": [3.11, 0.98, 0, 2.17, 7.15, 19.35, 24.82, 24.10, 36.15, 48.20],
                   "Europe": [42.33, 45.11, 43.11, 43.77, 41.35, 50.77, 43.85, 36.93],
                   "Asia": [27.10, 23.90, 23.61, 23.10, 23.61, 23.68, 19.32, 15.63, 17.43, 13.84, 10.25, 6.66, 3.07]}
    UNIQUE_ENEMY_WEIGHT = -0.07
    PAIR_FRIENDLY_WEIGHT = 0.96

    AREA_TERRITORIES = {key: value[1] for (key, value) in AREAS.items()}
    TERRITORIES_NEIGHBOUR = {}

    def __init__(self, player, territories, action=None):
        # todo clean, et repasser sur gym
        self.territories = territories
        self.player = player
        self.players = player.ai.game.players
        self.empty = [name for name, owner in self.territories.items() if owner == None]
        self.value = 0
        self.all_values = []
        self.action = action
        # 0 if first, 1 if second etc
        self.play_order = player.ai.game.turn_order.index(self.player.name)

    #peut etre passer sur la "montercarlorollout", une fonction dans MCTS qui va juste ajouter un territoire, s'appeller
    #recursivement puis l'enlever
    def next_random_state(self):
        terri = self.territories.copy()
        empt = [name for name, owner in self.territories.items() if owner == None]
        action = random.choice(empt)
        empt.remove(action)
        terri[action] = self.player.name
        return MCTSState(self.players[self.player.ai.game.turn_order[(self.play_order+1)%len(self.players)]], terri, action)

    def values(self):
        player_scores = {}
        for player in self.players.values():
            score = 0
            unique_enemy = set()
            allied_pairs = 0
            for t in self.territories.keys():
                for u in MCTSState.TERRITORIES_NEIGHBOUR[t]:
                    if self.territories[u] is not None and self.territories[u] != player.name:
                        unique_enemy.add(u)
                    elif self.territories[u] == player.name:
                        allied_pairs = allied_pairs + 0.5
            score = len(unique_enemy) * MCTSState.UNIQUE_ENEMY_WEIGHT + allied_pairs * MCTSState.PAIR_FRIENDLY_WEIGHT
            for area, list_terri in MCTSState.AREA_TERRITORIES.items():
                count = 0
                for terri in list_terri:
                    if self.territories[terri] == self.player.name:
                        count = count + 1
                score = score + MCTSState.AREA_WEIGHT[area][count]
            # just for 3 players
            if self.play_order == 0:
                score = score + 13.38

            elif self.play_order == 1:
                score = score + 5.35
            player_scores[player.name] = max(score, 0)
        player_rewards = {}
        for player in self.players.values():
            player_rewards[player.name] = player_scores[player.name] / sum(player_scores.values())
        self.all_values = player_rewards
        self.value = player_rewards[self.player.name]
        # .values() ?
        return self.all_values.values()

    def terminal(self):
        if len(self.empty) == 0:
            return True
        return False

    def __hash__(self):
        return int(hashlib.md5((str(self.territories)).encode('utf-8')).hexdigest(), 16)

    def __eq__(self, other):
        if hash(self) == hash(other):
            return True
        return False

    def __repr__(self):
        s = "Empty=%s;Action=%s" % (str(len(self.empty)), str(self.action))
        return s
