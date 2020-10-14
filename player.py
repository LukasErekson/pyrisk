from copy import deepcopy
from territory import Territory

class Player(object):
    def __init__(self, name, game, ai_class, ai_kwargs):
        self.name = name
        self.color = 0
        self.ord = 32
        self.ai = ai_class(self, game, game.world, **ai_kwargs)
        self.world = game.world
        self.area_control_counts = {a: 0 for a in self.world.areas.values()}
        self.max_draft_count = 0  # Measure the maximum draft count of the game
        self.turn_attack_count = 0  # How many attack actions were taken per turn
        self.total_troops_conquered = 0  # How many troops this player defeated

    @property
    def territories(self):
        for t in self.world.territories.values():
            if t.owner == self:
                yield t


    @property
    def territory_count(self):
        count = 0
        for t in self.world.territories.values():
            if t.owner == self:
                count += 1
        return count

    @property
    def areas(self):
        for a in self.world.areas.values():
            if a.owner == self:
                yield a

    @property
    def continent_control(self, area):
        """Returns the number of rounds which the player had control of
        the specific area/continent.

        Parameters
        ----------
            area (Area) : The area we're interested in finding the
                number of rounds under this player's control of.
        
        Returns
        -------
            int : Number of rounds this player had this area under control.
        """
        return self.area_control_counts[area.name]

    @property    
    def forces(self):
        return sum(t.forces for t in self.territories)

    @property
    def alive(self):
        return self.territory_count > 0

    @property
    def reinforcements(self):
        return max(self.territory_count//3, 3) + sum(a.value for a in self.areas)

    def __repr__(self):
        return "P;%s;%s" % (self.name, self.ai.__class__.__name__)

    def __hash__(self):
        return hash(("player", self.name))
        
    def __eq__(self, other):
        if isinstance(other, Player):
            return self.name == other.name
        return False

    def __deepcopy__(self, memo):
        newobj = Player(self.name, self, lambda *x, **y: None, {})
        newobj.color = self.color
        newobj.ord = self.ord
        newobj.world = deepcopy(self.world, memo)
        return newobj
