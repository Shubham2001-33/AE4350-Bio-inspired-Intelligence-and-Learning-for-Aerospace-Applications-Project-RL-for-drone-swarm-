'''
Generates environment based on configuration
'''

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np
import yaml

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)['env']
        self.obstacles = cfg['obstacles']

        super().__init__(
            grid_size = cfg['grid_size'],
            n_agents = cfg['agents'],
            obs_radius = cfg['obs_radius'],
            reward_type = cfg['reward_type'],
            seed = cfg['seed'],
        )

    def _gen_grid(self, width, height):
        # Set Seed
        self.np_random, _ = seeding.np_random(self.seed_val)

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate obstacles
        obstacle_list = []
        for _ in range(self.obstacles):
            while True:
                xy = (self.np_random.integers(1,width-1),self.np_random.integers(1,height-1))

                if self.grid.get(*xy) is None:
                    break

            self.grid.set(*xy,Wall())
            obstacle_list.append(xy)

        # Set all non-walls as uncovered
        self.grid.setAll(Uncovered())

        # Place Agents
        for i in range(self.agents.n_agents):
            while True:
                xy = (self.np_random.integers(1,width-1),self.np_random.integers(1,height-1))

                if xy not in self.agents.agent_pos.values() and xy not in obstacle_list:
                    self.agents.agent_pos[i] = xy
                    self.grid.set(*xy, None)
                    break

register(
    id='MiniGrid-Empty-8x8-v0',
    entry_point='gym_minigrid.envs:EmptyEnv'
)
