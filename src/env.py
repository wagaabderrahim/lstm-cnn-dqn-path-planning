"""
GridWorld Environment for 2D Mobile Robot Path Planning.
Supports both full-map global observation and local egocentric windows.
"""

from typing import Tuple, Optional
import numpy as np

def a_star_path_exists(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    """Checks if an unobstructed path exists between start and goal using 8-connected A*."""
    h, w = grid.shape
    if grid[start] == 1.0 or grid[goal] == 1.0:
        return False
        
    def heuristic(pos: Tuple[int, int]) -> float:
        return np.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
        
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    
    open_set = [(0, start)]
    g_score = {start: 0.0}
    closed_set = set()
    
    while open_set:
        open_set.sort()
        _, current = open_set.pop(0)
        
        if current in closed_set:
            continue
        closed_set.add(current)
        
        if current == goal:
            return True
            
        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                neighbor = (nr, nc)
                if grid[neighbor] == 1.0 or neighbor in closed_set:
                    continue
                move_cost = 1.0 if abs(dr) + abs(dc) == 1 else 1.414
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    open_set.append((f_score, neighbor))
    return False


class GridWorld:
    """2D occupancy-grid environment with 8-directional kinematics and dense reward shaping."""

    def __init__(
        self,
        size: int = 15,
        obstacle_ratio: float = 0.18,
        max_steps: int = 200,
        seed: Optional[int] = None,
        obs_radius: Optional[int] = None,
    ) -> None:
        self.size = size
        self.obstacle_ratio = obstacle_ratio
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.obs_radius = obs_radius

        self.grid = np.zeros((size, size), dtype=np.float32)
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.agent = self.start
        self.steps = 0

    def reset(self, max_attempts: int = 100) -> np.ndarray:
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        
        for _ in range(max_attempts):
            self.grid.fill(0.0)
            num_obstacles = int(self.size * self.size * self.obstacle_ratio / 4)
            for _ in range(num_obstacles):
                h = self.rng.integers(1, 4)
                w = self.rng.integers(1, 4)
                r = self.rng.integers(0, self.size - h)
                c = self.rng.integers(0, self.size - w)
                self.grid[r : r + h, c : c + w] = 1.0

            self.grid[self.start] = 0.0
            self.grid[self.goal] = 0.0
            
            if a_star_path_exists(self.grid, self.start, self.goal):
                self.agent = self.start
                self.steps = 0
                return self._obs()
        
        self.agent = self.start
        self.steps = 0
        return self._obs()

    def set_grid(
        self,
        grid: np.ndarray,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Sets a pre-defined occupancy grid and initializes agent and goal positions."""
        self.grid = np.array(grid, dtype=np.float32).copy()
        self.size = self.grid.shape[0]
        self.start = start if start is not None else (0, 0)
        self.goal = goal if goal is not None else (self.size - 1, self.size - 1)
        self.agent = self.start
        self.steps = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        if self.obs_radius is None:
            obs = np.copy(self.grid)
            obs[self.agent] = -1.0
            obs[self.goal] = 2.0
            return obs
        else:
            R = self.obs_radius
            window_size = 2 * R + 1
            obs = np.ones((window_size, window_size), dtype=np.float32) * 1.0
            
            ar, ac = self.agent
            for i in range(window_size):
                for j in range(window_size):
                    gr = ar - R + i
                    gc = ac - R + j
                    if 0 <= gr < self.size and 0 <= gc < self.size:
                        obs[i, j] = self.grid[gr, gc]
            
            obs[R, R] = -1.0
            gr, gc = self.goal
            if abs(gr - ar) <= R and abs(gc - ac) <= R:
                obs[gr - ar + R, gc - ac + R] = 2.0
            return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        r, c = self.agent
        directions = [
            (-1, 0),  (1, 0),  (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]
        dr, dc = directions[action]
        r = min(max(r + dr, 0), self.size - 1)
        c = min(max(c + dc, 0), self.size - 1)

        self.steps += 1
        next_pos = (r, c)
        collision = self.grid[next_pos] == 1.0
        if collision:
            next_pos = self.agent

        old_dist = self._manhattan(self.agent, self.goal)
        new_dist = self._manhattan(next_pos, self.goal)
        progress_reward = (old_dist - new_dist) * 0.5
        step_penalty = -0.05
        collision_penalty = -0.3 if collision else 0.0
        reached_goal = next_pos == self.goal
        goal_bonus = 5.0 if reached_goal else 0.0

        reward = progress_reward + step_penalty + collision_penalty + goal_bonus
        self.agent = next_pos
        done = reached_goal or self.steps >= self.max_steps
        info = {"reached_goal": reached_goal}
        return self._obs(), reward, done, info

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
