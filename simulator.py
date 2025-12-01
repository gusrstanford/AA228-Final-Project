import numpy as np
from queue import Queue

class Intersection:
    def __init__(self, intersection_id):
        self.intersection_id = intersection_id
        # up, left, down, right
        self.intersection_queues = [Queue() for _ in range(4)]
        # 0: green vertical (up/down), 1: green horizontal (left/right)
        self.current_phase = 0  

    def set_phase(self, phase):
        self.current_phase = phase

    def get_Queue_lengths(self):
        return [q.qsize() for q in self.intersection_queues]
    
    def get_phase(self):
        return self.current_phase
    
    def reset_queues(self):
        for q in self.intersection_queues:
            while not q.empty():
                q.get()

    def get_ID(self):
        return self.intersection_id


class TrafficSimulator:
    def __init__(self, intersection_list, main_prop, side_prop, main_reward_power=2, side_reward_power=1):
        # track if the simulator is dropping cars because lines are filling up
        self.num_missed_cars = 0
        self.timestep = 0
        self.max_queue_size = 10
        self.intersections = intersection_list
        self.main_prop = main_prop
        self.side_prop = side_prop
        self.main_reward_power = main_reward_power
        self.side_reward_power = side_reward_power
        self.light_phases = [0] * len(self.intersections)
        self.state = self.getState()

    def getState(self):
        state = []
        for intersection in self.intersections:
            state.extend(intersection.get_Queue_lengths())
            state.append(intersection.get_phase())
        return np.array(state, dtype=int)

    def reset(self):
        for intersection in self.intersections:
            intersection.reset_queues()
            intersection.set_phase(0)
        self.light_phases = [0] * len(self.intersections)
        self.state = self.getState()
        return self.state
    
    def calculate_reward(self):
        reward = 0.0
        for intersection in self.intersections:
            q_lengths = intersection.get_Queue_lengths()
            for direction, length in enumerate(q_lengths):
                if direction in (0, 2):  # main road (up/down)
                    reward -= (length ** self.main_reward_power)
                else:  # side roads (left/right)
                    reward -= (length ** self.side_reward_power)
        return reward


    def step(self, action):
        self.timestep += 1
        n = len(self.intersections)
        self.light_phases = list(action)
        down = [0] * n  # car moving from i to i+1
        up   = [0] * n  # car moving from i to i-1

        # 1) apply signals and dequeue cars
        for i, intersection in enumerate(self.intersections):
            intersection.set_phase(action[i])
            if action[i] == 0:
                # green vertical
                for d in [0, 2]:  # up, down
                    if not intersection.intersection_queues[d].empty():
                        intersection.intersection_queues[d].get()
                        if d == 0 and i < n-1:
                            down[i] = 1
                        elif d == 2 and i > 0:
                            up[i] = 1
            else:
                # green horizontal
                for d in [1, 3]:  # left, right
                    if not intersection.intersection_queues[d].empty():
                        intersection.intersection_queues[d].get()

        # 2) move cars along main road
        for i, car in enumerate(down):
            if car == 1 and i < n-1:
                if self.intersections[i+1].intersection_queues[0].qsize() < self.max_queue_size:
                    self.intersections[i+1].intersection_queues[0].put(1)
                else:
                    self.num_missed_cars += 1

        for i, car in enumerate(up):
            if car == 1 and i > 0:
                if self.intersections[i-1].intersection_queues[2].qsize() < self.max_queue_size:
                    self.intersections[i-1].intersection_queues[2].put(1)
                else:
                    self.num_missed_cars += 1
        
        # 3) spawn new arrivals
        for intersection in self.intersections:
            iid = intersection.get_ID()
            for d in range(4):
                if d == 0 and iid == 0:
                    # main entry top
                    if np.random.rand() < self.main_prop:
                        if intersection.intersection_queues[d].qsize() < self.max_queue_size:
                            intersection.intersection_queues[d].put(1)
                        else:
                            self.num_missed_cars += 1
                elif d == 2 and iid == len(self.intersections)-1:
                    # main entry bottom
                    if np.random.rand() < self.main_prop:
                        if intersection.intersection_queues[d].qsize() < self.max_queue_size:
                            intersection.intersection_queues[d].put(1)
                        else:
                            self.num_missed_cars += 1
                else:
                    # side roads
                    if np.random.rand() < self.side_prop:
                        if intersection.intersection_queues[d].qsize() < self.max_queue_size:
                            intersection.intersection_queues[d].put(1)
                        else:
                            self.num_missed_cars += 1

        self.state = self.getState()
        reward = self.calculate_reward()
        return self.state, reward


if __name__ == "__main__":
    intersections = [Intersection(i) for i in range(3)]
    sim = TrafficSimulator(intersections, main_prop=0.3, side_prop=0.1)
    state = sim.reset()
    print("Initial State:", state)
    for _ in range(10):
        action = [np.random.choice([0,1]) for _ in range(3)]
        next_state, reward = sim.step(action)
        print("Action:", action)
        print("Next State:", next_state)
        print("Reward:", reward)
