import numpy as np
from queue import Queue
import matplotlib.pyplot as plt

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
    def __init__(self, intersection_list, main_prob, side_prob, main_reward_power=2, side_reward_power=1, max_queue_size=8):
        # track if the simulator is dropping cars because lines are filling up
        self.num_missed_cars = 0
        self.timestep = 0
        self.max_queue_size = max_queue_size
        self.intersections = intersection_list
        self.main_prob = main_prob
        self.side_prob = side_prob
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
        self.num_missed_cars = 0
        self.timestep = 0
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
                # only spawns cars from the top of the main road
                if d == 0 and iid == 0:
                    # main entry top
                    if np.random.rand() < self.main_prob:
                        if intersection.intersection_queues[d].qsize() < self.max_queue_size:
                            intersection.intersection_queues[d].put(1)
                        else:
                            self.num_missed_cars += 1
                # only spawns cars from the bottom of the main road
                elif d == 2 and iid == len(self.intersections)-1:
                    # main entry bottom
                    if np.random.rand() < self.main_prob:
                        if intersection.intersection_queues[d].qsize() < self.max_queue_size:
                            intersection.intersection_queues[d].put(1)
                        else:
                            self.num_missed_cars += 1
                elif d in [1, 3]:
                    # spawns cars on all the side roads
                    if np.random.rand() < self.side_prob:
                        if intersection.intersection_queues[d].qsize() < self.max_queue_size:
                            intersection.intersection_queues[d].put(1)
                        else:
                            self.num_missed_cars += 1

        self.state = self.getState()
        reward = self.calculate_reward()
        return self.state, reward
    


def draw_state(state, light_phases):
    """
    Visualize a traffic state as a vertical corridor of intersections.

    - state is the flat numpy array returned by env.getState():
      [q0_up, q0_left, q0_down, q0_right, phase0,
       q1_up, q1_left, q1_down, q1_right, phase1, ...]
    - light_phases is a list/array of 0/1 for each intersection:
      0 = vertical green (main up/down), 1 = horizontal green (side left/right)
    """

    state = np.asarray(state, dtype=int)
    num_intersections = len(light_phases)

    # Each intersection contributes 5 entries: 4 queues + phase
    expected_len = num_intersections * 5
    if len(state) < expected_len:
        raise ValueError(
            f"State length {len(state)} is too short for {num_intersections} intersections "
            f"(expected at least {expected_len})."
        )

    fig, ax = plt.subplots(figsize=(4, 6))

    # Draw the main vertical road line (x=0)
    ax.plot([0, 0], [0, num_intersections - 1], linewidth=2, zorder=0)

    # For nicer plotting, put intersection 0 at the top
    for idx in range(num_intersections):
        # extract queues for this intersection
        base = idx * 5
        up_q, left_q, down_q, right_q, phase_in_state = state[base: base + 5]

        # y-position: flip so 0 is at top
        y = (num_intersections - 1) - idx

        # Draw light as 4 small circles in a square:
        # vertical (up/down) = main road, horizontal (left/right) = side road
        light_phase = light_phases[idx]
        color_vert = "green" if light_phase == 0 else "red"
        color_horiz = "red" if light_phase == 0 else "green"

        r = 0.07   # radius of each little light
        d = 0.13   # distance from center

        # top / bottom = vertical directions (0,2)
        top_circle    = plt.Circle((0,    y + d), r, color=color_vert)
        bottom_circle = plt.Circle((0,    y - d), r, color=color_vert)
        # left / right = horizontal directions (1,3)
        left_circle   = plt.Circle((-d,   y),     r, color=color_horiz)
        right_circle  = plt.Circle(( d,   y),     r, color=color_horiz)

        ax.add_patch(top_circle)
        ax.add_patch(bottom_circle)
        ax.add_patch(left_circle)
        ax.add_patch(right_circle)



        # MAIN ROAD queues (up/down) drawn vertically:
        #   up_q above the light, down_q below the light
        for j in range(up_q):
            ax.plot(-0.1, y + 0.3 * (j + 1), "ko", markersize=4)
        for j in range(down_q):
            ax.plot(0.1, y - 0.3 * (j + 1), "ko", markersize=4)

        # SIDE ROAD queues (left/right) drawn horizontally:
        #   left_q to the left, right_q to the right
        for j in range(left_q):
            ax.plot(-0.3 * (j + 1), y, "ko", markersize=4)
        for j in range(right_q):
            ax.plot(0.3 * (j + 1), y, "ko", markersize=4)

    ax.set_ylim(-1, num_intersections)
    ax.set_xlim(-3, 3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Traffic State")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    intersections = [Intersection(i) for i in range(4)]
    env = TrafficSimulator(intersections, main_prob=0.5, side_prob=0.1)

    state = env.reset()
    draw_state(state, env.light_phases)

    for _ in range(5):
        action = [np.random.choice([0, 1]) for _ in range(4)]
        state, reward = env.step(action)
        print("Action:", action, "Reward:", reward)
        draw_state(state, env.light_phases)

