import random
from collections import defaultdict
import simulator as sim
import itertools

def encode_state(state, num_intersections):
    """
    Map raw queue lengths into a coarse, tabular-friendly state.

    state: flat array from env.getState()
           [q0_up, q0_left, q0_down, q0_right, phase0, ...]
    returns: tuple of ints, e.g.
       (main0_bin, side0_bin, phase0,
        main1_bin, side1_bin, phase1, ...)
    """
    s = []
    for i in range(num_intersections):
        base = i * 5
        up, left, down, right, phase = state[base:base+5]
        main_total = up + down
        side_total = left + right

        # simple 3-bin discretization
        def bin_count(x):
            if x == 0:
                return 0
            elif x <= 3:
                return 1
            else:
                return 2

        main_bin = bin_count(main_total)
        side_bin = bin_count(side_total)

        s.extend([main_bin, side_bin, int(phase)])
    return tuple(s)


def get_all_actions(num_intersections):
    # each intersection phase ∈ {0,1}
    return [tuple(a) for a in itertools.product([0, 1], repeat=num_intersections)]


def q_learning(env,
               num_episodes=500,
               episode_length=200,
               alpha=0.1,
               gamma=0.95,
               epsilon=0.1):

    num_intersections = len(env.intersections)
    action_space = get_all_actions(num_intersections)
    Q = defaultdict(float)

    def epsilon_greedy_action(state_enc):
        if random.random() < epsilon:
            return random.choice(action_space)
        best_a = None
        best_q = -float('inf')
        for a in action_space:
            q = Q[(state_enc, a)]
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    rewards_per_episode = []

    for ep in range(num_episodes):
        raw_state = env.reset()
        s_enc = encode_state(raw_state, num_intersections)
        total_reward = 0.0

        for t in range(episode_length):
            a = epsilon_greedy_action(s_enc)

            next_raw_state, reward = env.step(list(a))
            s_next_enc = encode_state(next_raw_state, num_intersections)
            total_reward += reward

            max_next_q = max(Q[(s_next_enc, a_prime)] for a_prime in action_space)

            old_q = Q[(s_enc, a)]
            Q[(s_enc, a)] = old_q + alpha * (reward + gamma * max_next_q - old_q)

            s_enc = s_next_enc

        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode



def baseline_policy(env, freq=5):
    env.reset()
    num_intersections = len(env.intersections)

    # alternate the phases every 'freq' timesteps
    def policy(state):
        action = env.light_phases
        if env.timestep % freq == 0:
            if action[0] == 0:
                action = [1] * num_intersections
            else:
                action = [0] * num_intersections
        return action

    return policy


def greedy_policy(Q, env):
    num_intersections = len(env.intersections)
    action_space = get_all_actions(num_intersections)

    def policy(state):
        s_enc = encode_state(state, num_intersections)
        best_a = None
        best_q = -float('inf')
        for a in action_space:
            q = Q[(s_enc, a)]
            if q > best_q:
                best_q = q
                best_a = a
        return list(best_a)
    return policy



def evaluate_policy(env, policy, num_episodes=50, episode_length=200):
    total_rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        total = 0.0
        for t in range(episode_length):
            action = policy(state)
            state, reward = env.step(action)
            total += reward
        total_rewards.append(total)
    return total_rewards


def visualize_policy(env, policy, steps=30, title="Policy visualization"):
    """
    Step through the environment with a given policy and draw each state.
    """
    state = env.reset()
    print(f"\n=== {title} ===")
    for t in range(steps):
        print(f"t = {env.timestep}, reward so far timestep {t}")
        # draw current state
        sim.draw_state(state, env.light_phases)
        # choose action and step
        action = policy(state)
        state, reward = env.step(action)
        print(f"  action: {action}, reward_t: {reward}")


if __name__ == "__main__":
    main_prop = 0.3
    side_prop = 0.1
    intersection_list = [sim.Intersection(i) for i in range(4)]
    env = sim.TrafficSimulator(
        intersection_list,
        main_prop,
        side_prop,
        main_reward_power=1.5,
        side_reward_power=1,
    )

    # 1) Train Q-learning policy
    Q, rewards = q_learning(env, num_episodes=1000, episode_length=300, alpha=0.1, gamma=0.95, epsilon=0.2)
    policy = greedy_policy(Q, env)
    eval_rewards = evaluate_policy(env, policy, num_episodes=100)
    print("Average evaluation reward (RL):", sum(eval_rewards) / len(eval_rewards))

    # 2) Baseline policy
    baseline = baseline_policy(env, freq=5)
    baseline_rewards = evaluate_policy(env, baseline, num_episodes=100)
    print("Average baseline reward:", sum(baseline_rewards) / len(baseline_rewards))

    # 3) Visual comparison (one rollout each, step-by-step plots)
    #visualize_policy(env, baseline, steps=30, title="Baseline policy")
    visualize_policy(env, policy, steps=30, title="Learned Q policy")
