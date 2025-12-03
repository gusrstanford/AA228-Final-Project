import random
import csv
from collections import defaultdict
import simulator as sim
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_all_actions(num_intersections):
    # each intersection phase ∈ {0,1}
    return [tuple(a) for a in itertools.product([0, 1], repeat=num_intersections)]

def encode_state_global(state, num_intersections):
    """
    Global coarse encoding:

    - state is flat from env.getState():
      [q0_up, q0_left, q0_down, q0_right, phase0, q1_up, ...]
    - return (main_bin, side_bin, phase_pattern)

    main_bin / side_bin are 3-level bins of total cars on main/side roads.
    phase_pattern is an integer 0..(2^num_intersections - 1) encoding current phases.
    """
    state = np.asarray(state, dtype=int)

    total_main = 0
    total_side = 0
    phases = []

    for i in range(num_intersections):
        base = i * 5
        up, left, down, right, phase = state[base: base + 5]
        total_main += up + down
        total_side += left + right
        phases.append(int(phase))

    def bin_count(x):
        # You can tune these thresholds; this is a simple 3-bin scheme
        if x == 0:
            return 0
        elif x <= 6:
            return 1
        else:
            return 2

    main_bin = bin_count(total_main)
    side_bin = bin_count(total_side)

    # pack phases into a single integer, like a bit pattern
    phase_pattern = 0
    for p in phases:
        phase_pattern = (phase_pattern << 1) | p

    return (main_bin, side_bin, phase_pattern)


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
        print(f"Starting episode {ep+1}/{num_episodes}")
        raw_state = env.reset()
        s_enc = encode_state_global(raw_state, num_intersections)
        total_reward = 0.0

        for t in range(episode_length):
            a = epsilon_greedy_action(s_enc)

            next_raw_state, reward = env.step(list(a))
            s_next_enc = encode_state_global(next_raw_state, num_intersections)
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
        s_enc = encode_state_global(state, num_intersections)
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

def plot_comparison_rewards(num_intersections_list, rewards_rl, rewards_baseline, title="Policy Reward Comparison"):
    plt.figure(figsize=(8, 5))
    plt.plot(num_intersections_list, rewards_rl, marker='o', label='Q-learning Policy')
    plt.plot(num_intersections_list, rewards_baseline, marker='s', label='Baseline Policy')
    plt.xlabel('Number of Intersections')
    plt.ylabel('Average Reward per Episode')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_sim_on_multiple_intersections(
        traffic,
        max_intersections=6,
        num_episodes=2000,
        episode_length=300,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.3,
        baseline_freq=5
    ):
    """
    Run RL and baseline for 1..max_intersections intersections for a given traffic setting.
    Returns:
        num_list, rl_rewards, baseline_rewards, results_list (for CSV)
    """
    main_prob, side_prob = traffic
    rl_rewards = []
    baseline_rewards = []
    results_list = []

    for n in range(1, max_intersections + 1):
        print(f"\n=== Traffic {traffic}, num_intersections = {n} ===")
        intersection_list = [sim.Intersection(i) for i in range(n)]
        env = sim.TrafficSimulator(
            intersection_list,
            main_prob,
            side_prob,
            main_reward_power=2,
            side_reward_power=1,
            max_queue_size=8
        )

        # Q-learning
        Q, _ = q_learning(env,
                          num_episodes=num_episodes,
                          episode_length=episode_length,
                          alpha=alpha,
                          gamma=gamma,
                          epsilon=epsilon)
        policy = greedy_policy(Q, env)
        eval_rewards = evaluate_policy(env, policy, num_episodes=100, episode_length=episode_length)
        avg_rl = float(np.mean(eval_rewards))
        print("  Avg evaluation reward (RL):", avg_rl)

        # Baseline
        baseline = baseline_policy(env, freq=baseline_freq)
        baseline_eval = evaluate_policy(env, baseline, num_episodes=100, episode_length=episode_length)
        avg_baseline = float(np.mean(baseline_eval))
        print("  Avg baseline reward:", avg_baseline)

        rl_rewards.append(avg_rl)
        baseline_rewards.append(avg_baseline)

        results_list.append({
            "main_prob": main_prob,
            "side_prob": side_prob,
            "num_intersections": n,
            "avg_rl_reward": avg_rl,
            "avg_baseline_reward": avg_baseline
        })

    num_list = list(range(1, max_intersections + 1))
    return num_list, rl_rewards, baseline_rewards, results_list


def save_results_to_csv(results_list, filename):
    """
    results_list: list of dicts with keys:
        main_prob, side_prob, num_intersections, avg_rl_reward, avg_baseline_reward
    """
    if not results_list:
        print("No results to save.")
        return

    fieldnames = ["main_prob", "side_prob", "num_intersections",
                  "avg_rl_reward", "avg_baseline_reward"]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)
    print(f"Saved results to {filename}")

if __name__ == "__main__":
    heavy_traffic   = (0.7, 0.2)
    uniform_traffic = (0.1, 0.1)
    light_traffic   = (0.2, 0.05)

    all_results = []

    for name, traffic in [("heavy", heavy_traffic),
                          ("uniform", uniform_traffic),
                          ("light", light_traffic)]:
        if name != "uniform":
            continue

        num_list, rl_rewards, baseline_rewards, results = run_sim_on_multiple_intersections(
            traffic,
            max_intersections=6,
            num_episodes=6000,
            episode_length=300,
            alpha=0.1,
            gamma=0.95,
            epsilon=0.3,
            baseline_freq=5
        )

        plot_comparison_rewards(
            num_list,
            rl_rewards,
            baseline_rewards,
            title=f"Reward vs # Intersections ({name} traffic)"
        )

        all_results.extend(results)

    save_results_to_csv(all_results, "traffic_rl_vs_baseline_results.csv")
