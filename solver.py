import random
from collections import defaultdict
import simulator as sim
import itertools

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

    # defaultdict so unseen Qs start at 0.0
    Q = defaultdict(float)

    def epsilon_greedy_action(state_tuple):
        if random.random() < epsilon:
            return random.choice(action_space)
        # greedy
        best_a = None
        best_q = -float('inf')
        for a in action_space:
            q = Q[(state_tuple, a)]
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    rewards_per_episode = []

    for ep in range(num_episodes):
        state = env.reset()
        s = tuple(state)
        total_reward = 0.0

        for t in range(episode_length):
            # choose action
            a = epsilon_greedy_action(s)

            # interact with env
            next_state, reward = env.step(list(a))  # env expects list
            s_next = tuple(next_state)
            total_reward += reward

            # TD target
            # max_a' Q(s', a')
            max_next_q = max(Q[(s_next, a_prime)] for a_prime in action_space)

            # Q-update
            old_q = Q[(s, a)]
            Q[(s, a)] = old_q + alpha * (reward + gamma * max_next_q - old_q)

            s = s_next

        rewards_per_episode.append(total_reward)
        # you can add simple logging if you want

    return Q, rewards_per_episode

def baseline_policy(env,freq=5):
    env.reset()
    num_intersections = len(env.intersections)

    # alternate the phases every 'freq' timesteps, basic baseline for how mostr traffic lights work
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
        s = tuple(state)
        best_a = None
        best_q = -float('inf')
        for a in action_space:
            q = Q[(s, a)]
            if q > best_q:
                best_q = q
                best_a = a
        return list(best_a)  # env expects list
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

if __name__ == "__main__":
    main_prop = 0.7
    side_prop = 0.3
    intersection_list = [sim.Intersection(i) for i in range(3)]
    env = sim.TrafficSimulator(intersection_list, main_prop, side_prop, main_reward_power=2, side_reward_power=1)
    Q, rewards = q_learning(env, num_episodes=500)
    policy = greedy_policy(Q, env)
    eval_rewards = evaluate_policy(env, policy, num_episodes=100)
    print("Average evaluation reward:", sum(eval_rewards) / len(eval_rewards))
    baseline = baseline_policy(env, freq=5)
    baseline_rewards = evaluate_policy(env, baseline, num_episodes=100)
    print("Average baseline reward:", sum(baseline_rewards) / len(baseline_rewards))