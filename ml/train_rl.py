"""
Q-Learning Training Script for Dynamic Route Rationalization
==============================================================
Trains a Q-table agent to learn optimal routes on the Nashik road
network using the RL environment defined in rl_env.py.

The Q-table is indexed by discrete states:
  (current_node, destination_node, time_bucket, weather_idx)
and maps to action values for each possible neighbour transition.

Outputs:
  - q_table.pkl   (trained Q-table dictionary)
  - rl_stats.pkl  (training statistics for analysis)
"""

import os
import sys
import time
import numpy as np
import joblib
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.rl_env import (
    NashikRoutingEnv, NODES_LIST, NODE_TO_IDX, IDX_TO_NODE,
    ADJACENCY, WEATHER_LIST, time_to_bucket
)


def train_q_learning(
    env,
    n_episodes=50000,
    alpha=0.1,          # Learning rate
    gamma=0.95,         # Discount factor
    epsilon_start=1.0,  # Initial exploration rate
    epsilon_end=0.05,   # Final exploration rate
    epsilon_decay=0.9995,  # Epsilon decay per episode
    log_interval=5000,
):
    """
    Train a Q-learning agent on the Nashik routing environment.

    The Q-table is a nested dict:  Q[state][action] = value
    State = (current_node, dest_node, time_bucket, weather_idx)
    Action = index into the neighbour list of the current node

    Returns
    -------
    Q       : dict – the trained Q-table
    stats   : dict – training statistics
    """
    Q = defaultdict(lambda: defaultdict(float))

    epsilon = epsilon_start
    total_rewards = []
    success_count = 0
    episode_lengths = []

    print(f"\n  Training Q-Learning Agent")
    print(f"  {'=' * 40}")
    print(f"  Episodes        : {n_episodes:,}")
    print(f"  Learning rate   : {alpha}")
    print(f"  Discount factor : {gamma}")
    print(f"  Epsilon         : {epsilon_start} -> {epsilon_end}")
    print(f"  Nodes           : {env.n_nodes}")
    print(f"  Max steps/ep    : {env.max_steps}")
    print(f"  {'=' * 40}\n")

    start_time = time.time()

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            n_actions = env.get_valid_actions()

            # Epsilon-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                # Pick the best action from Q-table
                q_vals = [Q[state][a] for a in range(n_actions)]
                action = int(np.argmax(q_vals))

            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # Q-learning update
            n_next_actions = env.get_valid_actions()
            if done:
                max_future_q = 0.0
            else:
                max_future_q = max(
                    Q[next_state][a] for a in range(n_next_actions)
                ) if n_next_actions > 0 else 0.0

            Q[state][action] = Q[state][action] + alpha * (
                reward + gamma * max_future_q - Q[state][action]
            )

            state = next_state

        # Track statistics
        total_rewards.append(episode_reward)
        episode_lengths.append(env.steps)
        if env.current_node == env.destination:
            success_count += 1

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Periodic logging
        if episode % log_interval == 0:
            recent_rewards = total_rewards[-log_interval:]
            avg_reward = np.mean(recent_rewards)
            recent_lengths = episode_lengths[-log_interval:]
            avg_length = np.mean(recent_lengths)
            recent_successes = sum(
                1 for r in recent_rewards if r > 0
            )
            success_rate = recent_successes / log_interval * 100

            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed

            print(
                f"  Episode {episode:>6,} | "
                f"Avg Reward: {avg_reward:>8.1f} | "
                f"Avg Steps: {avg_length:>4.1f} | "
                f"Success: {success_rate:>5.1f}% | "
                f"e: {epsilon:.4f} | "
                f"{eps_per_sec:,.0f} ep/s"
            )

    elapsed = time.time() - start_time
    overall_success = success_count / n_episodes * 100

    print(f"\n  {'=' * 50}")
    print(f"  Training Complete!")
    print(f"  {'=' * 50}")
    print(f"  Total time      : {elapsed:.1f}s")
    print(f"  Q-table entries : {len(Q):,}")
    print(f"  Overall success : {overall_success:.1f}%")
    print(f"  Final epsilon   : {epsilon:.4f}")
    print(f"  {'=' * 50}")

    stats = {
        "n_episodes": n_episodes,
        "total_rewards": total_rewards,
        "episode_lengths": episode_lengths,
        "success_count": success_count,
        "success_rate": overall_success,
        "elapsed_time": elapsed,
        "final_epsilon": epsilon,
    }

    return dict(Q), stats


def evaluate_agent(env, Q, n_tests=500):
    """
    Evaluate the trained Q-table agent on random source-destination pairs.
    Returns success rate and average travel time for successful routes.
    """
    successes = 0
    total_times = []

    for _ in range(n_tests):
        state = env.reset()
        done = False
        total_time = 0.0

        while not done:
            n_actions = env.get_valid_actions()
            q_vals = [Q.get(state, {}).get(a, 0.0) for a in range(n_actions)]
            action = int(np.argmax(q_vals))

            next_state, reward, done, info = env.step(action)
            total_time += info.get("eff_time", 0)
            state = next_state

        if env.current_node == env.destination:
            successes += 1
            total_times.append(total_time)

    success_rate = successes / n_tests * 100
    avg_time = np.mean(total_times) if total_times else 0

    print(f"\n  Evaluation Results ({n_tests} random routes)")
    print(f"  {'=' * 40}")
    print(f"  Success rate    : {success_rate:.1f}%")
    print(f"  Avg travel time : {avg_time:.1f} min (successful routes)")
    print(f"  Failed routes   : {n_tests - successes}")
    print(f"  {'=' * 40}")

    return success_rate, avg_time


def demo_route(env, Q, source_name, dest_name, time_of_day=9, weather="clear"):
    """
    Demonstrate a single route found by the RL agent.
    """
    source = NODE_TO_IDX[source_name]
    dest = NODE_TO_IDX[dest_name]

    state = env.reset(
        source=source, destination=dest,
        time_of_day=time_of_day, day_of_week=1, weather=weather,
    )

    path = [source_name]
    total_time = 0.0
    done = False

    while not done:
        n_actions = env.get_valid_actions()
        q_vals = [Q.get(state, {}).get(a, 0.0) for a in range(n_actions)]
        action = int(np.argmax(q_vals))

        next_state, reward, done, info = env.step(action)
        path.append(info["node"])
        total_time += info.get("eff_time", 0)
        state = next_state

    success = env.current_node == env.destination
    status = "OK" if success else "FAILED"

    print(f"\n  {source_name} -> {dest_name} [{weather}, {time_of_day}:00]")
    print(f"  Path: {' -> '.join(path)}")
    print(f"  Time: {total_time:.1f} min  |  {status}")

    return path, total_time, success


if __name__ == "__main__":
    print("=" * 60)
    print("  Dynamic Route Rationalization — RL Agent Training")
    print("=" * 60)

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    model_path = os.path.join(script_dir, "model.pkl")
    encoder_path = os.path.join(script_dir, "weather_encoder.pkl")
    q_table_path = os.path.join(script_dir, "q_table.pkl")
    stats_path = os.path.join(script_dir, "rl_stats.pkl")

    # Load XGBoost model
    ml_model = None
    weather_encoder = None
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        ml_model = joblib.load(model_path)
        weather_encoder = joblib.load(encoder_path)
        print(f"  [OK] Loaded XGBoost model from {model_path}")
        print(f"  [OK] Loaded weather encoder from {encoder_path}")
    else:
        print("  [WARN] XGBoost model not found — using heuristic congestion")
        print("         Run ml/train_model.py first for better results")

    # Create environment
    env = NashikRoutingEnv(
        ml_model=ml_model,
        weather_encoder=weather_encoder,
        max_steps=20,
    )

    # Train Q-learning agent
    Q, stats = train_q_learning(
        env,
        n_episodes=10000,
        alpha=0.1,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        log_interval=1000,
    )

    # Evaluate
    success_rate, avg_time = evaluate_agent(env, Q, n_tests=500)

    # Demo routes
    print(f"\n  Demo Routes:")
    print(f"  {'=' * 50}")
    demo_route(env, Q, "CBS Chowk", "Mhasrul", time_of_day=9, weather="clear")
    demo_route(env, Q, "Panchavati", "Nashik Road", time_of_day=18, weather="rain")
    demo_route(env, Q, "Gangapur Road", "Deolali", time_of_day=12, weather="clear")
    demo_route(env, Q, "Ambad", "Mumbai Naka", time_of_day=8, weather="fog")

    # Save Q-table and stats
    joblib.dump(Q, q_table_path)
    joblib.dump(stats, stats_path)

    print(f"\n  [OK] Q-table saved to: {q_table_path}")
    print(f"       Size: {os.path.getsize(q_table_path) / 1024:.1f} KB")
    print(f"  [OK] Stats saved to: {stats_path}")
    print(f"\n  RL training complete! Agent is ready for deployment.")
    print("=" * 60)
