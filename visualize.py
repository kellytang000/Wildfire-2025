
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import numpy as np
import torch

from fire_evacuation_rl import FireEvacuationEnv, QMatrixDQN

def visualize_environment(mesh_dir="preprocess_data"):
    """Visualize environment"""
    mesh_dir = Path(mesh_dir)
    
    # Read data
    nodes = gpd.read_file(mesh_dir / "mesh_nodes.geojson")
    edges = gpd.read_file(mesh_dir / "mesh_edges.geojson")
    shelters = gpd.read_file(mesh_dir / "shelters.geojson")
    fires = gpd.read_file(mesh_dir / "fires.geojson")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Draw edges
    edges.plot(ax=ax, color='gray', linewidth=1, alpha=0.5)
    
    # Draw nodes
    nodes.plot(ax=ax, color='blue', markersize=50, alpha=0.6)
    
    # Draw exits
    shelters.plot(ax=ax, color='green', markersize=200, marker='s', 
                  edgecolor='black', linewidth=2, label='Exits')
    
    # Draw fires
    fires.plot(ax=ax, color='red', alpha=0.3, edgecolor='darkred', 
               linewidth=2, label='Fire Areas')
    
    # Add node labels
    for idx, node in nodes.iterrows():
        x, y = node.geometry.x, node.geometry.y
        ax.text(x, y, str(idx), fontsize=12, ha='center', va='center', 
                bbox=dict(boxstyle="circle,pad=0.1", facecolor="white", alpha=0.8))
    
    ax.set_title("Evacuation Environment Visualization", fontsize=16)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('environment.png', dpi=150, bbox_inches='tight')
    print("Environment map saved to: environment.png")
    plt.show()
    
def visualize_evacuation_path(agent, env, episode_limit=200):
    """Visualize evacuation path"""
    state, _ = env.reset()
    
    # Record people distribution at each time step
    people_history = [state.copy()]
    actions_history = []
    rewards_history = []
    
    done = False
    steps = 0
    total_reward = 0
    
    while not done and steps < episode_limit:
        action = agent.act(state, training=False)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        people_history.append(state.copy())
        actions_history.append(action)
        rewards_history.append(reward)
        total_reward += reward
        steps += 1
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: People distribution heatmap
    people_matrix = np.array(people_history).T
    im1 = ax1.imshow(people_matrix, cmap='hot', aspect='auto', interpolation='nearest')
    ax1.set_title(f"People Distribution Changes (Total steps: {steps}, Total reward: {total_reward:.1f})")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Room ID")
    plt.colorbar(im1, ax=ax1, label='Number of People')
    
    # Right plot: Evacuation progress
    total_people = [p.sum() for p in people_history]
    evacuated = [people_history[0].sum() - p for p in total_people]
    
    ax2.plot(total_people, 'b-', label='Remaining People', linewidth=2)
    ax2.plot(evacuated, 'g-', label='Evacuated People', linewidth=2)
    ax2.axhline(y=people_history[0].sum(), color='r', linestyle='--', alpha=0.5, label='Initial Total People')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Number of People')
    ax2.set_title('Evacuation Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    ax2.text(0.02, 0.98, f"Evacuated: {info['total_evacuated']}\nRemaining: {info['people_remaining']}", 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('evacuation_result.png', dpi=150, bbox_inches='tight')
    print("Evacuation result saved to: evacuation_result.png")
    plt.show()
    
    return people_history, actions_history

def plot_training_progress(scores_file='training_scores.npy'):
    """Visualize training progress"""
    if Path(scores_file).exists():
        scores = np.load(scores_file)
    else:
        # Generate sample data
        print("Training data not found, generating sample data...")
        scores = np.random.randn(1000) * 50 - 100
        scores = np.convolve(scores, np.ones(100)/100, mode='valid')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Score curve
    ax1.plot(scores, alpha=0.5, color='blue', linewidth=0.5)
    
    # Moving average
    window = min(100, len(scores) // 10)
    if len(scores) > window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax1.plot(moving_avg, color='red', linewidth=2, label=f'{window}-Episode Average')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Success rate
    success = [1 if s > 0 else 0 for s in scores]
    if len(success) > window:
        success_rate = np.convolve(success, np.ones(window)/window, mode='valid') * 100
        ax2.plot(success_rate, color='green', linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Evacuation Success Rate')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("Training progress saved to: training_progress.png")
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh-dir', default='preprocess_data', help='Data directory')
    parser.add_argument('--mode', choices=['env', 'path', 'train'], 
                       default='env', help='Visualization mode')
    parser.add_argument('--model-path', default='results/model_final.pth', 
                       help='Model path')
    args = parser.parse_args()
    
    if args.mode == 'env':
        visualize_environment(args.mesh_dir)
    elif args.mode == 'path':
        # Load trained model
        env = FireEvacuationEnv(args.mesh_dir, max_nodes=50)
        agent = QMatrixDQN(env.observation_space.shape[0], env.action_space.n, env.adjacency_matrix)
        
        if Path(args.model_path).exists():
            checkpoint = torch.load(args.model_path, map_location=agent.device)
            agent.q_network.load_state_dict(checkpoint['q_network'])
            agent.epsilon = 0.0
            print("Model loaded successfully")
        else:
            print(f"Warning: Model file not found {args.model_path}, using random policy")
            
        visualize_evacuation_path(agent, env)
    else:
        plot_training_progress()

if __name__ == "__main__":
    main()