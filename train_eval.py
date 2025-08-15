"""
Training and evaluation script
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from fire_evacuation_rl import (
    FireEvacuationEnv,
    train_qmp_dqn,
    QMatrixDQN
)


def evaluate_agent(env, agent, n_episodes=100):
    """Evaluate agent performance"""
    print(f"\nEvaluating agent performance ({n_episodes} episodes)...")
    
    scores = []
    people_evacuated = []
    evacuation_rates = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
        scores.append(total_reward)
        people_evacuated.append(info['total_evacuated'])
        evacuation_rates.append(info['evacuation_rate'])
        
        if episode % 20 == 0:
            print(f"  Evaluation Episode {episode}: Evacuated {info['total_evacuated']}/{env.initial_total_people} people ({info['evacuation_rate']:.1%})")
    
    results = {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_evacuated': np.mean(people_evacuated),
        'std_evacuated': np.std(people_evacuated),
        'mean_evacuation_rate': np.mean(evacuation_rates),
        'std_evacuation_rate': np.std(evacuation_rates),
        'max_evacuation_rate': max(evacuation_rates),
        'min_evacuation_rate': min(evacuation_rates)
    }
    
    print("\nEvaluation Results:")
    print(f"Average score: {results['mean_score']:.2f} ± {results['std_score']:.2f}")
    print(f"Average evacuated: {results['mean_evacuated']:.1f} ± {results['std_evacuated']:.1f}")
    print(f"Average evacuation rate: {results['mean_evacuation_rate']:.1%} ± {results['std_evacuation_rate']:.1%}")
    print(f"Maximum evacuation rate: {results['max_evacuation_rate']:.1%}")
    print(f"Minimum evacuation rate: {results['min_evacuation_rate']:.1%}")
    
    return results


def train(args):
    """Training function"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("Fire Evacuation RL Training - Q-Matrix Transfer Learning")
    print("="*70)
    
    start_time = time.time()
    
    print(f"\nUsing single environment for training...")
    agent = train_qmp_dqn(
        mesh_dir=args.mesh_dir,
        episodes=args.episodes,
        max_nodes=args.max_nodes
    )
        
    elapsed = time.time() - start_time
    print(f"\nTraining complete! Total time: {elapsed/60:.1f} minutes")
    
    print("\n" + "="*70)
    print("Final Model Evaluation")
    print("="*70)
    
    from fire_evacuation_rl import FireEvacuationEnv
    env = FireEvacuationEnv(args.mesh_dir, max_nodes=args.max_nodes)
    
    best_model_path = output_dir / "best_model.pth"
    if best_model_path.exists():
        agent.load_model(str(best_model_path))
        print("Loading best model for evaluation...")
    
    eval_results = evaluate_agent(env, agent, n_episodes=50)
    
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nAll results saved to: {output_dir}")


def evaluate(args):
    """Evaluation function"""
    from fire_evacuation_rl import FireEvacuationEnv, QMatrixDQN
    
    print("Creating evaluation environment...")
    env = FireEvacuationEnv(mesh_dir=args.mesh_dir, max_nodes=args.max_nodes)
    
    print("Creating agent...")
    agent = QMatrixDQN(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        adjacency_matrix=env.adjacency_matrix
    )
    
    model_path = Path(args.output_dir) / "best_model.pth"
    if not model_path.exists():
        model_path = Path(args.output_dir) / "final_model.pth"
        
    if model_path.exists():
        agent.load_model(str(model_path))
        print(f"Model loaded successfully: {model_path}")
    else:
        print(f"Warning: Model file not found")
        return
        
    results = evaluate_agent(env, agent, args.episodes)
    
    with open(Path(args.output_dir) / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nEvaluation results saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fire evacuation RL training and evaluation")
    
    parser.add_argument("--mesh-dir", default="preprocess_data", help="Mesh data directory")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--max-nodes", type=int, default=50, help="Maximum number of nodes")
    parser.add_argument("--mode", choices=['train', 'eval'], required=True, help="Run mode")
    parser.add_argument("--episodes", type=int, default=3000, help="Training/evaluation episodes")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()