import os
import sys

# Allow imports from SMOOD_GitHub/* when running this script directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from datetime import datetime
import numpy as np
import torch
import gymnasium
import random
from torch.utils.tensorboard import SummaryWriter
from core.ppo import PPO, Memory, ActorCritic
from core.gym_env import ur5GymEnv

title = 'PyBullet UR5 robot'


def get_args():
    parser = argparse.ArgumentParser(description=title)
    arg = parser.add_argument
    # Environment settings
    arg('--render', action='store_true', default=False, help='render the environment')
    arg('--randObjPos', action='store_true', default=False, help='randomize object position each episode')
    arg('--mel', type=int, default=150, help='max episode length')
    arg('--repeat', type=int, default=2, help='repeat action')
    arg('--simgrip', action='store_true', default=False, help='simulated gripper')
    arg('--tol_xy', type=float, default=0.01, help='XY success tolerance (m)')
    arg('--tol_z', type=float, default=0.01, help='Z success tolerance (m)')

    # Training settings
    arg('--seed', type=int, default=987, help='random seed')
    arg('--emb_size', type=int, default=128, help='embedding size')
    arg('--solved_reward', type=int, default=0, help='stop training if avg_reward > solved_reward')
    arg('--log_interval', type=int, default=100, help='interval for log')
    arg('--save_interval', type=int, default=100, help='interval for saving model')
    arg('--max_episodes', type=int, default=15000, help='max training episodes')
    arg('--update_timestep', type=int, default=1000, help='update policy every n timesteps')
    arg('--action_std', type=float, default=0.2, help='std for Gaussian policy')
    arg('--K_epochs', type=int, default=4, help='update policy for K epochs')
    arg('--eps_clip', type=float, default=0.2, help='PPO clip parameter')
    arg('--gamma', type=float, default=0.99, help='discount factor')
    arg('--lr', type=float, default=1e-3, help='learning rate')
    arg('--betas', type=tuple, default=(0.9, 0.999))
    arg('--loss_entropy_c', type=float, default=0.01, help='entropy term coefficient')
    arg('--loss_value_c', type=float, default=0.5, help='value loss coefficient')
    arg('--save_dir', type=str, default='saved_rl_models/', help='model save directory')
    arg('--cuda', dest='cuda', action='store_true', default=False, help='use CUDA')
    arg('--device_num', type=str, default="0", help='GPU number to use')
    arg('--model_path', type=str, default=None, help='path to checkpoint to load and resume training from')

    return parser.parse_args()


args = get_args()
np.set_printoptions(precision=2)
torch.set_printoptions(profile="full", precision=2)

CP_G, CP_C = '\033[32m', '\033[0m'
args.device = torch.device('cuda:' + str(args.device_num) if args.cuda else 'cpu')
print('Using device:', 'cuda' if args.cuda else 'cpu', ', device number:', args.device_num, ', GPUs:', torch.cuda.device_count())


def main():
    args.env_name = title
    print(CP_G + 'Environment name:', args.env_name, '' + CP_C)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # Setup TensorBoard logging
    log_dir = os.path.join(args.save_dir, 'tensorboard_logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    # Create a unique run directory with timestamp
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed_{args.seed}"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
    print(f'[TensorBoard] Logging to: {os.path.join(log_dir, run_name)}')

    # Create environment with new interface
    env = ur5GymEnv(
        renders=args.render,
        maxSteps=args.mel,
        actionRepeat=args.repeat,
        randObjPos=args.randObjPos,
        simulatedGripper=args.simgrip
    )
    # set tolerances dynamically from args
    env.tol_xy = args.tol_xy
    env.tol_z = args.tol_z

    # Seeding
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # PPO setup
    memory = Memory()
    ppo = PPO(args, env)

    # Load checkpoint if specified
    if args.model_path and os.path.exists(args.model_path):
        print(f'[INFO]Loading pretrained model from: {args.model_path}')
        ppo.policy.load_state_dict(torch.load(args.model_path, map_location=args.device))

    # Logging variables
    running_reward, avg_length = 0, 0
    running_xy_dist, running_z_dist = 0, 0
    success_count, total_episodes = 0, 0
    total_steps, time_step = 0, 0

    initial_lr = args.lr
    print(f'Starting training with XY tolerance={args.tol_xy:.3f}m, Z tolerance={args.tol_z:.3f}m')

    for i_episode in range(1, args.max_episodes + 1):
        state, _ = env.reset()
        episode_success = False

        for t in range(args.mel):
            time_step += 1
            total_steps += 1

            action = ppo.select_action(state, memory)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            if 'xy_dist' in info:
                running_xy_dist += info['xy_dist']
            if 'z_dist' in info:
                running_z_dist += info['z_dist']

            if info.get('is_success', False):
                episode_success = True

            if time_step % args.update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0

            running_reward += reward
            if done:
                break

        avg_length += t
        total_episodes += 1
        if episode_success:
            success_count += 1

        # Learning rate decay
        if i_episode % 1000 == 0:
            new_lr = max(initial_lr * (0.8 ** (i_episode // 1000)), 3e-4)
            for g in ppo.optimizer.param_groups:
                g['lr'] = new_lr
            print(f"[LR Update] Episode {i_episode}: learning_rate = {new_lr:.6f}")

        # Save periodically
        if i_episode % args.save_interval == 0:
            torch.save(
                ppo.policy.state_dict(),
                os.path.join(args.save_dir, f'model_epoch_{int(i_episode / args.save_interval)}.pth')
            )

        # Logging + auto curriculum
        if i_episode % args.log_interval == 0:
            avg_length = int(avg_length / args.log_interval)
            avg_reward = int(running_reward / args.log_interval)
            avg_xy = running_xy_dist / total_steps if total_steps > 0 else 0
            avg_z = running_z_dist / total_steps if total_steps > 0 else 0
            success_rate = 100.0 * success_count / total_episodes if total_episodes > 0 else 0.0

            print(
                f"Episode {i_episode:4d} | AvgLen: {avg_length:3d} | "
                f"AvgRew: {avg_reward:5d} | XY: {avg_xy:.4f} m | "
                f"Z: {avg_z:.4f} m | Success: {success_rate:.1f}%"
            )

            # Log to TensorBoard
            writer.add_scalar('Training/Reward', avg_reward, i_episode)
            writer.add_scalar('Training/XY_Error', avg_xy, i_episode)
            writer.add_scalar('Training/Z_Error', avg_z, i_episode)
            writer.add_scalar('Training/Avg_Length', avg_length, i_episode)
            writer.add_scalar('Training/Success_Rate', success_rate, i_episode)

            # Reset running stats
            running_reward = avg_length = 0
            running_xy_dist = running_z_dist = 0
            total_steps = 0
            success_count = total_episodes = 0

            # Warm-up phase
            if i_episode < 500:
                continue

            # Adaptive tolerance tightening
            if success_rate > 80.0:
                env.tol_xy = max(env.tol_xy * 0.95, 0.01)
                env.tol_z = max(env.tol_z * 0.95, 0.01)
                print(f"[ADAPT] New tolerances → XY={env.tol_xy:.3f} m, Z={env.tol_z:.3f} m")

            # Auto-refine precision stage
            if avg_reward > args.solved_reward:
                if env.tol_xy > 0.01:
                    env.tol_xy = max(env.tol_xy * 0.95, 0.01)
                    env.tol_z = max(env.tol_z * 0.95, 0.01)
                    print(f"[AUTO-REFINE] New tighter tolerances → XY={env.tol_xy:.3f}, Z={env.tol_z:.3f}")
                    continue
                elif success_rate > 90.0:
                    print("########## Solved to precision ##########")
                    torch.save(
                        ppo.policy.state_dict(),
                        os.path.join(args.save_dir, 'model_precise.pth')
                    )
                    break

    writer.close()
    print('[TensorBoard] Logging completed and writer closed.')


if __name__ == '__main__':
    main()

