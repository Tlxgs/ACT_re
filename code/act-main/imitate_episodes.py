import torch
import numpy as np
import os
import re
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed

# ---------------------------------------------------------------------------
# checkpoint / resume utilities
#
# Three kinds of files live in ckpt_dir:
#   policy_best.ckpt  - raw state_dict of the best validation epoch (what --eval loads)
#   policy_last.ckpt  - raw state_dict of the most recent checkpoint interval
#   train_state.pt    - self-contained resume state: model + optimizer + epoch +
#                       loss histories + best-val bookkeeping
# The .ckpt files stay raw state_dicts on purpose, so upstream eval/tooling that
# does `policy.load_state_dict(torch.load(path))` keeps working unchanged.
# ---------------------------------------------------------------------------

CKPT_BEST = 'policy_best.ckpt'
CKPT_LAST = 'policy_last.ckpt'
TRAIN_STATE = 'train_state.pt'
_ARCHIVE_RE = re.compile(r'^policy_epoch_(\d+)_seed_(\d+)\.ckpt$')


def atomic_torch_save(obj, path):
    """Write via a .tmp file + os.replace, so an interrupted run can never
    leave a half-written file where a valid checkpoint used to be."""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _parse_archive_name(name):
    m = _ARCHIVE_RE.match(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def list_archived_ckpts(ckpt_dir, seed):
    """[(epoch, path), ...] sorted by epoch, for this seed's archived checkpoints."""
    out = []
    if not os.path.isdir(ckpt_dir):
        return out
    for name in os.listdir(ckpt_dir):
        parsed = _parse_archive_name(name)
        if parsed is None or parsed[1] != seed:
            continue
        out.append((parsed[0], os.path.join(ckpt_dir, name)))
    return sorted(out)


def prune_archived_ckpts(ckpt_dir, seed, keep_last_k):
    """Keep only the newest `keep_last_k` archived checkpoints of this seed.
    Returns the list of removed file names. Only ever touches files matching
    policy_epoch_<e>_seed_<seed>.ckpt that this script itself produced."""
    if keep_last_k <= 0:
        return []
    items = list_archived_ckpts(ckpt_dir, seed)
    removed = []
    for _, path in items[:-keep_last_k]:
        try:
            os.remove(path)
            removed.append(os.path.basename(path))
        except OSError as exc:
            print(f'[ckpt] could not remove {path}: {exc}')
    return removed


def resume_signature(cfg):
    """Comparable snapshot of everything that has to match for a safe resume."""
    pc = cfg.get('policy_config') or {}
    return {
        'task_name': cfg.get('task_name'),
        'policy_class': cfg.get('policy_class'),
        'state_dim': cfg.get('state_dim'),
        'episode_len': cfg.get('episode_len'),
        'batch_size': cfg.get('batch_size'),
        'kl_weight': pc.get('kl_weight'),
        'chunk_size': pc.get('num_queries'),
        'hidden_dim': pc.get('hidden_dim'),
        'dim_feedforward': pc.get('dim_feedforward'),
        'backbone': pc.get('backbone'),
        'lr': cfg.get('lr'),
        'seed': cfg.get('seed'),
        'camera_names': list(cfg.get('camera_names') or []),
    }


def resolve_resume(ckpt_dir, resume, resume_ckpt, seed):
    """
    Decide where to continue training from. Returns
        {'model_path': str|None, 'state_path': str|None, 'start_epoch': int, 'source': str}

    state_path -> train_state.pt, authoritative: model + optimizer + history + epoch
    model_path -> weights-only start, optimizer cold-started
    """
    info = {'model_path': None, 'state_path': None, 'start_epoch': 0, 'source': 'fresh'}

    if resume_ckpt:
        path = resume_ckpt
        if not os.path.isfile(path):
            cand = os.path.join(ckpt_dir, resume_ckpt)
            if os.path.isfile(cand):
                path = cand
            else:
                raise FileNotFoundError(f'--resume_ckpt not found: {resume_ckpt}')
        parsed = _parse_archive_name(os.path.basename(path))
        if parsed is None:
            print(f'[resume] WARNING: no epoch in the file name of {os.path.basename(path)}; '
                  f'the global epoch counter restarts at 0. Use --resume to continue the same run, '
                  f'or point --ckpt_dir at a separate directory when branching off a checkpoint.')
        info.update(model_path=path,
                    start_epoch=(parsed[0] + 1) if parsed else 0,
                    source=f'weights-only (--resume_ckpt {path})')
        return info

    if not resume:
        return info

    state_path = os.path.join(ckpt_dir, TRAIN_STATE)
    if os.path.isfile(state_path):
        info.update(state_path=state_path, source=f'full resume ({state_path})')
        return info

    last_path = os.path.join(ckpt_dir, CKPT_LAST)
    if os.path.isfile(last_path):
        info.update(model_path=last_path,
                    source=f'weights-only, train_state.pt missing ({last_path})')
        return info

    archived = list_archived_ckpts(ckpt_dir, seed)
    if archived:
        epoch, path = archived[-1]
        info.update(model_path=path, start_epoch=epoch + 1,
                    source=f'weights-only, train_state.pt missing ({path})')
        return info

    print(f'[resume] no checkpoint found under {ckpt_dir} -> starting from scratch')
    return info


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'batch_size': batch_size_train,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'num_rollouts': args['num_rollouts'],
        # checkpoint / resume
        'resume': args['resume'],
        'resume_ckpt': args['resume_ckpt'],
        'save_every': args['save_every'],
        'epochs_per_run': args['epochs_per_run'],
        'archive_ckpts': args['archive_ckpts'],
        'keep_last_k': args['keep_last_k'],
        'no_optim_state': args['no_optim_state'],
        'eval_ckpt': args['eval_ckpt'],
    }

    if is_eval:
        ckpt_names = [config['eval_ckpt'] or CKPT_BEST]
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)

    if best_ckpt_info is None:
        print('[train] session finished without training; existing checkpoints left untouched.')
        return
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint (only if this session actually produced a new best;
    # otherwise the policy_best.ckpt already on disk is kept as-is)
    if best_state_dict is None:
        print(f'[train] best val loss stays {min_val_loss:.6f} @ epoch{best_epoch}; '
              f'{CKPT_BEST} on disk left untouched')
    else:
        ckpt_path = os.path.join(ckpt_dir, CKPT_BEST)
        atomic_torch_save(best_state_dict, ckpt_path)
        print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    ckpt_obj = torch.load(ckpt_path, map_location='cuda')
    if isinstance(ckpt_obj, dict) and 'model' in ckpt_obj:  # train_state.pt passed via --eval_ckpt
        ckpt_obj = ckpt_obj['model']
    loading_status = policy.load_state_dict(ckpt_obj)
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = config.get('num_rollouts', 50)
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']          # GLOBAL total, not "this session"
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    save_every = max(1, int(config.get('save_every') or 5))
    keep_last_k = int(config.get('keep_last_k') or 0)
    archive_ckpts = bool(config.get('archive_ckpts'))
    save_optim_state = not bool(config.get('no_optim_state'))
    epochs_per_run = int(config.get('epochs_per_run') or 0)

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    # ---------------- resume ----------------
    resume_info = resolve_resume(ckpt_dir, config.get('resume'), config.get('resume_ckpt'), seed)
    start_epoch = resume_info['start_epoch']
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_epoch = None
    best_state_dict = None
    best_dirty = False

    if resume_info['state_path'] is not None:
        state = torch.load(resume_info['state_path'], map_location='cuda')
        policy.load_state_dict(state['model'])
        if state.get('optimizer') is None:
            print('[resume] train_state.pt carries no optimizer state -> AdamW cold start')
        elif save_optim_state:
            optimizer.load_state_dict(state['optimizer'])
            print('[resume] AdamW moments restored')
        else:
            print('[resume] optimizer state present but --no_optim_state given -> AdamW cold start')
        start_epoch = int(state['epoch']) + 1
        train_history = state.get('train_history', [])
        validation_history = state.get('validation_history', [])
        min_val_loss = float(state.get('min_val_loss', np.inf))
        best_epoch = state.get('best_epoch')
        best_summary = f' @ epoch{best_epoch}' if best_epoch is not None else ''
        print(f'[resume] full state restored: last completed epoch {state["epoch"]}, '
              f'{len(validation_history)} epoch(s) of loss history, best val loss {min_val_loss:.6f}{best_summary}')
        # a silent hyperparameter drift on resume is a classic way to waste hours
        prev_sig = resume_signature(state.get('config') or {})
        curr_sig = resume_signature(config)
        mismatches = [f'{k}: {prev_sig[k]} -> {curr_sig[k]}'
                      for k in curr_sig
                      if prev_sig.get(k) is not None and prev_sig[k] != curr_sig[k]]
        if mismatches:
            print('[resume] WARNING: these settings differ from the checkpoint being resumed:')
            for m in mismatches:
                print(f'[resume]   {m}')
            print('[resume]   (weights still load, but the loss history / LR scale may no longer be comparable)')
    elif resume_info['model_path'] is not None:
        policy.load_state_dict(torch.load(resume_info['model_path'], map_location='cuda'))
        print(f'[resume] weights loaded from {resume_info["model_path"]}; AdamW cold start, '
              f'loss history empty, resuming at epoch {start_epoch}')
    if resume_info['source'] != 'fresh':
        print(f'[resume] source: {resume_info["source"]}')

    end_epoch = num_epochs
    if epochs_per_run > 0:
        end_epoch = min(num_epochs, start_epoch + epochs_per_run)

    if start_epoch >= end_epoch:
        print(f'\n[train] nothing to do: start epoch {start_epoch} >= target {end_epoch} '
              f'(global target {num_epochs}). Raise --num_epochs to keep training.')
        if len(validation_history) > 0:
            _plot_loss_curves(train_history, validation_history, ckpt_dir, seed, history_offset=0)
        return (best_epoch, min_val_loss, None)

    print(f'\n[train] epochs {start_epoch} -> {end_epoch - 1} '
          f'({end_epoch - start_epoch} this session, global target {num_epochs}), '
          f'checkpoint every {save_every} epoch(s) -> {ckpt_dir}\n')

    # move the dataloader / dropout streams forward so a resumed session does not
    # replay the exact same batch order; a fresh run (start_epoch = 0) is unchanged
    set_seed(seed + start_epoch)

    # Global epoch index of the first recorded history entry.
    #   fresh run          : start_epoch = 0,  empty history   -> 0
    #   weights-only resume: start_epoch = 14, empty history   -> 14
    #   full resume        : start_epoch = 34, 20 entries       -> 14
    # A full resume restores history that already begins *before* start_epoch, so the
    # offset must be start_epoch - len(history), not a flat 0 (that mislabelled the
    # x-axis by len(history) epochs).
    history_offset = start_epoch - len(validation_history)

    for epoch in tqdm(range(start_epoch, end_epoch)):
        print(f'\nEpoch {epoch}')
        train_len_before = len(train_history)

        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_epoch = epoch
                best_state_dict = deepcopy(policy.state_dict())
                best_dirty = True
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[train_len_before:])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # ---------------- checkpoint ----------------
        # every `save_every` epochs, and always on the last epoch of this session
        if (epoch + 1) % save_every == 0 or epoch == end_epoch - 1:
            model_sd = policy.state_dict()
            if best_dirty and best_state_dict is not None:
                atomic_torch_save(best_state_dict, os.path.join(ckpt_dir, CKPT_BEST))
                best_dirty = False
                print(f'[ckpt] {CKPT_BEST} updated (val loss {float(min_val_loss):.6f} @ epoch{best_epoch})')
            atomic_torch_save(model_sd, os.path.join(ckpt_dir, CKPT_LAST))
            state_obj = {
                'epoch': epoch,
                'seed': seed,
                'model': model_sd,
                'optimizer': optimizer.state_dict() if save_optim_state else None,
                'train_history': train_history,
                'validation_history': validation_history,
                'min_val_loss': float(min_val_loss),
                'best_epoch': best_epoch,
                'num_epochs': num_epochs,
                'config': {k: v for k, v in config.items() if not k.startswith('_')},
            }
            atomic_torch_save(state_obj, os.path.join(ckpt_dir, TRAIN_STATE))
            print(f'[ckpt] epoch {epoch} saved -> {CKPT_LAST}, {TRAIN_STATE}')
            if archive_ckpts:
                arch_name = f'policy_epoch_{epoch}_seed_{seed}.ckpt'
                atomic_torch_save(model_sd, os.path.join(ckpt_dir, arch_name))
                removed = prune_archived_ckpts(ckpt_dir, seed, keep_last_k)
                print(f'[ckpt] archived {arch_name}'
                      + (f'; pruned {len(removed)}: {", ".join(removed)}' if removed else ''))
            _plot_loss_curves(train_history, validation_history, ckpt_dir, seed, history_offset)

    if end_epoch < num_epochs:
        print(f'\n[train] session cap reached at epoch {end_epoch - 1} '
              f'({end_epoch - start_epoch} epoch(s) this run).')
        print(f'[train] continue with the SAME command plus --resume '
              f'(global target {num_epochs}, {num_epochs - end_epoch} epoch(s) remaining)')
    else:
        print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    return (best_epoch, float(min_val_loss), best_state_dict)


def _plot_loss_curves(train_history, validation_history, ckpt_dir, seed, history_offset):
    """Plot with a global x-axis: history_offset is the global epoch index of the
    first recorded entry (0 for a fresh run or a full resume)."""
    if len(validation_history) == 0:
        return
    last_epoch = history_offset + len(validation_history) - 1
    plot_history(train_history, validation_history, last_epoch, ckpt_dir, seed,
                 epoch_offset=history_offset)



def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed, epoch_offset=0):
    # save training curves
    if len(train_history) == 0:
        return
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        # epoch_offset shifts the x-axis so a session resumed from a weights-only
        # checkpoint still lines its curve up with the global epoch index
        train_x = np.linspace(epoch_offset, num_epochs, len(train_history)) if len(train_history) > 1 else np.array([num_epochs])
        val_x = np.linspace(epoch_offset, num_epochs, len(validation_history)) if len(validation_history) > 1 else np.array([num_epochs])
        plt.plot(train_x, train_values, label='train')
        plt.plot(val_x, val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()  # checkpointing every few epochs would otherwise leak figures
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--num_rollouts', action='store', type=int, default=50, required=False)

    # checkpoint / resume
    parser.add_argument('--resume', action='store_true',
                        help='continue from <ckpt_dir>/train_state.pt (falls back to the latest checkpoint if missing)')
    parser.add_argument('--resume_ckpt', action='store', type=str, default=None,
                        help='start from this checkpoint (weights only; epoch parsed from policy_epoch_<e>_seed_<s>.ckpt)')
    parser.add_argument('--save_every', action='store', type=int, default=5,
                        help='checkpoint interval in epochs (default 5)')
    parser.add_argument('--epochs_per_run', action='store', type=int, default=0,
                        help='stop after this many epochs in this session (0 = run up to --num_epochs). '
                             '--num_epochs is always the GLOBAL total, not the per-session count')
    parser.add_argument('--archive_ckpts', action='store_true',
                        help='also keep policy_epoch_<e>_seed_<s>.ckpt at every checkpoint interval')
    parser.add_argument('--keep_last_k', action='store', type=int, default=3,
                        help='with --archive_ckpts, keep only the newest K archived checkpoints (0 = keep all)')
    parser.add_argument('--no_optim_state', action='store_true',
                        help='omit AdamW moments from train_state.pt (smaller checkpoints, optimizer cold-starts)')
    parser.add_argument('--eval_ckpt', action='store', type=str, default=None,
                        help='checkpoint file to evaluate (default: policy_best.ckpt)')

    main(vars(parser.parse_args()))
