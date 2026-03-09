"""
modified from reset_env_rendered_episode.py & noisy_action_modifier.py
10 step마다 command displacement / real displacement의 누적 합(sum)을 계산해서 출력 및 그래프 표시
"""

from copy import deepcopy
import time
from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import os

from calvin_env.envs.tasks import Tasks

np.float = float

replay_action_type = "rel"  # 'abs', 'rel'
env_reset_period = None  # None, int
add_noise = False  # True, False. yet only rel_action modes.
noise_period = 10  # None, int
dist_measure_period = 10  # None, int
target_dataset_root_dir = "/data3/ksshin/datasets/CALVIN/calvin_debug_dataset/training"

action_from = 'eval' # 'dataset', 'eval'

def noise(action, pos_std=0.01, rot_std=1):
    """
    adds gaussian noise to position and orientation.
    units are m for pos and degree for rot
    """
    pos, orn, gripper = action
    rot_std = np.radians(rot_std)
    pos_noise = np.random.normal(0, pos_std, 3)
    rot_noise = p.getQuaternionFromEuler(np.random.normal(0, rot_std, 3))
    pos, orn = p.multiplyTransforms(pos, orn, pos_noise, rot_noise)
    return pos, orn, gripper


@hydra.main(config_path="../../conf", config_name="config_data_collection")
def run_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

    root_dir = Path(target_dataset_root_dir)

    ann_path = root_dir / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_path, allow_pickle=True).item()
    indx_ranges = ann["info"]["indx"]
    task_names = ann["language"]["task"]

    tasks = hydra.utils.instantiate(cfg.tasks)
    prev_info = None
    t1 = time.time()

    save_root = root_dir / (
        f"replay_frames_lang_ranges_"
        f"[replay_action_type({replay_action_type})_"
        f"env_reset_period({env_reset_period})_"
        f"add_noise({add_noise})_"
        f"noise_period({noise_period})_"
        f"dist_measure_period({dist_measure_period})_SUM]"
    )
    save_root.mkdir(parents=True, exist_ok=True)

    for ridx, (start, end) in enumerate(indx_ranges):
        print(f"\n=== range {ridx}: {start} ~ {end} ===")
        save_dir = save_root

        step_ids = []

        # step별 거리 저장
        cmd_step_dists = []
        real_step_dists = []

        # dist_measure_period마다 계산된 "sum" 저장
        cmd_period_sums = []
        cmd_period_x = []
        real_period_sums = []
        real_period_x = []

        local_step = 0
        prev_tcp_pos = None

        for i in range(start, end + 1):
            file = root_dir / f"episode_{i:07d}.npz"
            if not file.exists():
                print(f"[skip] missing: {file}")
                continue

            data = np.load(file)

            if env_reset_period is not None:
                if (i - start) % env_reset_period == 0:
                    print(f"reset {i}")
                    env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                    prev_info = None
                    prev_tcp_pos = None
            else:
                if i == start:
                    print(f"reset {i}")
                    env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                    prev_info = None
                    prev_tcp_pos = None

            if replay_action_type == "rel":
                action = data["rel_actions"]
            elif replay_action_type == "abs":
                action = data["actions"].astype(np.float32)
                pos = action[:3].astype(np.float32)
                euler = action[3:6].astype(np.float32)
                gripper = np.array([action[6]], dtype=np.float32)
                action = (pos, euler, gripper)
            else:
                raise ValueError("Wrong replay_action_type")

            if replay_action_type == "rel" and add_noise and noise_period is not None and i % noise_period == 0:
                pos = action[:3]
                orn = p.getQuaternionFromEuler(action[3:6])
                gripper = action[6]

                pos, orn, gripper = noise(
                    (pos, orn, gripper),
                    pos_std=10,
                    rot_std=10,
                )

                euler = p.getEulerFromQuaternion(orn)
                action = np.concatenate([pos, euler, [gripper]])

            # -------------------------
            # command displacement (step dist)
            # -------------------------
            cmd_xyz = np.asarray(action[:3], dtype=np.float32)
            cmd_dist = float(np.linalg.norm(cmd_xyz))
            cmd_step_dists.append(cmd_dist)

            # env step
            o, _, _, info = env.step(action)

            # -------------------------
            # real displacement (step dist)
            # -------------------------
            curr_tcp_pos = np.asarray(o["robot_obs"][:3], dtype=np.float32)
            if prev_tcp_pos is None:
                real_dist = 0.0
            else:
                real_dist = float(np.linalg.norm(curr_tcp_pos - prev_tcp_pos))
            prev_tcp_pos = curr_tcp_pos
            real_step_dists.append(real_dist)

            step_ids.append(local_step)

            # -------------------------
            # dist_measure_period마다 "sum" 계산/출력
            # -------------------------
            if dist_measure_period is not None and (local_step + 1) % dist_measure_period == 0:
                cmd_window = cmd_step_dists[-dist_measure_period:]
                real_window = real_step_dists[-dist_measure_period:]

                cmd_sum = float(np.sum(cmd_window))
                real_sum = float(np.sum(real_window))

                cmd_period_sums.append(cmd_sum)
                cmd_period_x.append(local_step)

                real_period_sums.append(real_sum)
                real_period_x.append(local_step)

                print(
                    f"[range {ridx:04d}] "
                    f"step {local_step - dist_measure_period + 1:04d} ~ {local_step:04d} | "
                    f"CMD sum={cmd_sum:.6f} | "
                    f"REAL sum={real_sum:.6f}"
                )

            local_step += 1

            print(info["scene_info"]["lights"]["led"]["logical_state"])
            if prev_info is not None:
                print(tasks.get_task_info(prev_info, info))
            prev_info = deepcopy(info)

            img = o["rgb_obs"]["rgb_static"]
            cv2.imwrite(str(save_dir / f"frame_{i:07d}.png"), img[:, :, ::-1])

            out = {
                "rgb_static": data["rgb_static"],
                "rgb_gripper": data["rgb_gripper"],
                "actions": data["actions"],
                "rel_actions": data["rel_actions"],
                "robot_obs": data["robot_obs"],
                "scene_obs": data["scene_obs"],
                "rgb_static_noisy": o["rgb_obs"]["rgb_static"],
                "rgb_gripper_noisy": o["rgb_obs"]["rgb_gripper"],
            }

            out_file = root_dir / f"episode_{i:07d}_noisy_action_image_added.npz"
            np.savez_compressed(out_file, **out)

            time.sleep(0.01)

        # =========================
        # 구간 그래프 저장
        # =========================
        if replay_action_type == "rel" and len(step_ids) > 0:
            if task_names is not None and ridx < len(task_names):
                task_title = str(task_names[ridx])
            else:
                task_title = f"range_{ridx:04d}"

            title_parts = [task_title]
            if len(cmd_step_dists) > 0:
                title_parts.append(f"cmd_step_mean={float(np.mean(cmd_step_dists)):.6f}")
            if len(real_step_dists) > 0:
                title_parts.append(f"real_step_mean={float(np.mean(real_step_dists)):.6f}")
            title = " | ".join(title_parts)

            plt.figure(figsize=(12, 4))

            # step dist(참고용으로 유지). 원치 않으면 이 두 plot 줄 삭제하면 됨.
            # plt.plot(
            #     step_ids,
            #     cmd_step_dists,
            #     linewidth=1.0,
            #     color="blue",
            #     alpha=0.4,
            #     label="CMD step dist = ||rel_xyz|| (per-step)",
            # )
            plt.plot(
                step_ids,
                real_step_dists,
                linewidth=1.0,
                color="green",
                alpha=0.4,
                label="REAL step dist = ||tcp(t)-tcp(t-1)|| (per-step)",
            )

            # ✅ 10-step sum
            # if len(cmd_period_sums) > 0:
            #     plt.plot(
            #         cmd_period_x,
            #         cmd_period_sums,
            #         marker="o",
            #         linewidth=1.5,
            #         color="orange",
            #         label=f"CMD sum / {dist_measure_period} steps",
            #     )
            if len(real_period_sums) > 0:
                plt.plot(
                    real_period_x,
                    real_period_sums,
                    marker="o",
                    linewidth=1.5,
                    color="red",
                    label=f"REAL sum / {dist_measure_period} steps",
                )

            plt.title(title)
            plt.xlabel("step (within range)")
            plt.ylabel("distance sum (m) over window")
            plt.grid(True, alpha=0.3)
            plt.legend()

            fig_path = save_root / f"range_{ridx:04d}_{start:07d}_{end:07d}.png"
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"[saved] distance plot -> {fig_path}")

    print("elapsed:", time.time() - t1)

def replay_eval():
    from calvin_env.envs.play_table_env import get_env
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
    from omegaconf import OmegaConf
    import json
    import glob
    env = get_env(Path(os.environ['ORIGINAL_CALVIN_ABCD_D']) / "validation", show_gui=False)
    conf_dir = Path(os.environ['CONF_DIR']) / "conf"
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    save_dir = './'
    log_root = "/home/ksshin/projects/sparc/UD-VLA_old_with_data/logs/calvin_exp_main/univla_calvin_abcd_video_i2ia_dis/eval_20260214_1616/log"

    with open("/home/ksshin/projects/sparc/UD-VLA/reference/RoboVLMs/configs/data/calvin/eval_sequences.json", "r") as f:
        eval_sequences = json.load(f)
    eval_sequences = eval_sequences[:100]

    # 각 eval_sequence의 5개 subtask 묶음만 따로 보관
    subtask_list = [item[1] for item in eval_sequences]  # length: 100, each length: 5

    # 2) 로그 폴더 순회하면서 action_pred들을 하나의 리스트로 모으기
    #    필요하면 sequence별로도 같이 보관
    all_actions = []          # 모든 sequence의 모든 7-dim action을 한 리스트에 flat하게 저장
    actions_per_sequence = [] # sequence별 action 리스트
    success_counts = []       # 각 sequence의 y(success_count)

    for x in range(100):
        # x_y 형식 폴더 찾기 (예: 0_5, 1_3 ...)
        candidates = glob.glob(os.path.join(log_root, f"{x}_*"))
        candidates = [p for p in candidates if os.path.isdir(p)]

        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one folder for sequence {x}, but got: {candidates}")

        xy_dir = candidates[0]
        folder_name = os.path.basename(xy_dir)
        _, y = folder_name.split("_")
        success_counts.append(int(y))

        action_pred_dir = os.path.join(xy_dir, f"{x}_action_pred")
        npy_files = sorted(glob.glob(os.path.join(action_pred_dir, "action_pred_*.npy")))

        seq_actions = []
        for npy_file in npy_files:
            arr = np.load(npy_file)   # expected shape: (10, 7)
            seq_actions.extend(arr.tolist())   # 10개의 7-dim action 추가

        actions_per_sequence.append(seq_actions)
        all_actions.extend(seq_actions)

    # 예시:
    # subtask_list[x]         -> x번째 eval_sequence의 5개 subtask
    # actions_per_sequence[x] -> x번째 eval_sequence의 모든 7-dim action 리스트
    # success_counts[x]       -> x번째 eval_sequence의 success_count(y)
    # all_actions             -> 전체 sequence의 action을 flat하게 합친 리스트

    for initial_state, eval_sequence in eval_sequences:
        step_ids = []

        # step별 거리 저장
        cmd_step_dists = []
        real_step_dists = []

        # dist_measure_period마다 계산된 "sum" 저장
        cmd_period_sums = []
        cmd_period_x = []
        real_period_sums = []
        real_period_x = []

        local_step = 0
        prev_tcp_pos = None

        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        prev_info = None
        prev_tcp_pos = None

        for seq_i, actions in enumerate(actions_per_sequence):
            local_step = 0
            for action in actions:
                cmd_xyz = np.asarray(action[:3], dtype=np.float32)
                cmd_dist = float(np.linalg.norm(cmd_xyz))
                cmd_step_dists.append(cmd_dist)

                o, _, _, info = env.step(action)

                curr_tcp_pos = np.asarray(o["robot_obs"][:3], dtype=np.float32)
                if prev_tcp_pos is None:
                    real_dist = 0.0
                else:
                    real_dist = float(np.linalg.norm(curr_tcp_pos - prev_tcp_pos))
                prev_tcp_pos = curr_tcp_pos
                real_step_dists.append(real_dist)

                step_ids.append(local_step)

                if dist_measure_period is not None and (local_step + 1) % dist_measure_period == 0:
                    cmd_window = cmd_step_dists[-dist_measure_period:]
                    real_window = real_step_dists[-dist_measure_period:]

                    cmd_sum = float(np.sum(cmd_window))
                    real_sum = float(np.sum(real_window))

                    cmd_period_sums.append(cmd_sum)
                    cmd_period_x.append(local_step)

                    real_period_sums.append(real_sum)
                    real_period_x.append(local_step)

                    print(
                        f"step {local_step - dist_measure_period + 1:04d} ~ {local_step:04d} | "
                        f"CMD sum={cmd_sum:.6f} | "
                        f"REAL sum={real_sum:.6f}"
                    )


                img = o["rgb_obs"]["rgb_static"]
                cv2.imwrite(str(save_dir + f"frame_{local_step:07d}.png"), img[:, :, ::-1])
                
                local_step += 1
                time.sleep(0.01)

            task_names = subtask_list

            title_parts = [task_names]

            plt.figure(figsize=(12, 4))
            plt.plot(
                step_ids,
                real_step_dists,
                linewidth=1.0,
                color="green",
                alpha=0.4,
                label="REAL step dist = ||tcp(t)-tcp(t-1)|| (per-step)",
            )
            if len(real_period_sums) > 0:
                plt.plot(
                    real_period_x,
                    real_period_sums,
                    marker="o",
                    linewidth=1.5,
                    color="red",
                    label=f"REAL sum / {dist_measure_period} steps",
                )

            plt.title(title_parts)
            plt.xlabel("step (within range)")
            plt.ylabel("distance sum (m) over window")
            plt.grid(True, alpha=0.3)
            plt.legend()

            fig_path = save_dir + f"{seq_i:0000d}.png"
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"[saved] distance plot -> {fig_path}")

            breakpoint()

if __name__ == "__main__":
    start_time = time.time()
    if action_from == 'dataset':
        run_env()
    elif action_from == 'eval':
        replay_eval()
    else:
        print('wrong action_from')
    end_time = time.time()
    print("실행 시간:", end_time - start_time, "초")