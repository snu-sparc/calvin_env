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

from calvin_env.envs.tasks import Tasks

replay_action_type = "rel"  # 'abs', 'rel'
env_reset_period = None  # None, int
add_noise = False  # True, False. yet only rel_action modes.
noise_period = 10  # None, int
dist_measure_period = 10  # None, int
target_dataset_root_dir = "/data3/ksshin/datasets/CALVIN/calvin_debug_dataset/training"


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


if __name__ == "__main__":
    start_time = time.time()
    run_env()
    end_time = time.time()
    print("실행 시간:", end_time - start_time, "초")