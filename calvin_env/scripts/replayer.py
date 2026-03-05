"""
modified from reset_env_rendered_episode.py & noisy_action_modifier.py
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

mode = "rel"         # 'abs', 'rel'
reset_period = 1 # None, int
add_noise = True # yet only rel_action modes.

# ✅ 거리 계산 모드: "cmd" | "real" | "both"
dist_mode = "both"

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

TASK_ABCD_D_ROOT_DIR = "/data1/sparc/calvin/dataset/task_ABCD_D/training"
DEBUG_DATASET_ROOT_DIR = "/data3/ksshin/datasets/CALVIN/calvin_debug_dataset/training"

@hydra.main(config_path="../../conf", config_name="config_data_collection")
def run_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

    root_dir = Path(DEBUG_DATASET_ROOT_DIR)

    ann_path = root_dir / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_path, allow_pickle=True).item()
    indx_ranges = ann["info"]["indx"]

    # ridx별 task 이름
    try:
        task_names = ann["language"]["task"]
    except Exception:
        task_names = None

    tasks = hydra.utils.instantiate(cfg.tasks)
    prev_info = None
    t1 = time.time()

    save_root = root_dir / "replay_frames_lang_ranges"
    save_root.mkdir(parents=True, exist_ok=True)

    for ridx, (start, end) in enumerate(indx_ranges):
        print(f"\n=== range {ridx}: {start} ~ {end} ===")
        save_dir = save_root

        # =========================
        # 거리 로그 (cmd / real)
        # =========================
        step_ids = []
        cmd_step_dists = []
        real_step_dists = []

        cmd_avg10_vals, cmd_avg10_x = [], []
        real_avg10_vals, real_avg10_x = [], []

        local_step = 0
        prev_tcp_pos = None  # real distance 계산용

        for i in range(start, end + 1):
            file = root_dir / f"episode_{i:07d}.npz"
            if not file.exists():
                print(f"[skip] missing: {file}")
                continue

            data = np.load(file)

            if reset_period is not None:
                if (i - start) % reset_period == 0:
                    print(f"reset {i}")
                    env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                    prev_info = None
                    prev_tcp_pos = None  # ✅ reset 경계에서 real 거리 튐 방지
            else:
                if i == start:
                    env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                    prev_info = None
                    prev_tcp_pos = None  # ✅ reset 경계에서 real 거리 튐 방지

            if mode == "rel":
                action = data["rel_actions"]  # (7,)
                if add_noise:
                    pos = action[:3]
                    orn = p.getQuaternionFromEuler(action[3:6])
                    gripper = action[6]

                    pos, orn, gripper = noise(
                        (pos, orn, gripper),
                        pos_std=10,
                        rot_std=10,
                    )

                    euler = p.getEulerFromQuaternion(orn)
                    action_noisy = np.concatenate([pos, euler, [gripper]])
                else:
                    action_noisy = action

                # ===== env step =====
                o, _, _, info = env.step(action_noisy)

                # =========================
                # ✅ 1) CMD 기반 거리 (명령 rel xyz norm)
                # =========================
                if dist_mode in ("cmd", "both"):
                    cmd_xyz = action_noisy[:3]  # env에 넣은 rel pos delta
                    cmd_dist = float(np.linalg.norm(cmd_xyz))
                    cmd_step_dists.append(cmd_dist)

                # =========================
                # ✅ 2) REAL 기반 거리 (실제 TCP 이동: robot_obs의 tcp_pos 사용)
                # robot_obs (15,):
                #   tcp_pos(3), tcp_euler(3), gripper_width(1), arm_joints(7), gripper_action(1)
                # =========================
                if dist_mode in ("real", "both"):
                    curr_tcp_pos = np.asarray(o["robot_obs"][:3], dtype=np.float32)

                    if prev_tcp_pos is None:
                        real_dist = 0.0  # 첫 step은 비교 대상이 없어 0 처리
                    else:
                        real_dist = float(np.linalg.norm(curr_tcp_pos - prev_tcp_pos))

                    prev_tcp_pos = curr_tcp_pos
                    real_step_dists.append(real_dist)

                # 공통 x축
                step_ids.append(local_step)

                # 10-step 평균 (비중첩 윈도우)
                if (local_step + 1) % 10 == 0:
                    if dist_mode in ("cmd", "both") and len(cmd_step_dists) >= 10:
                        m = float(np.mean(cmd_step_dists[-10:]))
                        cmd_avg10_vals.append(m)
                        cmd_avg10_x.append(local_step)

                    if dist_mode in ("real", "both") and len(real_step_dists) >= 10:
                        m = float(np.mean(real_step_dists[-10:]))
                        real_avg10_vals.append(m)
                        real_avg10_x.append(local_step)

                local_step += 1

            elif mode == "abs":
                action7 = data["actions"].astype(np.float32)

                pos = action7[:3].astype(np.float32)
                euler = action7[3:6].astype(np.float32)
                gripper = np.array([action7[6]], dtype=np.float32)

                action_abs = (pos, euler, gripper)
                o, _, _, info = env.step(action_abs)

            else:
                raise ValueError("Wrong mode")

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
        # ✅ ridx 구간 그래프 저장
        # =========================
        if mode == "rel" and len(step_ids) > 0:
            if task_names is not None and ridx < len(task_names):
                task_title = str(task_names[ridx])
            else:
                task_title = f"range_{ridx:04d}"

            # 제목에 overall mean 포함
            title_parts = [task_title]
            if dist_mode in ("cmd", "both") and len(cmd_step_dists) > 0:
                title_parts.append(f"cmd_mean={float(np.mean(cmd_step_dists)):.6f}")
            if dist_mode in ("real", "both") and len(real_step_dists) > 0:
                title_parts.append(f"real_mean={float(np.mean(real_step_dists)):.6f}")
            title = " | ".join(title_parts)

            plt.figure(figsize=(12, 4))

            if dist_mode in ("cmd", "both"):
                plt.plot(step_ids, cmd_step_dists, linewidth=1.2, label="CMD step dist = ||rel_xyz||")
                if len(cmd_avg10_vals) > 0:
                    plt.plot(cmd_avg10_x, cmd_avg10_vals, marker="o", linewidth=1.2, label="CMD mean / 10 steps")

            if dist_mode in ("real", "both"):
                plt.plot(step_ids, real_step_dists, linewidth=1.2, label="REAL step dist = ||tcp(t)-tcp(t-1)||")
                if len(real_avg10_vals) > 0:
                    plt.plot(real_avg10_x, real_avg10_vals, marker="o", linewidth=1.2, label="REAL mean / 10 steps")

            plt.title(title)
            plt.xlabel("step (within range)")
            plt.ylabel("distance (m)")
            plt.grid(True, alpha=0.3)
            plt.legend()

            fig_path = save_root / f"range_{ridx:04d}_{start:07d}_{end:07d}_dist_{dist_mode}.png"
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