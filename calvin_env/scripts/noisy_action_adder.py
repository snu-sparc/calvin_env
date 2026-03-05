"""
modified from reset_env_rendered_episode.py & noisy_action_modifier.py
"""

from copy import deepcopy
import glob
import os
from pathlib import Path
import time

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

from calvin_env.envs.tasks import Tasks
from calvin_env.utils import utils

mode = 'rel'  # 'abs', 'rel'
reset_period = 1

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

TASK_ABCD_D_ROOT_DIR = '/data1/sparc/calvin/dataset/task_ABCD_D/training'
DEBUG_DATASET_ROOT_DIR = '/data3/ksshin/datasets/CALVIN/calvin_debug_dataset/training'

@hydra.main(config_path="../../conf", config_name="config_data_collection")
def run_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

    root_dir = Path(DEBUG_DATASET_ROOT_DIR)

    # 1) lang annotation index ranges 로드
    ann_path = root_dir / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_path, allow_pickle=True).item()
    indx_ranges = ann["info"]["indx"]  # [(start,end), ...]

    # (추가) ridx별 task 이름 (없을 수도 있으니 안전 처리)
    task_names = None
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
        # (추가) 이동거리 로그 초기화
        # =========================
        step_dists = []      # 매 step 이동거리 (명령 rel_action의 xyz norm)
        step_ids = []        # x축(구간 내 step index)
        avg10_vals = []      # 10 step 윈도우 평균값들
        avg10_x = []         # 그 평균값을 찍을 x 위치 (윈도우 마지막 step index 등)

        local_step = 0

        for i in range(start, end + 1):
            file = root_dir / f"episode_{i:07d}.npz"
            if not file.exists():
                print(f"[skip] missing: {file}")
                continue

            data = np.load(file)

            if (i - start) % reset_period == 0:
                print(f"reset {i}")
                env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                prev_info = None

            if mode == 'rel':
                action = data["rel_actions"]  # shape (7,)

                pos = action[:3]
                orn = p.getQuaternionFromEuler(action[3:6])
                gripper = action[6]

                pos, orn, gripper = noise(
                    (pos, orn, gripper),
                    pos_std=10,  # 기존 유지
                    rot_std=10
                )

                euler = p.getEulerFromQuaternion(orn)
                action_noisy = np.concatenate([pos, euler, [gripper]])

                action_noisy = data["rel_actions"]  # shape (7,)

                # =========================================
                # (추가) "명령한 rel_action" 기반 이동거리 계산
                # - end effector 실제 이동이 아니라 rel xyz delta norm
                # =========================================
                cmd_xyz = action_noisy[:3]  # env에 넣은 rel pos delta
                step_dist = float(np.linalg.norm(cmd_xyz))
                step_dists.append(step_dist)
                step_ids.append(local_step)

                # (추가) 10 step마다 평균 계산(비중첩 윈도우)
                if (local_step + 1) % 10 == 0:
                    window_mean = float(np.mean(step_dists[-10:]))
                    avg10_vals.append(window_mean)
                    avg10_x.append(local_step)  # 윈도우 마지막 스텝 위치에 표시

                o, _, _, info = env.step(action_noisy)

                local_step += 1

            elif mode == 'abs':
                # 요구사항이 rel_action 기반이므로 abs 모드에서는 이동거리 계산을 하지 않음
                action7 = data["actions"].astype(np.float32)  # (7,)

                pos = action7[:3].astype(np.float32)
                euler = action7[3:6].astype(np.float32)
                gripper = np.array([action7[6]], dtype=np.float32)

                action_abs = (pos, euler, gripper)
                o, _, _, info = env.step(action_abs)

            else:
                print('Wrong mode')
                import sys
                sys.exit()

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
        # (추가) ridx 구간 끝나면 그래프 저장
        # =========================
        if mode == "rel" and len(step_dists) > 0:
            overall_mean = float(np.mean(step_dists))

            # task title
            if task_names is not None and ridx < len(task_names):
                task_title = str(task_names[ridx])
            else:
                task_title = f"range_{ridx:04d}"

            title = f"{task_title} | mean(step_dist)={overall_mean:.6f}"

            plt.figure(figsize=(12, 4))
            plt.plot(step_ids, step_dists, linewidth=1.2, label="step distance (||rel_xyz||)")

            # 10 step 평균 표시 (점/선)
            if len(avg10_vals) > 0:
                plt.plot(avg10_x, avg10_vals, marker="o", linewidth=1.2, label="mean distance / 10 steps")

            plt.title(title)
            plt.xlabel("step (within range)")
            plt.ylabel("commanded distance (norm of rel xyz)")
            plt.grid(True, alpha=0.3)
            plt.legend()

            fig_path = save_root / f"range_{ridx:04d}_{start:07d}_{end:07d}_cmd_dist.png"
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"[saved] distance plot -> {fig_path}")

    print("elapsed:", time.time() - t1)


if __name__ == "__main__":
    import time
    start_time = time.time()
    run_env()
    end_time = time.time()
    print("실행 시간:", end_time - start_time, "초")