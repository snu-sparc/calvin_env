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

"""
This script loads a rendered episode and replays it using the recorded actions.
Optionally, gaussian noise can be added to the actions.
"""


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
DEBUG_DATASET_ROOT_DIR = '/data1/sparc/calvin/dataset/calvin_debug_dataset/training'

@hydra.main(config_path="../../conf", config_name="config_data_collection")
def run_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

    root_dir = Path(DEBUG_DATASET_ROOT_DIR)

    # 1) lang annotation index ranges 로드
    ann_path = root_dir / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_path, allow_pickle=True).item()
    indx_ranges = ann["info"]["indx"]  # [(start,end), ...]

    tasks = hydra.utils.instantiate(cfg.tasks)
    prev_info = None
    t1 = time.time()

    save_root = root_dir / "replay_frames_lang_ranges"
    save_root.mkdir(parents=True, exist_ok=True)

    for ridx, (start, end) in enumerate(indx_ranges):
        print(f"\n=== range {ridx}: {start} ~ {end} ===")
        #save_dir = save_root / f"range_{ridx:04d}_{start:07d}_{end:07d}"
        #save_dir.mkdir(parents=True, exist_ok=True)
        save_dir = save_root

        for i in range(start, end + 1):
            file = root_dir / f"episode_{i:07d}.npz"
            if not file.exists():
                print(f"[skip] missing: {file}")
                continue

            data = np.load(file)

            # (선택) 원본 rgb_static 저장하고 싶으면 이름 분리
            # img0 = data["rgb_static"]
            # cv2.imwrite(str(save_dir / f"frame_{i:07d}_pre.png"), img0[:, :, ::-1])

            # 3) reset 규칙은 기존처럼 유지 (32 step마다) -> 1 step으로 바꿈
            if (i - start) % 1 == 0:
                print(f"reset {i}")
                env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                prev_info = None  # 구간 시작마다 비교 초기화(원하면 유지해도 됨)

            origin_actions = data["actions"]
            origin_rel_actions = data["rel_actions"]
            origin_rgb_static = data["rgb_static"]
            origin_rgb_gripper = data["rgb_gripper"]
            origin_robot_obs = data["robot_obs"]
            origin_scene_obs = data["scene_obs"]

            # 4) rel_actions -> noise 적용 -> step
            action = data["rel_actions"]  # shape (7,)

            pos = action[:3]
            orn = p.getQuaternionFromEuler(action[3:6])
            gripper = action[6]

            pos, orn, gripper = noise((pos, orn, gripper),
                                      pos_std=10,   # TODO: 단위 m라면 10은 매우 큼(원래 의도면 유지)
                                      rot_std=10)

            euler = p.getEulerFromQuaternion(orn)
            action_noisy = np.concatenate([pos, euler, [gripper]])

            o, _, _, info = env.step(action_noisy)

            # 5) task info 출력(기존 유지)
            print(info["scene_info"]["lights"]["led"]["logical_state"])
            if prev_info is not None:
                print(tasks.get_task_info(prev_info, info))
            prev_info = deepcopy(info)

            # 6) step 후 관측 이미지 저장 (덮어쓰기 방지)
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

    print("elapsed:", time.time() - t1)


if __name__ == "__main__":
    import time
    start_time = time.time()
    run_env()
    end_time = time.time()
    print("실행 시간:", end_time - start_time, "초")
