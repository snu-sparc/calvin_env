"""
modified from reset_env_rendered_episode.py & noisy_action_modifier.py

기존 기능:
- 10 step마다 command displacement / real displacement의 누적 합(sum)을 계산해서 출력 및 그래프 표시

추가 기능:
- 10 step마다 첫 step과 마지막 step 사이의 displacement도 계산해서 출력 및 그래프 표시
  예) 0~9, 10~19, 20~29 ...
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

import shutil
import gc
from hydra import compose, initialize

from calvin_env.envs.tasks import Tasks

np.float = float

replay_action_type = "rel"  # 'abs', 'rel'
env_reset_period = 1  # None, int
add_action_noise = True  # True, False. yet only rel_action modes.
action_noise_period = 1  # None, int
dist_measure_period = 10  # None, int

split = 'training'

target_dataset_root_dir = f"{os.environ['ORIGINAL_CALVIN_ABCD_D_DIR']}/{split}"

action_from = 'dataset'  # 'dataset', 'eval'
NUM_SEQUENCES = 10

### only for run_env
modify_clean_image = True
add_ood_env = True
processing_limit = 1000
save_log = True
noise_scale = 2.5
num_colors = 20
random_config_selection = True  # True: randomly select config, False: sequential (ridx % num_colors)
random_config_seed = 42
processed_output_save_dir = f"{os.environ['ORIGINAL_CALVIN_ABCD_D_NOISE_DIR']}/{split}"

# =========================
# [ADDED] gaussian blur augmentation
# =========================
add_random_gaussian_blur = True

# reproducible seed
blur_random_seed = 1234

# per image blur patch 개수
blur_num_patches_min = 3
blur_num_patches_max = 10

# gaussian blur kernel 후보 (홀수 권장)
blur_kernel_candidates = [9, 15, 21, 31]

# patch 크기 비율 범위 (이미지 width/height 대비)
blur_size_ratio_min = 0.05
blur_size_ratio_max = 0.5

# patch 모양 후보
blur_shape_candidates = ["triangle", "rectangle", "circle", "ellipse"]

# 각 카메라별 적용 여부
apply_blur_to_static = True
apply_blur_to_gripper = True

# =========================
# [ADDED] reproducible random gaussian blur patches
# =========================
def apply_random_gaussian_blur_patches(
    image,
    rng,
    num_patches_range=(1, 5),
    kernel_candidates=(9, 15, 21, 31),
    size_ratio_range=(0.05, 0.25),
    shape_candidates=("circle", "ellipse", "rectangle", "triangle"),
):
    """
    image: H x W x C
    rng: np.random.Generator
    랜덤 위치/모양/크기/개수의 patch 영역에만 Gaussian blur 적용
    rectangle / ellipse / triangle은 rotation 가능
    """
    if image is None:
        return image

    out = image.copy()
    h, w = out.shape[:2]

    # blur kernel 선택
    k = int(rng.choice(kernel_candidates))
    if k % 2 == 0:
        k += 1

    blurred = cv2.GaussianBlur(out, (k, k), sigmaX=0)

    # patch 개수 선택
    n_patches = int(rng.integers(num_patches_range[0], num_patches_range[1] + 1))

    mask = np.zeros((h, w), dtype=np.uint8)

    for _ in range(n_patches):
        shape = rng.choice(shape_candidates)

        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))

        rw = max(1, int(rng.uniform(*size_ratio_range) * w))
        rh = max(1, int(rng.uniform(*size_ratio_range) * h))

        angle = float(rng.uniform(0, 360))

        if shape == "circle":
            radius = max(1, min(rw, rh) // 2)
            cv2.circle(mask, (cx, cy), radius, 255, thickness=-1)

        elif shape == "ellipse":
            axes = (max(1, rw // 2), max(1, rh // 2))
            cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, thickness=-1)

        elif shape == "rectangle":
            rot_rect = ((float(cx), float(cy)), (float(rw), float(rh)), angle)
            box = cv2.boxPoints(rot_rect)   # (4, 2)
            box = np.int32(box)
            cv2.fillConvexPoly(mask, box, 255)

        elif shape == "triangle":
            # 중심 기준의 기본 삼각형을 만든 뒤 회전 + 평행이동
            half_w = rw / 2.0
            half_h = rh / 2.0

            # 위쪽 꼭짓점 1개, 아래쪽 꼭짓점 2개
            pts = np.array([
                [0.0, -half_h],
                [-half_w, half_h],
                [half_w, half_h],
            ], dtype=np.float32)

            theta = np.deg2rad(angle)
            rot = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta),  np.cos(theta)],
            ], dtype=np.float32)

            pts = pts @ rot.T
            pts[:, 0] += cx
            pts[:, 1] += cy

            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

            pts = np.int32(pts)
            cv2.fillConvexPoly(mask, pts, 255)

    out[mask > 0] = blurred[mask > 0]
    return out

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

# if add_ood_env:
#     config_name="config_data_collection_ood_env"
# else:
#     config_name="config_data_collection"
# @hydra.main(config_path="../../conf", config_name=config_name)
# def run_env(cfg):
def run_env():
    #env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

    if add_ood_env:
        config_name_list = [f"config_data_collection_{i}" for i in range(num_colors)]
    else:
        config_name_list = ["config_data_collection"]
    conf_dir = Path(__file__).resolve().parent / "../../conf"
    conf_dir = conf_dir.resolve()

    if random_config_selection:
        config_rng = np.random.default_rng(random_config_seed)

    with initialize(config_path="../../conf"):
        cfg = compose(config_name=config_name_list[0])

        root_dir = Path(target_dataset_root_dir)

        ann_path = root_dir / "lang_annotations" / "auto_lang_ann.npy"
        ann = np.load(ann_path, allow_pickle=True).item()
        indx_ranges = ann["info"]["indx"]
        task_names = ann["language"]["task"]

        tasks = hydra.utils.instantiate(cfg.tasks)
        prev_info = None
        t1 = time.time()

    if save_log:
        save_root = Path(processed_output_save_dir) / (
            f"replay_frames_lang_ranges_"
            f"[replay_action_type({replay_action_type})_"
            f"env_reset_period({env_reset_period})_"
            f"add_action_noise({add_action_noise})_"
            f"action_noise_period({action_noise_period})_"
            f"dist_measure_period({dist_measure_period})_SUM_AND_ENDPOINT]"
            f"add_random_gaussian_blur({add_random_gaussian_blur})]"
        )
        save_root.mkdir(parents=True, exist_ok=True)

    for ridx, (start, end) in enumerate(indx_ranges):

        if ridx >= processing_limit:
            break

        if random_config_selection:
            cfg_idx = int(config_rng.integers(0, len(config_name_list)))
            if add_action_noise and modify_clean_image:
                cfg_idx_no_noise = int(config_rng.integers(0, len(config_name_list)))
        else:
            cfg_idx = ridx % len(config_name_list)
            if add_action_noise and modify_clean_image:
                cfg_idx_no_noise = (ridx + 1) % len(config_name_list)

        with initialize(config_path="../../conf"):
            print(f"\n=== range {ridx}: {start} ~ {end} ===")
            selected_config_name = config_name_list[cfg_idx]
            selected_cfg = compose(config_name=selected_config_name)
            print(f"[config] using {selected_config_name}")

            if add_action_noise and modify_clean_image:
                selected_config_name_no_noise = config_name_list[cfg_idx_no_noise]
                selected_cfg_no_noise = compose(config_name=selected_config_name_no_noise)
                print(f"[config_no_noise] using {selected_config_name_no_noise}")
        
        env = hydra.utils.instantiate(selected_cfg.env, show_gui=False, use_vr=False, use_scene_info=True)
        if add_action_noise and modify_clean_image:
            env_no_noise = hydra.utils.instantiate(selected_cfg_no_noise.env, show_gui=False, use_vr=False, use_scene_info=True)

        prev_info = None
        prev_tcp_pos = None
        
        save_dir = save_root

        step_ids = []

        # step별 거리 저장
        cmd_step_dists = []
        real_step_dists = []

        # step별 xyz 저장 (endpoint displacement 계산용)
        cmd_step_xyzs = []
        real_step_xyzs = []

        # dist_measure_period마다 계산된 "sum" 저장
        cmd_period_sums = []
        cmd_period_x = []
        real_period_sums = []
        real_period_x = []

        # dist_measure_period마다 계산된 "첫 step ~ 마지막 step displacement" 저장
        cmd_period_endpoint_dists = []
        cmd_period_endpoint_x = []
        real_period_endpoint_dists = []
        real_period_endpoint_x = []

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
                    if add_action_noise and modify_clean_image:
                        env_no_noise.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                    prev_info = None
                    prev_tcp_pos = None
            else:
                if i == start:
                    print(f"reset {i}")
                    env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                    if add_action_noise and modify_clean_image:
                        env_no_noise.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
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

            # save original action before noise for env_no_noise
            action_original = action if not isinstance(action, np.ndarray) else action.copy()

            if replay_action_type == "rel" and add_action_noise and action_noise_period is not None and i % action_noise_period == 0:
                print('#####################noise added#####################')
                print(action)
                pos = action[:3]
                orn = p.getQuaternionFromEuler(action[3:6])
                gripper = action[6]

                pos, orn, gripper = noise(
                    (pos, orn, gripper),
                    pos_std=noise_scale,
                    rot_std=noise_scale,
                )

                euler = p.getEulerFromQuaternion(orn)
                action = np.concatenate([pos, euler, [gripper]])
                print(action)

            # -------------------------
            # command displacement (step dist)
            # -------------------------
            cmd_xyz = np.asarray(action[:3], dtype=np.float32)
            cmd_dist = float(np.linalg.norm(cmd_xyz))
            cmd_step_dists.append(cmd_dist)
            cmd_step_xyzs.append(cmd_xyz.copy())

            # env step
            o, _, _, info = env.step(action)

            # env_no_noise step (original action without noise)
            if add_action_noise and modify_clean_image:
                o_no_noise, _, _, _ = env_no_noise.step(action_original)

            # -------------------------
            # real displacement (step dist)
            # -------------------------
            curr_tcp_pos = np.asarray(o["robot_obs"][:3], dtype=np.float32)
            real_step_xyzs.append(curr_tcp_pos.copy())

            if prev_tcp_pos is None:
                real_dist = 0.0
            else:
                real_dist = float(np.linalg.norm(curr_tcp_pos - prev_tcp_pos))
            prev_tcp_pos = curr_tcp_pos
            real_step_dists.append(real_dist)

            step_ids.append(local_step)

            # -------------------------
            # dist_measure_period마다 "sum" + "endpoint displacement" 계산/출력
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

                start_idx = local_step - dist_measure_period + 1
                end_idx = local_step

                cmd_endpoint_dist = float(
                    np.linalg.norm(cmd_step_xyzs[end_idx] - cmd_step_xyzs[start_idx])
                )
                real_endpoint_dist = float(
                    np.linalg.norm(real_step_xyzs[end_idx] - real_step_xyzs[start_idx])
                )

                cmd_period_endpoint_dists.append(cmd_endpoint_dist)
                cmd_period_endpoint_x.append(local_step)

                real_period_endpoint_dists.append(real_endpoint_dist)
                real_period_endpoint_x.append(local_step)

                print(
                    f"[range {ridx:04d}] "
                    f"step {start_idx:04d} ~ {end_idx:04d} | "
                    f"CMD sum={cmd_sum:.6f} | "
                    f"REAL sum={real_sum:.6f} | "
                    f"CMD endpoint={cmd_endpoint_dist:.6f} | "
                    f"REAL endpoint={real_endpoint_dist:.6f}"
                )

            local_step += 1

            print(info["scene_info"]["lights"]["led"]["logical_state"])
            if prev_info is not None:
                print(tasks.get_task_info(prev_info, info))
            prev_info = deepcopy(info)

            img = o["rgb_obs"]["rgb_static"]
            gripper_img = o["rgb_obs"]["rgb_gripper"]

            # =========================
            # [ADDED] deterministic blur per frame/camera
            # 같은 i 에 대해서 항상 같은 blur가 나오도록 seed 고정
            # =========================
            if add_random_gaussian_blur:
                # frame index i 기준으로 카메라별 seed 분리
                static_rng = np.random.default_rng(blur_random_seed + i * 2 + 0)
                gripper_rng = np.random.default_rng(blur_random_seed + i * 2 + 1)

                if apply_blur_to_static:
                    img = apply_random_gaussian_blur_patches(
                        img,
                        rng=static_rng,
                        num_patches_range=(blur_num_patches_min, blur_num_patches_max),
                        kernel_candidates=blur_kernel_candidates,
                        size_ratio_range=(blur_size_ratio_min, blur_size_ratio_max),
                        shape_candidates=blur_shape_candidates,
                    )

                if apply_blur_to_gripper:
                    gripper_img = apply_random_gaussian_blur_patches(
                        gripper_img,
                        rng=gripper_rng,
                        num_patches_range=(blur_num_patches_min, blur_num_patches_max),
                        kernel_candidates=blur_kernel_candidates,
                        size_ratio_range=(blur_size_ratio_min, blur_size_ratio_max),
                        shape_candidates=blur_shape_candidates,
                    )

                # 저장용 obs에 반영
                o["rgb_obs"]["rgb_static"] = img
                o["rgb_obs"]["rgb_gripper"] = gripper_img

            if save_log:
                cv2.imwrite(str(save_dir / f"frame_{i:07d}.png"), img[:, :, ::-1])
                cv2.imwrite(str(save_dir / f"gripper_frame_{i:07d}.png"), gripper_img[:, :, ::-1])

            out = {
                "rgb_static": data["rgb_static"],
                "rgb_gripper": data["rgb_gripper"],
                "actions": data["actions"],
                "rel_actions": data["rel_actions"],
                "robot_obs": data["robot_obs"],
                "scene_obs": data["scene_obs"],
                "rgb_static_noisy": o["rgb_obs"]["rgb_static"],
                "rgb_gripper_noisy": o["rgb_obs"]["rgb_gripper"],
                'robot_obs_xyz': o["robot_obs"][:3]
            }
            if add_action_noise and modify_clean_image:
                out["rgb_static"] = o_no_noise["rgb_obs"]["rgb_static"]
                out["rgb_gripper"] = o_no_noise["rgb_obs"]["rgb_gripper"]

            if save_log:
                cv2.imwrite(str(save_dir / f"clean_frame_{i:07d}.png"), o_no_noise["rgb_obs"]["rgb_static"][:, :, ::-1])
                cv2.imwrite(str(save_dir / f"clean_gripper_frame_{i:07d}.png"), o_no_noise["rgb_obs"]["rgb_gripper"][:, :, ::-1])

            out_file = Path(processed_output_save_dir) / f"episode_{i:07d}_noisy_action_image_added.npz"
            np.savez_compressed(out_file, **out)
        
        try:
            env.close()
        except Exception as e:
            print(f"Warning closing env: {e}")
        finally:
            if hasattr(env, "ownsPhysicsClient"):
                env.ownsPhysicsClient = False
            if hasattr(env, "cid"):
                env.cid = -1
            env = None

        if add_action_noise and modify_clean_image:
            try:
                env_no_noise.close()
            except Exception as e:
                print(f"Warning closing env_no_noise: {e}")
            finally:
                if hasattr(env_no_noise, "ownsPhysicsClient"):
                    env_no_noise.ownsPhysicsClient = False
                if hasattr(env_no_noise, "cid"):
                    env_no_noise.cid = -1
                env_no_noise = None

        gc.collect()
        cv2.destroyAllWindows()

        # =========================
        # 구간 그래프 저장
        # =========================
        if save_log:
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

                # step dist(참고용으로 유지)
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

                # 기존 10-step sum 유지
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

                # 추가: 10-step endpoint displacement
                # if len(cmd_period_endpoint_dists) > 0:
                #     plt.plot(
                #         cmd_period_endpoint_x,
                #         cmd_period_endpoint_dists,
                #         marker="s",
                #         linewidth=1.5,
                #         color="brown",
                #         label=f"CMD endpoint dist / {dist_measure_period} steps",
                #     )
                if len(real_period_endpoint_dists) > 0:
                    plt.plot(
                        real_period_endpoint_x,
                        real_period_endpoint_dists,
                        marker="s",
                        linewidth=1.5,
                        color="purple",
                        label=f"REAL endpoint dist / {dist_measure_period} steps",
                    )

                plt.title(title)
                plt.xlabel("step (within range)")
                plt.ylabel("distance (m)")
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

    env = get_env(Path(os.environ['ORIGINAL_CALVIN_ABCD_D_DIR']) / "validation", show_gui=False)
    conf_dir = Path(os.environ['CONF_DIR']) / "conf"
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)

    save_dir = "./replay_frames_eval/"
    os.makedirs(save_dir, exist_ok=True)
    #log_root = "/home/ksshin/projects/sparc/UD-VLA_old_with_data/logs/calvin_exp_main/univla_calvin_abcd_video_i2ia_dis/eval_20260214_1616/log" #baseline
    #log_root = "/home/ksshin/projects/sparc/UD-VLA_old_with_data/logs/calvin_exp_main/univla_calvin_abcd_video_i2ia_dis/eval_20260217_0035/log" #obj 1 (blue->cyan)
    log_root = "/home/ksshin/projects/sparc/UD-VLA_old_with_data/logs/calvin_exp_main/univla_calvin_abcd_video_i2ia_dis/eval_20260215_2128/log" #env

    with open(
        "/home/ksshin/projects/sparc/UD-VLA/reference/RoboVLMs/configs/data/calvin/eval_sequences.json",
        "r",
    ) as f:
        eval_sequences = json.load(f)
    eval_sequences = eval_sequences[:NUM_SEQUENCES]

    # 각 eval_sequence의 5개 subtask 묶음
    subtask_list = [item[1] for item in eval_sequences]

    # sequence별 action prediction 로드
    actions_per_sequence = []
    success_counts = []

    for x in range(NUM_SEQUENCES):
        candidates = glob.glob(os.path.join(log_root, f"{x}_*"))
        candidates = [p for p in candidates if os.path.isdir(p)]

        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one folder for sequence {x}, but got: {candidates}"
            )

        xy_dir = candidates[0]
        folder_name = os.path.basename(xy_dir)
        _, y = folder_name.split("_")
        success_counts.append(int(y))

        action_pred_dir = os.path.join(xy_dir, f"{x}_action_pred")
        npy_files = sorted(glob.glob(os.path.join(action_pred_dir, "action_pred_*.npy")))

        seq_actions = []
        for npy_file in npy_files:
            arr = np.load(npy_file)  # expected shape: (10, 7)
            seq_actions.extend(arr.tolist())

        actions_per_sequence.append(seq_actions)

    # 길이 sanity check
    if len(actions_per_sequence) != len(eval_sequences):
        raise ValueError(
            f"Length mismatch: len(actions_per_sequence)={len(actions_per_sequence)}, "
            f"len(eval_sequences)={len(eval_sequences)}"
        )

    # -------------------------
    # 1:1 매칭 replay
    # eval_sequences[i] <-> actions_per_sequence[i]
    # -------------------------
    for eval_idx, (initial_state, eval_sequence) in enumerate(eval_sequences):
        actions = actions_per_sequence[eval_idx]
        task_names = subtask_list[eval_idx]
        success_count = success_counts[eval_idx]

        # sequence별 독립 측정 버퍼
        step_ids = []

        # step별 거리 저장
        cmd_step_dists = []
        real_step_dists = []

        # step별 xyz 저장 (endpoint displacement 계산용)
        cmd_step_xyzs = []
        real_step_xyzs = []

        # dist_measure_period마다 계산된 "sum" 저장
        cmd_period_sums = []
        cmd_period_x = []
        real_period_sums = []
        real_period_x = []

        # dist_measure_period마다 계산된 "첫 step ~ 마지막 step displacement" 저장
        cmd_period_endpoint_dists = []
        cmd_period_endpoint_x = []
        real_period_endpoint_dists = []
        real_period_endpoint_x = []

        local_step = 0
        prev_tcp_pos = None
        prev_info = None

        # 해당 eval_idx의 initial_state로 reset
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        print(f"\n=== eval sequence {eval_idx:03d} ===")
        print(f"subtasks: {task_names}")
        print(f"success_count: {success_count}")
        print(f"num_actions: {len(actions)}")

        for action in actions:
            action = np.asarray(action, dtype=np.float32)

            # -------------------------
            # command displacement (per-step)
            # -------------------------
            cmd_xyz = np.asarray(action[:3], dtype=np.float32)
            cmd_dist = float(np.linalg.norm(cmd_xyz))
            cmd_step_dists.append(cmd_dist)
            cmd_step_xyzs.append(cmd_xyz.copy())

            # env step
            o, _, _, info = env.step(action)

            # -------------------------
            # real displacement (per-step)
            # -------------------------
            curr_tcp_pos = np.asarray(o["robot_obs"][:3], dtype=np.float32)
            real_step_xyzs.append(curr_tcp_pos.copy())

            if prev_tcp_pos is None:
                real_dist = 0.0
            else:
                real_dist = float(np.linalg.norm(curr_tcp_pos - prev_tcp_pos))

            prev_tcp_pos = curr_tcp_pos
            real_step_dists.append(real_dist)

            step_ids.append(local_step)

            # -------------------------
            # dist_measure_period마다
            # 1) sum
            # 2) endpoint displacement
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

                start_idx = local_step - dist_measure_period + 1
                end_idx = local_step

                cmd_endpoint_dist = float(
                    np.linalg.norm(cmd_step_xyzs[end_idx] - cmd_step_xyzs[start_idx])
                )
                real_endpoint_dist = float(
                    np.linalg.norm(real_step_xyzs[end_idx] - real_step_xyzs[start_idx])
                )

                cmd_period_endpoint_dists.append(cmd_endpoint_dist)
                cmd_period_endpoint_x.append(local_step)

                real_period_endpoint_dists.append(real_endpoint_dist)
                real_period_endpoint_x.append(local_step)

                print(
                    f"[eval {eval_idx:03d}] "
                    f"step {start_idx:04d} ~ {end_idx:04d} | "
                    f"CMD sum={cmd_sum:.6f} | "
                    f"REAL sum={real_sum:.6f} | "
                    f"CMD endpoint={cmd_endpoint_dist:.6f} | "
                    f"REAL endpoint={real_endpoint_dist:.6f}"
                )

            # 프레임 저장
            img = o["rgb_obs"]["rgb_static"]
            cv2.imwrite(
                os.path.join(save_dir, f"eval_{eval_idx:03d}_frame_{local_step:07d}.png"),
                img[:, :, ::-1],
            )

            prev_info = deepcopy(info)
            local_step += 1

        # -------------------------
        # sequence별 그래프 저장
        # -------------------------
        title_parts = [
            f"eval_idx={eval_idx:03d}",
            f"success={success_count}",
            f"subtasks={task_names}",
        ]
        if len(cmd_step_dists) > 0:
            title_parts.append(f"cmd_step_mean={float(np.mean(cmd_step_dists)):.6f}")
        if len(real_step_dists) > 0:
            title_parts.append(f"real_step_mean={float(np.mean(real_step_dists)):.6f}")

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

        if len(real_period_endpoint_dists) > 0:
            plt.plot(
                real_period_endpoint_x,
                real_period_endpoint_dists,
                marker="s",
                linewidth=1.5,
                color="purple",
                label=f"REAL endpoint dist / {dist_measure_period} steps",
            )

        plt.title(" \n ".join(title_parts))
        plt.xlabel("step (within sequence)")
        plt.ylabel("distance (m)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        fig_path = os.path.join(save_dir, f"distance_eval_{eval_idx:03d}.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"[saved] distance plot -> {fig_path}")

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

    src_dir = Path(target_dataset_root_dir)
    dst_dir = Path(processed_output_save_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    items_to_copy = [
        "statistics.yaml",
        "lang_annotations",
        "replay_frames_lang_ranges",
        "ep_start_end_ids.npy",
        ".hydra",
        "ep_lens.npy",
        "scene_info.npy",
        "lang_paraphrase-MiniLM-L3-v2",
    ]

    for name in items_to_copy:
        src = src_dir / name
        dst = dst_dir / name

        if not src.exists():
            print(f"Not found: {src}")
            continue

        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"Copied directory: {src} -> {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"Copied file: {src} -> {dst}")
