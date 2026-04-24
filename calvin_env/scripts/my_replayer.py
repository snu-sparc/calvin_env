"""
my_replayer.py — Calvin 환경 replay + noisy data augmentation 파이프라인
(modified from reset_env_rendered_episode.py & noisy_action_modifier.py)

================================================================================
실행 모드  (action_from 변수로 선택)
================================================================================

  'dataset'          run_env()            데이터셋 replay → noisy/clean 이미지 쌍 생성
  'eval'             replay_eval()        평가 로그 action prediction replay
  'fix_annotations'  fix_annotations_only()  env 없이 annotation만 재생성

================================================================================
필수 환경변수
================================================================================

  # 공통 (모든 모드)
  export ORIGINAL_CALVIN_ABCD_D_DIR=/path/to/original/calvin/dataset
  export ORIGINAL_CALVIN_ABCD_D_NOISE_DIR=/path/to/output

  # replay_eval 모드 추가 필요
  export CONF_DIR=/path/to/calvin_conf

================================================================================
실행 방법
================================================================================

  python my_replayer.py

================================================================================
주요 설정 변수 (파일 상단에서 직접 수정)
================================================================================

  [공통]
  action_from            실행 모드 ('dataset' / 'eval' / 'fix_annotations')
  split                  데이터 split ('training' / 'validation')
  verbose                True: step마다 상세 디버그 출력

  [run_env 전용]
  processing_limit       처리할 시퀀스(ridx) 수 상한
  env_reset_period       N step마다 데이터셋 상태로 환경 리셋 (1=매 step, None=시퀀스 시작만)
  modify_clean_image     True: PASS 2에서 clean env를 별도 실행하여 clean image 생성
  save_log               True: 프레임 PNG + distance plot 저장
  dist_measure_period    N step마다 displacement 통계 계산 (None=비활성화)

  [OOD 환경 — 테이블 텍스처/조명 다양화]
  add_ood_env            True: scene별 다중 OOD config 사용
                           사전 준비: calvin_color_customizer.py --target {T} --num_colors N
  num_colors             add_ood_env=True일 때 scene당 config 개수
  random_config_selection  True: ridx마다 OOD config 랜덤 선택 / False: 순차 선택
  random_config_seed     random_config_selection 재현용 seed

  [Action noise]
  add_action_noise       True: rel_action에 가우시안 노이즈 추가 (PASS 1)
  action_noise_period    N step마다 노이즈 추가 (1=매 step)
  noise_scale            노이즈 크기 (pos_std(m) / rot_std(°) 공통값)
  replay_action_type     'rel' (relative 7D) / 'abs' (absolute 7D)

  [Block color diversification — 블록 색상 다양화]
  diversify_block_colors  True: 매 ridx마다 red/blue/pink 블록을 랜덤 색상으로 교체
                            사전 준비: calvin_color_customizer.py --diversify_blocks
  num_block_colors        사용 가능 색상 수 (고정값=11, URDF 인덱스 0~10)
  block_color_seed        재현용 seed
  same_block_colors_for_clean
    True  → noisy/clean env 동일 블록 색 (3색 사용)
    False → noisy/clean env 서로 다른 블록 색 (6색 사용, 겹침 없음)

  [Gaussian blur augmentation]
  add_random_gaussian_blur  True: 랜덤 위치/모양/크기의 blur 패치를 이미지에 적용
  blur_random_seed          재현용 seed (frame index와 조합)
  blur_num_patches_min/max  한 이미지당 blur 패치 수 범위
  blur_kernel_candidates    GaussianBlur 커널 크기 후보 (홀수 권장)
  blur_size_ratio_min/max   패치 크기 비율 범위 (이미지 너비/높이 대비)
  blur_shape_candidates     패치 모양 후보 ('circle', 'ellipse', 'rectangle', 'triangle')
  apply_blur_to_static      True: static 카메라 이미지에 blur 적용
  apply_blur_to_gripper     True: gripper 카메라 이미지에 blur 적용

================================================================================
출력 파일 구조  (processed_output_save_dir/{split}/ 아래)
================================================================================

  episode_{ridx:04d}_{i:07d}_noisy_action_image_added.npz
    ├── rgb_static         clean image  (PASS 2 렌더링, modify_clean_image=True일 때)
    ├── rgb_gripper        clean gripper image
    ├── rgb_static_noisy   noisy image  (PASS 1 렌더링)
    ├── rgb_gripper_noisy  noisy gripper image
    ├── actions            원본 absolute action
    ├── rel_actions        원본 relative action
    ├── robot_obs          원본 robot observation
    └── scene_obs          원본 scene observation

  lang_annotations/
    ├── auto_lang_ann_modified.npy       clean env 기준 색상 치환 annotation
    ├── auto_lang_ann_noisy_modified.npy noisy env 기준 색상 치환 annotation
    └── block_color_map_log.json         ridx별 {clean/noisy}_color_map, {original/clean/noisy}_ann 기록

  replay_frames_lang_ranges_[...]/      save_log=True일 때 프레임 PNG + distance plot

================================================================================
일반적인 사용 순서
================================================================================

  # 1. 블록 URDF 색상 다양화 (최초 1회)
  python calvin_color_customizer.py --diversify_blocks

  # 2. 테이블 텍스처 다양화 (OOD 사용 시, 최초 1회)
  python calvin_color_customizer.py --target A --num_colors 20
  python calvin_color_customizer.py --target B --num_colors 20
  python calvin_color_customizer.py --target C --num_colors 20
  python calvin_color_customizer.py --target D --num_colors 20

  # 3. 데이터셋 replay (action_from='dataset' 설정 후)
  export ORIGINAL_CALVIN_ABCD_D_DIR=/path/to/task_ABCD_D
  export ORIGINAL_CALVIN_ABCD_D_NOISE_DIR=/path/to/output
  python my_replayer.py

  # 4. annotation만 재생성 (env 실행 없이, action_from='fix_annotations' 설정 후)
  python my_replayer.py
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

# =============================================================================
# General — 공통 설정
# =============================================================================
verbose = False                         # True: 각 step마다 상세 디버그 메시지 출력

replay_action_type = "rel"              # 'rel': relative action (7D: dx,dy,dz,droll,dpitch,dyaw,gripper)
                                        # 'abs': absolute action (7D: x,y,z,roll,pitch,yaw,gripper)
action_from = 'dataset'                 # 'dataset': 데이터셋의 action을 재생 (run_env)
                                        # 'eval': 평가 로그의 action prediction을 재생 (replay_eval)
split = 'training'                      # 사용할 데이터 split ('training' 또는 'validation')
dist_measure_period = None                # N step마다 거리 측정 통계를 계산. None이면 비활성화

target_dataset_root_dir = f"{os.environ['ORIGINAL_CALVIN_ABCD_D_DIR']}/{split}"     # 원본 dataset 경로
processed_output_save_dir = f"{os.environ['ORIGINAL_CALVIN_ABCD_D_NOISE_DIR']}/{split}"  # 가공된 출력 저장 경로

# =============================================================================
# run_env 전용 — 데이터셋 replay 모드 설정
# =============================================================================
env_reset_period = 1                    # N step마다 환경을 데이터셋 상태로 리셋
                                        #   1: 매 step마다 리셋 (가장 정확한 재현)
                                        #   None: 시퀀스 시작 시에만 리셋
processing_limit = 20                    # 처리할 language annotation 시퀀스 수 (ridx 상한)
save_log = True                         # True: 프레임 PNG + distance plot 저장
save_log_limit = None
modify_clean_image = True               # True: PASS 2에서 clean env를 별도로 돌려 원본 action의 이미지를 생성
                                        #   (noisy action으로 렌더링한 이미지와 쌍을 이룬다)
add_ood_env = False                     # True: scene별 OOD 환경 config (조명/텍스처 변경) 사용
                                        # False: 기본 config 사용 (scene 알파벳에 맞는 _0 config)
num_colors = 20                          # add_ood_env=True일 때, scene당 OOD config 개수
                                        #   예: config_data_collection_A_0, _A_1, _A_2
random_config_selection = False          # True: OOD config를 랜덤 선택
                                        # False: ridx % num_colors 순차 선택
random_config_seed = 42                 # random_config_selection의 재현성을 위한 seed

# Action noise — noisy env (PASS 1)에 적용되는 action 노이즈
add_action_noise = True                 # True: rel_action에 가우시안 노이즈 추가
                                        #   noisy action → noisy image (PASS 1)
                                        #   원본 action → clean image (PASS 2, modify_clean_image=True일 때)
action_noise_period = 1                 # N step마다 노이즈 추가. None이면 비활성화
                                        #   1: 매 step마다, 2: 짝수 step마다, ...
noise_scale = 5                        # 노이즈 크기. pos_std(m) / rot_std(°) 동일값 사용
                                        #   큰 값 = 심한 노이즈, 작은 값(2.5 등) = 경미한 노이즈

# Block color diversification — 블록 색상 다양화
diversify_block_colors = True          # True: 매 시퀀스마다 red/blue/pink 블록의 색상을 랜덤 변경
                                        #   사전 준비 필요: calvin_color_customizer.py --diversify_blocks
num_block_colors = 8                   # 사용 가능한 색상 수 (= len(BASIC_CALLABLE_COLOR_LIST))
                                        #   변경하지 말 것 — URDF 파일이 0~10 인덱스로 생성되어 있음 -> 8로 변경 시 obj ood를 위해 block 색깔에 orange, purple, gray를 데이터셋 생성시에 사용하지 않음.
block_color_seed = 9999                 # 블록 색상 랜덤 선택의 재현성을 위한 seed
same_block_colors_for_clean = False      # True: env(noisy)와 env_no_noise(clean)가 동일한 블록 색 사용
                                        #   → 11색 중 3색만 사용
                                        # False: env와 env_no_noise에 서로 겹치지 않는 색 배정
                                        #   → 11색 중 6색 사용 (3색 + 3색, 겹침 없음)

# Gaussian blur augmentation — 랜덤 가우시안 블러 패치
add_random_gaussian_blur = True        # True: 이미지에 랜덤 위치/모양/크기의 blur 패치 적용
blur_random_seed = 1234                 # 블러 패치의 재현성을 위한 seed (frame index와 조합)
blur_num_patches_min = 3                # 한 이미지당 최소 blur 패치 수
blur_num_patches_max = 10               # 한 이미지당 최대 blur 패치 수
blur_kernel_candidates = [9, 15, 21, 31]  # GaussianBlur 커널 크기 후보 (홀수)
blur_size_ratio_min = 0.05              # 패치 크기의 최소 비율 (이미지 너비/높이 대비)
blur_size_ratio_max = 0.5               # 패치 크기의 최대 비율
blur_shape_candidates = ["triangle", "rectangle", "circle", "ellipse"]  # 패치 모양 후보
apply_blur_to_static = True             # True: static 카메라 이미지에 blur 적용
apply_blur_to_gripper = True            # True: gripper 카메라 이미지에 blur 적용

# =============================================================================
# replay_eval 전용 — 평가 로그 재생 모드 설정
# =============================================================================
NUM_SEQUENCES = 10                      # 재생할 eval sequence 수

# =============================================================================
# Helper functions
# =============================================================================
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

def get_scene_for_episode(episode_idx, scene_info):
    """
    scene_info: dict like {'calvin_scene_A': [1802438, 2406143], ...}
    Returns the scene letter (e.g. 'A') for a given episode index.
    """
    for scene_name, (s, e) in scene_info.items():
        if s <= episode_idx <= e:
            return scene_name.split('_')[-1]  # e.g. 'A' from 'calvin_scene_A'
    raise ValueError(f"Episode {episode_idx} does not belong to any scene in scene_info")

# =============================================================================
# Constants — 블록 색상 다양화에 사용되는 11가지 색상 목록
# 인덱스가 URDF 파일 접미사로 사용됨 (예: block_red_middle_8.urdf → orange)
# =============================================================================
BASIC_CALLABLE_COLOR_LIST = [
    "red",     # 0  (원본 block_red의 색)
    "blue",    # 1  (원본 block_blue의 색)
    "pink",    # 2  (원본 block_pink의 색)
    "black",   # 3
    "white",   # 4
    "green",   # 5
    "yellow",  # 6
    "brown",   # 7
    "orange",  # 8
    "purple",  # 9
    "gray",    # 10
]


def randomize_block_colors_in_cfg(cfg, rng, num_block_colors, indices=None):
    """Hydra compose() 후, instantiate() 전에 block URDF 경로를 랜덤 색상 인덱스로 교체.

    Calvin 환경의 3개 블록(red, blue, pink)에 대해 BASIC_CALLABLE_COLOR_LIST에서
    겹치지 않는 색상을 선택하여 해당 인덱스의 URDF 파일로 교체한다.

    예시 흐름:
      1. indices = [8, 5, 9] 가 선택 (orange, green, purple)
      2. block_red → blocks/block_red_middle_8.urdf (orange URDF)
         block_blue → blocks/block_blue_middle_5.urdf (green URDF)
         block_pink → blocks/block_pink_middle_9.urdf (purple URDF)
      3. color_map = {"red": "orange", "blue": "green", "pink": "purple"}

    Args:
        cfg: Hydra compose()로 생성된 config (scene.objects.movable_objects에 접근)
        rng: np.random.Generator — indices=None일 때 랜덤 선택에 사용
        num_block_colors: 사용 가능한 색상 수 (= len(BASIC_CALLABLE_COLOR_LIST) = 11)
        indices: 외부에서 전달할 색상 인덱스 배열 (길이 3). None이면 rng로 자동 선택.
                 env/env_no_noise 간 색상 겹침을 방지할 때 외부에서 미리 뽑아 전달.

    Returns:
        color_map: dict  e.g. {"red": "orange", "blue": "green", "pink": "purple"}
    """
    # 서로 겹치지 않는 3개의 색상 인덱스 선택
    if indices is None:
        indices = rng.choice(num_block_colors, size=3, replace=False)
    block_keys = ["block_red", "block_blue", "block_pink"]   # Hydra config 내 블록 키
    orig_colors = ["red", "blue", "pink"]                     # 원본 블록 색상 이름
    color_map = {}  # {원본색: 새색} 매핑 결과

    movable = cfg.scene.objects.movable_objects
    for block_key, orig_color, color_idx in zip(block_keys, orig_colors, indices):
        color_idx = int(color_idx)
        new_color_name = BASIC_CALLABLE_COLOR_LIST[color_idx]  # 인덱스 → 색상 이름
        color_map[orig_color] = new_color_name

        if block_key not in movable:
            continue
        original_file = movable[block_key].file  # e.g. "blocks/block_red_middle.urdf"
        # 뽑힌 색이 원래 블록 색과 같으면 원본 URDF를 그대로 사용 (인덱스 접미사 불필요)
        if new_color_name == orig_color:
            if verbose:
                print(f"  [block] {block_key}: keep original {original_file} ({orig_color} == {new_color_name})")
        else:
            # URDF 파일명에 색상 인덱스 추가: block_red_middle.urdf → block_red_middle_8.urdf
            base, ext = os.path.splitext(original_file)
            new_file = f"{base}_{color_idx}{ext}"
            movable[block_key].file = new_file
            if verbose:
                print(f"  [block] {block_key}: {original_file} -> {new_file} ({orig_color} -> {new_color_name})")

    return color_map


def modify_lang_annotations(ann, ridx, color_map):
    """단일 sequence(ridx)의 language annotation에서 색상 이름을 color_map에 따라 교체.

    ann["language"]["ann"][ridx]의 텍스트에서 red/blue/pink를 새 색상으로 치환.
    task name은 변경하지 않는다.

    Args:
        ann: auto_lang_ann.npy에서 로드된 dict (수정은 복사본에 대해 이뤄져야 함)
        ridx: 수정할 annotation의 인덱스
        color_map: {"red": "orange", "blue": "green", "pink": "brown"} 등

    Returns:
        original_text: 원본 annotation 텍스트
        new_text: 색상 치환된 annotation 텍스트
    """
    original_text = ann["language"]["ann"][ridx]
    new_text = original_text
    for orig_color, new_color in color_map.items():
        new_text = new_text.replace(orig_color, new_color)
    ann["language"]["ann"][ridx] = new_text
    return original_text, new_text


def noise(action, pos_std=0.01, rot_std=1):
    """위치(pos)와 방향(orn)에 가우시안 노이즈를 추가한다.

    pybullet의 multiplyTransforms로 노이즈를 합성하여
    물리적으로 유효한 변환을 보장한다.

    Args:
        action: (pos, orn, gripper) 튜플
            pos: [x, y, z] 위치 (미터)
            orn: [qx, qy, qz, qw] 쿼터니언 방향
            gripper: 그리퍼 상태 값
        pos_std: 위치 노이즈의 표준편차 (미터)
        rot_std: 회전 노이즈의 표준편차 (도, degree)

    Returns:
        (noisy_pos, noisy_orn, gripper) — gripper는 노이즈 없이 그대로 반환
    """
    pos, orn, gripper = action
    rot_std = np.radians(rot_std)                                    # degree → radian
    pos_noise = np.random.normal(0, pos_std, 3)                      # 3D 위치 노이즈
    rot_noise = p.getQuaternionFromEuler(np.random.normal(0, rot_std, 3))  # 3D 회전 노이즈 → 쿼터니언
    pos, orn = p.multiplyTransforms(pos, orn, pos_noise, rot_noise)  # 원본 × 노이즈 합성
    return pos, orn, gripper


def run_env():
    """데이터셋 replay 모드의 메인 함수.

    전체 흐름:
      1. scene_info 로드 → 에피소드별 scene 식별 (A/B/C/D)
      2. config 준비 (OOD 여부, 블록 색상 등)
      3. annotation 로드 → 시퀀스(ridx)별 에피소드 범위 파악
      4. 시퀀스별 루프:
         ├─ PASS 1 (noisy env): noisy action → noisy image 렌더링
         ├─ PASS 2 (clean env, optional): 원본 action → clean image 렌더링  
         ├─ npz 저장: 원본 + noisy 이미지 쌍
         └─ distance plot 저장
      5. 블록 색상 다양화 결과 저장 (annotation + color map log)
    """

    # ----- scene_info 로드 -----
    # scene_info: {'calvin_scene_A': [start, end], 'calvin_scene_B': [...], ...}
    # 에피소드 인덱스로 해당 에피소드가 어느 scene에 속하는지 판별하는 데 사용
    scene_info_path = Path(target_dataset_root_dir).parent / "scene_info.npy"
    if not scene_info_path.exists():
        scene_info_path = Path(target_dataset_root_dir) / "scene_info.npy"
    scene_info = np.load(scene_info_path, allow_pickle=True).item()
    print(f"[scene_info] loaded from {scene_info_path}")
    if verbose:
        print(f"  scene_info: {scene_info}")

    if add_ood_env:
        # ----- OOD 환경 config 목록 구성 -----
        # scene별로 조명/텍스처가 다른 config를 num_colors개씩 준비
        # 예: {'A': ['config_data_collection_A_0', ..._A_1, ..._A_2],
        #      'B': ['config_data_collection_B_0', ..._B_1, ..._B_2], ...}
        scene_letters = sorted(set(
            name.split('_')[-1] for name in scene_info.keys()
        ))
        config_name_lists = {
            sc: [f"config_data_collection_{sc}_{i}" for i in range(num_colors)]
            for sc in scene_letters
        }
        if verbose:
            print(f"[config] scene-specific configs: { {sc: len(v) for sc, v in config_name_lists.items()} }")
    else:
        config_name_lists = None
    conf_dir = Path(__file__).resolve().parent / "../../conf"
    conf_dir = conf_dir.resolve()

    if random_config_selection:
        config_rng = np.random.default_rng(random_config_seed)  # config 선택용 RNG (재현 가능)

    if diversify_block_colors:
        block_color_rng = np.random.default_rng(block_color_seed)  # 블록 색상 선택용 RNG (재현 가능)

    # ----- 초기 config 로드 (task oracle 등 공통 설정용) -----
    # 첫 번째 scene의 기본 config(_0)로 tasks 등 공통 객체를 생성
    first_scene = sorted(scene_info.keys())[0].split('_')[-1]
    initial_config_name = config_name_lists[first_scene][0] if add_ood_env else f"config_data_collection_{first_scene}_0"
    with initialize(config_path="../../conf"):
        cfg = compose(config_name=initial_config_name)

        root_dir = Path(target_dataset_root_dir)

        # ----- annotation 로드 -----
        # indx_ranges: [[start, end], ...] — 각 language annotation이 커버하는 에피소드 범위
        # task_names: ['turn_on_lightbulb', 'push_blue_block_left', ...] — 각 시퀀스의 task 이름
        ann_path = root_dir / "lang_annotations" / "auto_lang_ann.npy"
        ann = np.load(ann_path, allow_pickle=True).item()
        indx_ranges = ann["info"]["indx"]
        task_names = ann["language"]["task"]

        # 블록 색상 다양화: annotation을 deepcopy하여 색상 치환 결과를 별도로 관리
        if diversify_block_colors:
            from copy import deepcopy as _dc
            ann_modified = _dc(ann)          # clean env 기준 색상 치환 annotation
            ann_noisy_modified = _dc(ann)    # noisy env 기준 색상 치환 annotation
            color_map_log = []  # list of dicts: {ridx, start, end, task_name, clean_color_map, noisy_color_map, original_ann, clean_ann, noisy_ann}

        tasks = hydra.utils.instantiate(cfg.tasks)
        prev_info = None
        t1 = time.time()

    if save_log:
        # ----- 로그 저장 경로 생성 -----
        # 모든 주요 설정값을 폴더명에 포함시켜 실험 조건을 구분
        save_root = Path(processed_output_save_dir) / (
            f"replay_frames_lang_ranges_"
            f"[act({replay_action_type})"
            f"_reset({env_reset_period})"
            f"_noise({add_action_noise}_p{action_noise_period}_s{noise_scale})"
            f"_ood({add_ood_env})"
            f"_clean({modify_clean_image})"
            f"_block({diversify_block_colors}_same{same_block_colors_for_clean})"
            f"_blur({add_random_gaussian_blur})"
            f"_dist({dist_measure_period})"
            f"_limit({processing_limit})]"
        )
        save_root.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 시퀀스(ridx)별 메인 루프
    # 각 ridx는 하나의 language annotation에 대응하며,
    # indx_ranges[ridx] = [start, end]가 해당 시퀀스의 에피소드 범위
    # =========================================================================
    for ridx, (start, end) in enumerate(indx_ranges):

        if ridx >= processing_limit:
            break

        # ----- 이 시퀀스가 속한 scene 식별 -----
        scene_letter = get_scene_for_episode(start, scene_info)

        # ----- scene에 맞는 config 후보 목록 결정 -----
        if add_ood_env:
            # OOD 모드: 해당 scene의 여러 config 중 택 1
            config_name_list = config_name_lists[scene_letter]
        else:
            # 기본 모드: scene 알파벳에 맞는 기본 config(_0)만 사용
            config_name_list = [f"config_data_collection_{scene_letter}_0"]

        # ----- config 인덱스 선택 (noisy env / clean env 각각) -----
        if random_config_selection:
            cfg_idx = int(config_rng.integers(0, len(config_name_list)))
            if add_action_noise and modify_clean_image:
                cfg_idx_no_noise = int(config_rng.integers(0, len(config_name_list)))
        else:
            cfg_idx = ridx % len(config_name_list)
            if add_action_noise and modify_clean_image:
                cfg_idx_no_noise = (ridx + 1) % len(config_name_list)

        with initialize(config_path="../../conf"):
            print(f"\n=== range {ridx}: {start} ~ {end} (scene {scene_letter}) ===")

            # ----- Hydra config compose: noisy env용 -----
            selected_config_name = config_name_list[cfg_idx]
            selected_cfg = compose(config_name=selected_config_name)
            if verbose:
                print(f"[config] using {selected_config_name}")

            # ----- Hydra config compose: clean env용 (PASS 2) -----
            if add_action_noise and modify_clean_image:
                selected_config_name_no_noise = config_name_list[cfg_idx_no_noise]
                selected_cfg_no_noise = compose(config_name=selected_config_name_no_noise)
                if verbose:
                    print(f"[config_no_noise] using {selected_config_name_no_noise}")

            # ----- 블록 색상 랜덤화 (compose 후, instantiate 전에 config 수정) -----
            if diversify_block_colors:
                if add_action_noise and modify_clean_image and not same_block_colors_for_clean:
                    # [다른 색 모드] 11색 중 6색을 한번에 뽑아 env와 env_no_noise에 3개씩 분배
                    # → 두 환경 간 블록 색상이 절대 겹치지 않음
                    all_indices = block_color_rng.choice(num_block_colors, size=6, replace=False)
                    noisy_indices = all_indices[:3]   # env (noisy) 용
                    clean_indices = all_indices[3:]    # env_no_noise (clean) 용
                    noisy_color_map = randomize_block_colors_in_cfg(selected_cfg, block_color_rng, num_block_colors, indices=noisy_indices)
                    color_map = randomize_block_colors_in_cfg(selected_cfg_no_noise, block_color_rng, num_block_colors, indices=clean_indices)
                else:
                    # [같은 색 모드] 3색만 뽑아서 env에 적용
                    color_map = randomize_block_colors_in_cfg(selected_cfg, block_color_rng, num_block_colors)
                    noisy_color_map = color_map  # noisy/clean 색상 동일
                    if add_action_noise and modify_clean_image:
                        # env_no_noise에도 동일한 URDF 파일 경로를 복사 → 같은 블록 색
                        movable_no_noise = selected_cfg_no_noise.scene.objects.movable_objects
                        movable_noisy = selected_cfg.scene.objects.movable_objects
                        for bk in ["block_red", "block_blue", "block_pink"]:
                            if bk in movable_noisy and bk in movable_no_noise:
                                movable_no_noise[bk].file = movable_noisy[bk].file

                # ----- language annotation에서 블록 색상 이름 치환 -----
                # task_name에 red/blue/pink가 포함된 경우만 치환
                # 예: "push red block left" → "push orange block left"
                task_name = task_names[ridx]
                if any(c in task_name for c in ["red", "blue", "pink"]):
                    orig_ann, new_ann = modify_lang_annotations(ann_modified, ridx, color_map)
                    _, noisy_ann = modify_lang_annotations(ann_noisy_modified, ridx, noisy_color_map)
                    color_map_log.append({
                        "ridx": ridx, "start": start, "end": end, "task_name": task_name,
                        "clean_color_map": color_map.copy(),
                        "noisy_color_map": noisy_color_map.copy(),
                        "original_ann": orig_ann,
                        "clean_ann": new_ann,
                        "noisy_ann": noisy_ann,
                    })
                    if verbose:
                        print(f"  [lang] ridx={ridx} task={task_name}: '{orig_ann}' -> clean='{new_ann}' noisy='{noisy_ann}'")
                else:
                    color_map_log.append({
                        "ridx": ridx, "start": start, "end": end, "task_name": task_name,
                        "clean_color_map": color_map.copy(),
                        "noisy_color_map": noisy_color_map.copy(),
                        "original_ann": None, "clean_ann": None, "noisy_ann": None,
                    })
        
        # ============ PASS 1: noisy env ============
        # noisy action을 실행하여 noisy image를 렌더링하는 단계
        # config에 설정된 scene(조명/텍스처/블록색)으로 환경을 생성
        env = hydra.utils.instantiate(selected_cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

        prev_info = None
        prev_tcp_pos = None
        
        save_dir = save_root

        step_ids = []               # 시퀀스 내 local step 인덱스 (0, 1, 2, ...)

        # ----- 거리 측정 버퍼 -----
        # command displacement: action 명령의 xyz 크기 (||rel_xyz||)
        cmd_step_dists = []         # step별 command displacement
        # real displacement: 실제 TCP가 이동한 거리 (||tcp(t) - tcp(t-1)||)
        real_step_dists = []        # step별 real displacement

        # step별 xyz 좌표 (endpoint displacement 계산에 사용)
        cmd_step_xyzs = []          # command xyz 누적
        real_step_xyzs = []         # real TCP xyz 누적

        # dist_measure_period마다 집계된 "구간 합산" 거리
        cmd_period_sums = []        # command displacement 구간 합
        cmd_period_x = []           # 해당 구간의 마지막 step 번호
        real_period_sums = []       # real displacement 구간 합
        real_period_x = []

        # dist_measure_period마다 집계된 "구간 endpoint" 거리
        # (구간 첫 step ~ 마지막 step 간의 직선 거리)
        cmd_period_endpoint_dists = []
        cmd_period_endpoint_x = []
        real_period_endpoint_dists = []
        real_period_endpoint_x = []

        local_step = 0
        prev_tcp_pos = None

        # PASS 1 → PASS 2 간 데이터 전달용 버퍼
        # key: episode 인덱스, value: dict (원본/noisy 이미지 + action + obs)
        pending_outputs = {}

        for i in range(start, end + 1):
            # ----- 에피소드 데이터 로드 -----
            file = root_dir / f"episode_{i:07d}.npz"
            if not file.exists():
                print(f"[skip] missing: {file}")
                continue

            data = np.load(file)

            # ----- 환경 리셋 -----
            # env_reset_period step마다 데이터셋의 scene/robot 상태로 리셋
            # period=1이면 매 step마다 리셋 → action 재생이 가장 정확
            if env_reset_period is not None:
                if (i - start) % env_reset_period == 0:
                    if verbose:
                        print(f"reset {i}")
                    env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                    prev_info = None
                    prev_tcp_pos = None
            else:
                if i == start:
                    if verbose:
                        print(f"reset {i}")
                    env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                    prev_info = None
                    prev_tcp_pos = None

            # ----- action 로드 (replay_action_type에 따라 다름) -----
            if replay_action_type == "rel":
                action = data["rel_actions"]   # 7D: dx,dy,dz,droll,dpitch,dyaw,gripper
            elif replay_action_type == "abs":
                action = data["actions"].astype(np.float32)  # 7D: x,y,z,roll,pitch,yaw,gripper
                pos = action[:3].astype(np.float32)
                euler = action[3:6].astype(np.float32)
                gripper = np.array([action[6]], dtype=np.float32)
                action = (pos, euler, gripper)  # abs 모드는 튜플로 전달
            else:
                raise ValueError("Wrong replay_action_type")

            # ----- 노이즈 추가 전 원본 action 보존 (PASS 2에서 사용) -----
            action_original = action if not isinstance(action, np.ndarray) else action.copy()

            # ----- action에 가우시안 노이즈 추가 (rel 모드 전용) -----
            # action_noise_period마다 노이즈를 추가하여 noisy action 생성
            # 이 noisy action으로 PASS 1 env를 step → noisy image 렌더링
            if replay_action_type == "rel" and add_action_noise and action_noise_period is not None and i % action_noise_period == 0:
                if verbose:
                    print('#####################noise added#####################')
                    print('action (origin): ', action)
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
                if verbose:
                    print('action (noised): ', action)

            # ----- command displacement 기록 -----
            # action의 xyz 성분 크기 = 명령한 이동 거리
            cmd_xyz = np.asarray(action[:3], dtype=np.float32)
            cmd_dist = float(np.linalg.norm(cmd_xyz))
            cmd_step_dists.append(cmd_dist)
            cmd_step_xyzs.append(cmd_xyz.copy())

            # ----- env step: (noisy) action 실행 → observation 획득 -----
            o, _, _, info = env.step(action)

            # ----- real displacement 기록 -----
            # 실제 TCP(tool center point)의 이동 거리 = ||tcp(t) - tcp(t-1)||
            curr_tcp_pos = np.asarray(o["robot_obs"][:3], dtype=np.float32)
            real_step_xyzs.append(curr_tcp_pos.copy())

            if prev_tcp_pos is None:
                real_dist = 0.0
            else:
                real_dist = float(np.linalg.norm(curr_tcp_pos - prev_tcp_pos))
            prev_tcp_pos = curr_tcp_pos
            real_step_dists.append(real_dist)

            step_ids.append(local_step)

            # ----- dist_measure_period마다 구간 통계 계산 -----
            # 1) sum: 최근 N step의 거리 합산 (cmd / real 각각)
            # 2) endpoint: 구간 첫 step ~ 마지막 step 간 직선 거리
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

                if verbose:
                    print(
                        f"[range {ridx:04d}] "
                        f"step {start_idx:04d} ~ {end_idx:04d} | "
                        f"CMD sum={cmd_sum:.6f} | "
                        f"REAL sum={real_sum:.6f} | "
                        f"CMD endpoint={cmd_endpoint_dist:.6f} | "
                        f"REAL endpoint={real_endpoint_dist:.6f}"
                    )

            local_step += 1

            if verbose:
                print(info["scene_info"]["lights"]["led"]["logical_state"])
                if prev_info is not None:
                    print(tasks.get_task_info(prev_info, info))
            prev_info = deepcopy(info)

            img = o["rgb_obs"]["rgb_static"]
            gripper_img = o["rgb_obs"]["rgb_gripper"]

            if add_random_gaussian_blur:
                # ----- 가우시안 블러 augmentation -----
                # frame index(i) 기반으로 카메라별 독립 seed 생성 → 재현 가능
                # static: blur_random_seed + i*2 + 0
                # gripper: blur_random_seed + i*2 + 1
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

            if save_log and (save_log_limit is None or save_log_limit > (i - start)):
                cv2.imwrite(str(save_dir / f"frame_{i:07d}.png"), img[:, :, ::-1])
                cv2.imwrite(str(save_dir / f"gripper_frame_{i:07d}.png"), gripper_img[:, :, ::-1])

            # ----- PASS 1 결과를 pending_outputs에 저장 -----
            # 키 구성:
            #   rgb_static / rgb_gripper:    원본 데이터셋 이미지 (PASS 2에서 clean 이미지로 교체됨)
            #   rgb_static_noisy / rgb_gripper_noisy: PASS 1에서 렌더링된 noisy 이미지
            #   actions / rel_actions:        원본 action
            #   robot_obs / scene_obs:        원본 observation
            #   robot_obs_xyz:                noisy env에서의 실제 TCP 위치
            pending_outputs[i] = {
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

        # ----- noisy env 정리 -----
        # PASS 1이 끝나면 pybullet physics client를 해제하고 GC 수행
        # EGL 충돌 방지를 위해 PASS 2와 동시 실행하지 않고 순차적으로 처리
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
        gc.collect()

        # ============ PASS 2: clean env ============
        # 원본(noise 없는) action으로 env_no_noise를 step하여 clean image 생성
        # PASS 1의 noisy image와 쌍을 이루는 "정답" 이미지
        # (PASS 1 후 순차적으로 실행하여 EGL 충돌 방지)
        if add_action_noise and modify_clean_image:
            print(f"  [PASS 2] clean env for range {ridx}: {start} ~ {end}")
            env_no_noise = hydra.utils.instantiate(selected_cfg_no_noise.env, show_gui=False, use_vr=False, use_scene_info=True)

            for i in range(start, end + 1):
                if i not in pending_outputs:
                    continue

                file = root_dir / f"episode_{i:07d}.npz"
                data = np.load(file)

                if env_reset_period is not None:
                    if (i - start) % env_reset_period == 0:
                        env_no_noise.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
                else:
                    if i == start:
                        env_no_noise.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])

                if replay_action_type == "rel":
                    action_original = data["rel_actions"]
                elif replay_action_type == "abs":
                    action_original = data["actions"].astype(np.float32)
                    pos = action_original[:3].astype(np.float32)
                    euler = action_original[3:6].astype(np.float32)
                    gripper_val = np.array([action_original[6]], dtype=np.float32)
                    action_original = (pos, euler, gripper_val)

                # clean env을 원본 action으로 step → clean image 획득
                o_no_noise, _, _, _ = env_no_noise.step(action_original)

                # pending_outputs의 rgb_static/rgb_gripper를 clean 이미지로 교체
                # (최종 npz에는 rgb_static=clean, rgb_static_noisy=noisy 쌍이 저장됨)
                pending_outputs[i]["rgb_static"] = o_no_noise["rgb_obs"]["rgb_static"]
                pending_outputs[i]["rgb_gripper"] = o_no_noise["rgb_obs"]["rgb_gripper"]

                if save_log and (save_log_limit is None or save_log_limit > (i - start)):
                    cv2.imwrite(str(save_dir / f"clean_frame_{i:07d}.png"), o_no_noise["rgb_obs"]["rgb_static"][:, :, ::-1])
                    cv2.imwrite(str(save_dir / f"clean_gripper_frame_{i:07d}.png"), o_no_noise["rgb_obs"]["rgb_gripper"][:, :, ::-1])

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

        # ============ npz 저장 ============
        # 각 에피소드별로 episode_{ridx:04d}_{i:07d}_noisy_action_image_added.npz 생성
        # ridx prefix로 overlapping episode range 간 덮어쓰기 방지
        # 포함 데이터: 원본(clean) 이미지, noisy 이미지, action, observation
        for i in sorted(pending_outputs.keys()):
            out_file = Path(processed_output_save_dir) / f"episode_{ridx:04d}_{i:07d}_noisy_action_image_added.npz"
            np.savez_compressed(out_file, **pending_outputs[i])

        del pending_outputs
        gc.collect()
        cv2.destroyAllWindows()

        # =========================================================
        # 시퀀스별 distance plot 저장
        # - REAL step dist: step별 TCP 이동 거리 (초록선)
        # - REAL sum: N step 구간 합산 (빨간 마커)
        # - REAL endpoint dist: N step 구간 양 끝점 직선거리 (보라 마커)
        # =========================================================
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

    # =========================================================================
    # 블록 색상 다양화 결과 저장
    # diversify_block_colors=True일 때만 실행
    # 1) auto_lang_ann_modified.npy: 색상 치환된 language annotation
    #    - 원본의 "push red block" → "push orange block" 등
    # 2) block_color_map_log.json: 시퀀스별 색상 매핑 기록
    #    - 어떤 ridx에서 어떤 색으로 바뀌었는지 추적 가능
    # =========================================================================
    if diversify_block_colors:
        import json as _json

        out_dir = Path(processed_output_save_dir) / "lang_annotations"
        out_dir.mkdir(parents=True, exist_ok=True)

        n_processed = min(processing_limit, len(indx_ranges))

        # ----- modified annotation 저장 (processing_limit 범위만 잘라서) -----
        ann_out = {
            "language": {
                "ann": ann_modified["language"]["ann"][:n_processed],
                "task": ann_modified["language"]["task"][:n_processed],
            },
            "info": {
                "indx": ann_modified["info"]["indx"][:n_processed],
                "episodes": ann_modified["info"]["episodes"][:n_processed]
                            if "episodes" in ann_modified["info"] else [],
            },
        }
        if "emb" in ann_modified["language"]:
            ann_out["language"]["emb"] = ann_modified["language"]["emb"][:n_processed]

        ann_out_path = out_dir / "auto_lang_ann_modified.npy"
        np.save(ann_out_path, ann_out)
        print(f"[saved] clean annotations ({n_processed}/{len(indx_ranges)}) -> {ann_out_path}")

        # ----- noisy annotation 저장 (noisy env 블록 색상 기준) -----
        ann_noisy_out = {
            "language": {
                "ann": ann_noisy_modified["language"]["ann"][:n_processed],
                "task": ann_noisy_modified["language"]["task"][:n_processed],
            },
            "info": {
                "indx": ann_noisy_modified["info"]["indx"][:n_processed],
                "episodes": ann_noisy_modified["info"]["episodes"][:n_processed]
                            if "episodes" in ann_noisy_modified["info"] else [],
            },
        }
        if "emb" in ann_noisy_modified["language"]:
            ann_noisy_out["language"]["emb"] = ann_noisy_modified["language"]["emb"][:n_processed]

        ann_noisy_out_path = out_dir / "auto_lang_ann_noisy_modified.npy"
        np.save(ann_noisy_out_path, ann_noisy_out)
        print(f"[saved] noisy annotations ({n_processed}/{len(indx_ranges)}) -> {ann_noisy_out_path}")

        # color mapping log 저장
        log_out_path = out_dir / "block_color_map_log.json"
        log_data = []
        for entry in color_map_log:
            log_data.append({
                "ridx": int(entry["ridx"]),
                "start": int(entry["start"]),
                "end": int(entry["end"]),
                "task_name": entry["task_name"],
                "clean_color_map": entry["clean_color_map"],
                "noisy_color_map": entry["noisy_color_map"],
                "original_ann": entry["original_ann"],
                "clean_ann": entry["clean_ann"],
                "noisy_ann": entry["noisy_ann"],
            })
        with open(log_out_path, "w", encoding="utf-8") as f:
            _json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"[saved] color mapping log ({len(log_data)} entries) -> {log_out_path}")

def fix_annotations_only():
    """env를 생성하지 않고, block color RNG 시퀀스만 재현하여
    language annotation과 block_color_map_log를 PASS 2(clean env) 색상 기준으로 재생성한다.

    run_env()와 동일한 RNG 소비 패턴을 재현하므로,
    동일한 seed/설정에서 동일한 색상 매핑이 생성된다.

    전제 조건:
      - diversify_block_colors = True
      - 기존 run_env()와 동일한 설정값 (seed, processing_limit 등)
    """
    assert diversify_block_colors, "fix_annotations_only requires diversify_block_colors=True"

    # ----- scene_info 로드 -----
    scene_info_path = Path(target_dataset_root_dir).parent / "scene_info.npy"
    if not scene_info_path.exists():
        scene_info_path = Path(target_dataset_root_dir) / "scene_info.npy"
    scene_info = np.load(scene_info_path, allow_pickle=True).item()
    print(f"[scene_info] loaded from {scene_info_path}")

    # ----- OOD config 목록 구성 (env는 만들지 않지만, config 개수에 따라 RNG 소비가 달라짐) -----
    if add_ood_env:
        scene_letters = sorted(set(
            name.split('_')[-1] for name in scene_info.keys()
        ))
        config_name_lists = {
            sc: [f"config_data_collection_{sc}_{i}" for i in range(num_colors)]
            for sc in scene_letters
        }
    else:
        config_name_lists = None

    # ----- RNG 초기화 (run_env와 동일한 seed) -----
    if random_config_selection:
        config_rng = np.random.default_rng(random_config_seed)

    block_color_rng = np.random.default_rng(block_color_seed)

    # ----- annotation 로드 -----
    root_dir = Path(target_dataset_root_dir)
    ann_path = root_dir / "lang_annotations" / "auto_lang_ann.npy"
    ann = np.load(ann_path, allow_pickle=True).item()
    indx_ranges = ann["info"]["indx"]
    task_names = ann["language"]["task"]

    from copy import deepcopy as _dc
    ann_modified = _dc(ann)
    ann_noisy_modified = _dc(ann)
    color_map_log = []

    # =========================================================================
    # 시퀀스(ridx)별 루프 — run_env()와 동일한 RNG 소비 순서를 재현
    # =========================================================================
    for ridx, (start, end) in enumerate(indx_ranges):
        if ridx >= processing_limit:
            break

        scene_letter = get_scene_for_episode(start, scene_info)

        # ----- config 후보 목록 (len만 필요) -----
        if add_ood_env:
            config_name_list = config_name_lists[scene_letter]
        else:
            config_name_list = [f"config_data_collection_{scene_letter}_0"]

        # ----- config RNG 소비 (run_env와 동일 패턴 유지) -----
        if random_config_selection:
            _cfg_idx = int(config_rng.integers(0, len(config_name_list)))
            if add_action_noise and modify_clean_image:
                _cfg_idx_no_noise = int(config_rng.integers(0, len(config_name_list)))

        # ----- block color RNG 소비 (run_env와 동일 패턴 유지) -----
        if add_action_noise and modify_clean_image and not same_block_colors_for_clean:
            # [다른 색 모드] 6색 뽑기 → noisy(앞 3개), clean(뒤 3개)
            all_indices = block_color_rng.choice(num_block_colors, size=6, replace=False)
            noisy_indices = all_indices[:3]
            clean_indices = all_indices[3:]
            # clean env 기준 color_map 생성 (cfg 수정은 불필요 — annotation만 변경)
            color_map = {}
            noisy_color_map = {}
            orig_colors = ["red", "blue", "pink"]
            for orig_color, color_idx in zip(orig_colors, clean_indices):
                color_map[orig_color] = BASIC_CALLABLE_COLOR_LIST[int(color_idx)]
            for orig_color, color_idx in zip(orig_colors, noisy_indices):
                noisy_color_map[orig_color] = BASIC_CALLABLE_COLOR_LIST[int(color_idx)]
        else:
            # [같은 색 모드 또는 PASS 2 없음] 3색 뽑기
            indices = block_color_rng.choice(num_block_colors, size=3, replace=False)
            color_map = {}
            orig_colors = ["red", "blue", "pink"]
            for orig_color, color_idx in zip(orig_colors, indices):
                color_map[orig_color] = BASIC_CALLABLE_COLOR_LIST[int(color_idx)]
            noisy_color_map = color_map  # 같은 색

        # ----- language annotation 치환 -----
        task_name = task_names[ridx]
        if any(c in task_name for c in ["red", "blue", "pink"]):
            orig_ann, new_ann = modify_lang_annotations(ann_modified, ridx, color_map)
            _, noisy_ann = modify_lang_annotations(ann_noisy_modified, ridx, noisy_color_map)
            color_map_log.append({
                "ridx": ridx, "start": start, "end": end, "task_name": task_name,
                "clean_color_map": color_map.copy(),
                "noisy_color_map": noisy_color_map.copy(),
                "original_ann": orig_ann,
                "clean_ann": new_ann,
                "noisy_ann": noisy_ann,
            })
            if verbose:
                print(f"  [lang] ridx={ridx} task={task_name}: '{orig_ann}' -> clean='{new_ann}' noisy='{noisy_ann}'")
        else:
            color_map_log.append({
                "ridx": ridx, "start": start, "end": end, "task_name": task_name,
                "clean_color_map": color_map.copy(),
                "noisy_color_map": noisy_color_map.copy(),
                "original_ann": None, "clean_ann": None, "noisy_ann": None,
            })

        if ridx % 500 == 0:
            print(f"  processed ridx={ridx}/{min(processing_limit, len(indx_ranges))}")

    # =========================================================================
    # 결과 저장
    # =========================================================================
    import json as _json

    out_dir = Path(processed_output_save_dir) / "lang_annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_processed = min(processing_limit, len(indx_ranges))

    ann_out = {
        "language": {
            "ann": ann_modified["language"]["ann"][:n_processed],
            "task": ann_modified["language"]["task"][:n_processed],
        },
        "info": {
            "indx": ann_modified["info"]["indx"][:n_processed],
            "episodes": ann_modified["info"]["episodes"][:n_processed]
                        if "episodes" in ann_modified["info"] else [],
        },
    }
    if "emb" in ann_modified["language"]:
        ann_out["language"]["emb"] = ann_modified["language"]["emb"][:n_processed]

    ann_out_path = out_dir / "auto_lang_ann_modified.npy"
    np.save(ann_out_path, ann_out)
    print(f"[saved] clean annotations ({n_processed}/{len(indx_ranges)}) -> {ann_out_path}")

    # ----- noisy annotation 저장 -----
    ann_noisy_out = {
        "language": {
            "ann": ann_noisy_modified["language"]["ann"][:n_processed],
            "task": ann_noisy_modified["language"]["task"][:n_processed],
        },
        "info": {
            "indx": ann_noisy_modified["info"]["indx"][:n_processed],
            "episodes": ann_noisy_modified["info"]["episodes"][:n_processed]
                        if "episodes" in ann_noisy_modified["info"] else [],
        },
    }
    if "emb" in ann_noisy_modified["language"]:
        ann_noisy_out["language"]["emb"] = ann_noisy_modified["language"]["emb"][:n_processed]

    ann_noisy_out_path = out_dir / "auto_lang_ann_noisy_modified.npy"
    np.save(ann_noisy_out_path, ann_noisy_out)
    print(f"[saved] noisy annotations ({n_processed}/{len(indx_ranges)}) -> {ann_noisy_out_path}")

    log_out_path = out_dir / "block_color_map_log.json"
    log_data = []
    for entry in color_map_log:
        log_data.append({
            "ridx": int(entry["ridx"]),
            "start": int(entry["start"]),
            "end": int(entry["end"]),
            "task_name": entry["task_name"],
            "clean_color_map": entry["clean_color_map"],
            "noisy_color_map": entry["noisy_color_map"],
            "original_ann": entry["original_ann"],
            "clean_ann": entry["clean_ann"],
            "noisy_ann": entry["noisy_ann"],
        })
    with open(log_out_path, "w", encoding="utf-8") as f:
        _json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"[saved] color mapping log ({len(log_data)} entries) -> {log_out_path}")


def replay_eval():
    """평가 로그 재생 모드의 메인 함수.

    전체 흐름:
      1. validation 환경 로드
      2. eval_sequences.json에서 initial_state + subtask 목록 로드
      3. 평가 로그(log_root)에서 모델의 action prediction npy 로드
      4. 각 eval sequence마다:
         - initial_state로 환경 리셋
         - action prediction을 순차적으로 step
         - 프레임 PNG + distance plot 저장
    """
    from calvin_env.envs.play_table_env import get_env
    from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
    from omegaconf import OmegaConf
    import json
    import glob

    # ----- 환경 및 task oracle 로드 -----
    env = get_env(Path(os.environ['ORIGINAL_CALVIN_ABCD_D_DIR']) / "validation", show_gui=False)
    conf_dir = Path(os.environ['CONF_DIR']) / "conf"
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)

    save_dir = "./replay_frames_eval/"   # 프레임/plot 저장 디렉토리
    os.makedirs(save_dir, exist_ok=True)

    # ----- 평가 로그 경로 -----
    # 각 실험 결과 폴더를 지정 (log_root 아래에 0_3/, 1_2/ 등 시퀀스별 폴더가 있음)
    #log_root = "/home/ksshin/projects/sparc/UD-VLA_old_with_data/logs/calvin_exp_main/univla_calvin_abcd_video_i2ia_dis/eval_20260214_1616/log" #baseline
    #log_root = "/home/ksshin/projects/sparc/UD-VLA_old_with_data/logs/calvin_exp_main/univla_calvin_abcd_video_i2ia_dis/eval_20260217_0035/log" #obj 1 (blue->cyan)
    log_root = "/home/ksshin/projects/sparc/UD-VLA_old_with_data/logs/calvin_exp_main/univla_calvin_abcd_video_i2ia_dis/eval_20260215_2128/log" #env

    # ----- eval_sequences 로드 -----
    # 구조: [[initial_state, [subtask1, subtask2, ...]], ...]
    # initial_state: 환경 초기 상태, subtasks: 5개 연속 task 이름
    with open(
        "/home/ksshin/projects/sparc/UD-VLA/reference/RoboVLMs/configs/data/calvin/eval_sequences.json",
        "r",
    ) as f:
        eval_sequences = json.load(f)
    eval_sequences = eval_sequences[:NUM_SEQUENCES]

    # ----- 각 eval_sequence의 subtask 목록 추출 -----
    subtask_list = [item[1] for item in eval_sequences]

    # ----- sequence별 action prediction 로드 -----
    # 로그 폴더 구조: log_root/{x}_{y}/ (x=sequence번호, y=성공 subtask 수)
    # 그 안에 {x}_action_pred/action_pred_*.npy (각 10-step chunk)
    actions_per_sequence = []
    success_counts = []

    for x in range(NUM_SEQUENCES):
        # 해당 sequence 번호의 폴더 탐색 (예: "0_3" = sequence 0, 3개 성공)
        candidates = glob.glob(os.path.join(log_root, f"{x}_*"))
        candidates = [p for p in candidates if os.path.isdir(p)]

        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one folder for sequence {x}, but got: {candidates}"
            )

        # 폴더명에서 성공 횟수 파싱 (예: "0_3" → y=3)
        xy_dir = candidates[0]
        folder_name = os.path.basename(xy_dir)
        _, y = folder_name.split("_")
        success_counts.append(int(y))

        # action prediction npy 파일들을 로드하여 1D 리스트로 평탄화
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

    # =========================================================================
    # eval_sequences[i] ↔ actions_per_sequence[i] 1:1 매칭 replay
    # 각 sequence마다: initial_state로 리셋 → action을 순차 실행 → 프레임/plot 저장
    # =========================================================================
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
        if verbose:
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

                if verbose:
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
    # ----- 메인 실행 -----
    start_time = time.time()
    if action_from == 'dataset':
        run_env()          # 데이터셋 replay 모드
    elif action_from == 'eval':
        replay_eval()      # 평가 로그 replay 모드
    elif action_from == 'fix_annotations':
        fix_annotations_only()  # annotation만 재생성 (env 없이)
    else:
        print('wrong action_from')
    end_time = time.time()
    print("실행 시간:", end_time - start_time, "초")

    # ----- 원본 데이터셋의 메타 파일들을 출력 디렉토리로 복사 -----
    # training에 필요한 statistics, annotation, scene_info 등을
    # 가공된 데이터셋 폴더에도 함께 보관하기 위함
    src_dir = Path(target_dataset_root_dir)
    dst_dir = Path(processed_output_save_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    items_to_copy = [
        "statistics.yaml",                # action 정규화 통계
        "lang_annotations",               # language annotation (원본)
        "ep_start_end_ids.npy",           # 에피소드 시작/끝 인덱스
        ".hydra",                         # Hydra config 기록
        "ep_lens.npy",                    # 에피소드 길이
        "scene_info.npy",                 # scene별 에피소드 범위
        "lang_paraphrase-MiniLM-L3-v2",   # sentence embedding 캐시
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
