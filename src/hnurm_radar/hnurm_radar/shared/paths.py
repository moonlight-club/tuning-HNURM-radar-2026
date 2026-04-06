"""
paths.py — 项目路径统一管理模块（支持场景切换）

所有代码中需要引用项目文件的地方，统一从此模块导入路径常量，
避免硬编码绝对路径，方便项目迁移。

场景切换：
    在 configs/main_config.yaml 中设置 global.scene 即可切换场景：
      scene: "competition"  → 赛场环境（28×15m）
      scene: "lab"          → 实验室测试环境
    不同场景使用不同的地图、点云等资源文件，无需手动替换文件。

原理：
    通过本文件自身的位置（__file__）向上回溯 4 层目录，自动定位项目根目录。

维护须知：
    1. 整个项目文件夹移动到其他位置 → 无需任何修改，自动适配。
    2. 项目内部目录结构变更 → 只需修改本文件中对应的路径常量。
    3. 如果移动了本文件自身 → 需要调整 PROJECT_ROOT 的回溯层数。
    4. 新增场景 → 在 main_config.yaml 的 scenes 下添加配置即可。
"""

import os

# 项目根目录（通过当前文件位置向上推导）
# paths.py 位于 src/hnurm_radar/hnurm_radar/shared/paths.py
PROJECT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

# ======================== 一级目录 ========================
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
MAP_DIR = os.path.join(PROJECT_ROOT, "map")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RECORD_DIR = os.path.join(PROJECT_ROOT, "record")
TEST_RESOURCES_DIR = os.path.join(PROJECT_ROOT, "test_resources")

# ======================== 配置文件 ========================
MAIN_CONFIG_PATH = os.path.join(CONFIGS_DIR, "main_config.yaml")
DETECTOR_CONFIG_PATH = os.path.join(CONFIGS_DIR, "detector_config.yaml")
CONVERTER_CONFIG_PATH = os.path.join(CONFIGS_DIR, "converter_config.yaml")
PERSPECTIVE_CALIB_PATH = os.path.join(CONFIGS_DIR, "perspective_calib.json")
BYTETRACK_CONFIG_PATH = os.path.join(CONFIGS_DIR, "bytetrack.yaml")
BOTSORT_CONFIG_PATH = os.path.join(CONFIGS_DIR, "botsort.yaml")
HAP_CONFIG_PATH = os.path.join(CONFIGS_DIR, "HAP_config.json")
ICP_RVIZ_PATH = os.path.join(CONFIGS_DIR, "icp.rviz")

# ======================== 模型权重 ========================
STAGE_ONE_PATH = os.path.join(WEIGHTS_DIR, "stage_one.pt")
STAGE_TWO_PATH = os.path.join(WEIGHTS_DIR, "stage_two.pt")
STAGE_THREE_PATH = os.path.join(WEIGHTS_DIR, "stage_three.pt")

# ======================== ultralytics 路径 ========================
ULTRALYTICS_DIR = os.path.join(PROJECT_ROOT, "src", "hnurm_radar")


# ======================== 场景配置加载 ========================
def _load_scene_config():
    """从 main_config.yaml 读取当前场景配置，返回场景字典。

    如果读取失败（文件不存在、格式错误等），回退到赛场默认值。
    """
    defaults = {
        "std_map": "map/std_map.png",
        "pfa_map": "map/pfa_map_2025.jpg",
        "pcd_file": "data/pcds.pcd",
        "downsampled_pcd": "data/pcds_downsampled.pcd",
        "field_width": 28.0,
        "field_height": 15.0,
    }
    try:
        from ruamel.yaml import YAML
        yaml = YAML()
        with open(MAIN_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.load(f)
        scene_name = cfg.get("global", {}).get("scene", "competition")
        scenes = cfg.get("scenes", {})
        scene_cfg = scenes.get(scene_name, {})
        if scene_cfg:
            # 用场景配置覆盖默认值
            for k in defaults:
                if k in scene_cfg:
                    defaults[k] = scene_cfg[k]
    except Exception:
        pass  # 读取失败时使用默认值
    return defaults


_scene = _load_scene_config()

# ======================== 场景相关路径 ========================
STD_MAP_PATH = os.path.join(PROJECT_ROOT, _scene["std_map"])
PFA_MAP_2025_PATH = os.path.join(PROJECT_ROOT, _scene["pfa_map"])
PFA_MAP_RED_PATH = os.path.join(PROJECT_ROOT, _scene.get("pfa_map_red", _scene["pfa_map"]))
PFA_MAP_BLUE_PATH = os.path.join(PROJECT_ROOT, _scene.get("pfa_map_blue", _scene["pfa_map"]))
PFA_MAP_MASK_2025_PATH = os.path.join(MAP_DIR, "pfa_map_mask_2025.jpg")

PCD_FILE_PATH = os.path.join(PROJECT_ROOT, _scene["pcd_file"])
PCD_DOWNSAMPLED_PATH = os.path.join(PROJECT_ROOT, _scene["downsampled_pcd"])

# 场景物理尺寸（供 perspective_calibrator 等使用）
FIELD_WIDTH = float(_scene["field_width"])
FIELD_HEIGHT = float(_scene["field_height"])


# ======================== 工具函数 ========================
def resolve_path(path_str: str) -> str:
    """将路径解析为绝对路径。

    如果 path_str 已经是绝对路径，直接返回；
    否则视为相对于 PROJECT_ROOT 的相对路径，拼接后返回。
    """
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(PROJECT_ROOT, path_str)


def get_scene_name() -> str:
    """返回当前场景名称（如 'competition' 或 'lab'）"""
    try:
        from ruamel.yaml import YAML
        yaml = YAML()
        with open(MAIN_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.load(f)
        return cfg.get("global", {}).get("scene", "competition")
    except Exception:
        return "competition"
