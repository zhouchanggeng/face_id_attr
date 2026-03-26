"""头部姿态估计模块。

基于 PFLD 98 点关键点 + cv2.solvePnP 估算偏航角(Yaw)、俯仰角(Pitch)、翻滚角(Roll)。
不需要额外模型，复用 PFLDAligner 的关键点预测能力。

原理:
    1. 从 98 点中选取 6 个稳定的 3D 参考点（鼻尖、下巴、左右眼角、左右嘴角）
    2. 定义这些点在标准 3D 人脸模型中的坐标
    3. 用 solvePnP 求解 2D-3D 对应关系，得到旋转向量
    4. 将旋转向量转换为欧拉角 (Yaw, Pitch, Roll)

WFLW 98 点索引:
    鼻尖: 54, 下巴: 16, 左眼外角: 60, 右眼外角: 72, 左嘴角: 76, 右嘴角: 82
"""
import cv2
import numpy as np
from typing import Optional, Tuple


# 标准 3D 人脸模型参考点（单位: mm，原点在鼻尖）
# 坐标系: X 向右, Y 向下, Z 向前（朝向相机）
MODEL_POINTS_3D = np.float64([
    [0.0, 0.0, 0.0],          # 鼻尖
    [0.0, 63.6, 12.5],        # 下巴
    [-43.3, -32.7, 26.0],     # 左眼外角
    [43.3, -32.7, 26.0],      # 右眼外角
    [-28.9, 28.9, 24.1],      # 左嘴角
    [28.9, 28.9, 24.1],       # 右嘴角
])

# 对应的 WFLW 98 点索引
LANDMARK_IDX_6 = [54, 16, 60, 72, 76, 82]


def estimate_head_pose(landmarks_98: np.ndarray,
                       image_size: Tuple[int, int]) -> dict:
    """从 98 点关键点估算头部姿态。

    Args:
        landmarks_98: shape (98, 2)，原图坐标系下的 98 个关键点
        image_size: (width, height) 图像尺寸

    Returns:
        {
            "yaw": float,    # 偏航角（左右转头），正值=向右，单位: 度
            "pitch": float,  # 俯仰角（抬头低头），正值=抬头，单位: 度
            "roll": float,   # 翻滚角（歪头），正值=向右歪，单位: 度
            "rotation_vector": np.ndarray,
            "translation_vector": np.ndarray,
        }
    """
    w, h = image_size

    # 提取 6 个关键点
    pts_2d = landmarks_98[LANDMARK_IDX_6].astype(np.float64)

    # 相机内参（近似）
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.float64([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ])
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # solvePnP
    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D, pts_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                "rotation_vector": None, "translation_vector": None}

    # 旋转向量 -> 旋转矩阵 -> 欧拉角
    rmat, _ = cv2.Rodrigues(rvec)
    yaw, pitch, roll = _rotation_matrix_to_euler(rmat)

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "rotation_vector": rvec,
        "translation_vector": tvec,
    }


def _rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """旋转矩阵转欧拉角 (yaw, pitch, roll)，单位: 度。

    约定: X 轴向右，Y 轴向下，Z 轴向前（相机坐标系）。
    """
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


def draw_head_pose_axes(image: np.ndarray, landmarks_98: np.ndarray,
                        pose: dict, axis_length: float = 50) -> np.ndarray:
    """在图像上绘制头部姿态的 3D 坐标轴。

    红色=X轴(左右), 绿色=Y轴(上下), 蓝色=Z轴(前后/深度)
    """
    if pose.get("rotation_vector") is None:
        return image

    h, w = image.shape[:2]
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.float64([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ])
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # 3D 坐标轴端点
    axes_3d = np.float64([
        [axis_length, 0, 0],   # X 轴
        [0, axis_length, 0],   # Y 轴
        [0, 0, axis_length],   # Z 轴
    ])

    # 投影到 2D
    nose_2d = landmarks_98[54].astype(np.float64).reshape(1, 2)
    axes_2d, _ = cv2.projectPoints(axes_3d, pose["rotation_vector"],
                                    pose["translation_vector"],
                                    camera_matrix, dist_coeffs)

    origin = tuple(nose_2d[0].astype(int))
    vis = image.copy()
    cv2.line(vis, origin, tuple(axes_2d[0].ravel().astype(int)), (0, 0, 255), 2)  # X 红
    cv2.line(vis, origin, tuple(axes_2d[1].ravel().astype(int)), (0, 255, 0), 2)  # Y 绿
    cv2.line(vis, origin, tuple(axes_2d[2].ravel().astype(int)), (255, 0, 0), 2)  # Z 蓝
    return vis
