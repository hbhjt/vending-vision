from vision.config import settings
from vision.pose_estimator import PoseEstimator
from vision.face_detector import FaceDetector
from vision.person_detector import PersonDetector
from vision.logger import logger
from vision.age_gender_estimator import AgeGenderEstimator
from vision.camera_manager import get_all_camera_statuses

def check_camera():
    """
    检查摄像头是否可打开。

    mock 模式下不强制检查真实摄像头。
    """
    if settings.MOCK_SCENARIO != "off":
        return {
            "ok": True,
            "message": f"mock mode enabled: {settings.MOCK_SCENARIO}"
        }

    try:
        statuses = get_all_camera_statuses()
        ok = all(status.get("ok") for status in statuses.values())
        return {
            "ok": ok,
            "message": "top/front cameras checked",
            "detail": statuses,
        }
    except Exception as e:
        return {
            "ok": False,
            "message": str(e)
        }


def check_pose_model():
    """
    检查 MediaPipe Pose 是否可初始化。
    """
    try:
        _ = PoseEstimator()
        return {
            "ok": True,
            "message": "pose estimator initialized"
        }
    except Exception as e:
        logger.exception("Pose model check failed")
        return {
            "ok": False,
            "message": str(e)
        }


def check_face_detector():
    """
    检查人脸检测器是否可初始化。
    """
    try:
        detector = FaceDetector()
        return {
            "ok": True,
            "message": f"face detector initialized: {detector.backend}"
        }
    except Exception as e:
        logger.exception("Face detector check failed")
        return {
            "ok": False,
            "message": str(e)
        }


def check_person_detector():
    """
    检查轻量人体检测模型状态。

    人体检测是 proximity 的增强项；模型不存在时允许回退到姿态辅助。
    """
    try:
        detector = PersonDetector()
        status = detector.status()

        return {
            "ok": True,
            "modelReady": status["ready"],
            "mode": status["backend"],
            "message": (
                "person detector ready"
                if status["ready"]
                else "person detector not ready, fallback to pose proximity"
            ),
            "detail": status,
        }

    except Exception as e:
        logger.exception("Person detector check failed")
        return {
            "ok": True,
            "modelReady": False,
            "mode": "error",
            "message": f"person detector check failed, fallback to pose: {e}",
        }


def check_age_gender_model():
    """
    检查年龄性别模型状态。

    注意：
    年龄性别模型不是强依赖。
    如果模型文件不存在，可以 fallback 到 mock，
    所以 ok 仍然返回 True，但 mode 会显示 mock。
    """
    try:
        estimator = AgeGenderEstimator()
        status = estimator.status()

        return {
            "ok": True,
            "modelReady": status["ok"],
            "mode": status["mode"],
            "message": status["message"]
        }

    except Exception as e:
        logger.exception("Age/Gender model check failed")
        return {
            "ok": True,
            "modelReady": False,
            "mode": "mock",
            "message": f"age/gender check failed, fallback to mock: {e}"
        }

def run_self_check():
    """
    运行完整自检。
    """
    checks = {
        "camera": check_camera(),
        "pose": check_pose_model(),
        "face": check_face_detector(),
        "person": check_person_detector(),
        "ageGender": check_age_gender_model()
    }

    required_ok = (
        checks["camera"]["ok"]
        and checks["pose"]["ok"]
        and checks["face"]["ok"]
    )

    return {
        "ok": required_ok,
        "checks": checks
    }
