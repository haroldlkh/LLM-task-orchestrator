import math
from .base import BanditPolicy


class UCBPolicy(BanditPolicy):
    def __init__(self, arm_ids: list[str] | None = None, exploration: float = 1.2, state: dict | None = None):
        self.exploration = float(exploration)
        self.total_pulls = 0
        self.arms: dict[str, dict] = {}
        for arm_id in arm_ids or []:
            self.arms[arm_id] = {"pulls": 0, "reward_sum": 0.0, "mean_reward": 0.0}
        if state:
            self.total_pulls = int(state.get("total_pulls", 0) or 0)
            for arm_id, arm_state in (state.get("arms", {}) or {}).items():
                self.arms[arm_id] = {
                    "pulls": int(arm_state.get("pulls", 0) or 0),
                    "reward_sum": float(arm_state.get("reward_sum", 0.0) or 0.0),
                    "mean_reward": float(arm_state.get("mean_reward", 0.0) or 0.0),
                }

    def ensure_arm(self, arm_id: str) -> None:
        self.arms.setdefault(arm_id, {"pulls": 0, "reward_sum": 0.0, "mean_reward": 0.0})

    def select_arm(self, allowed_arm_ids: list[str], context: dict | None = None) -> str:
        if not allowed_arm_ids:
            raise ValueError("allowed_arm_ids must be non-empty")
        for arm_id in allowed_arm_ids:
            self.ensure_arm(arm_id)
        cold = [arm_id for arm_id in allowed_arm_ids if self.arms[arm_id]["pulls"] == 0]
        if cold:
            return sorted(cold)[0]
        total = max(self.total_pulls, 1)
        best_arm = None
        best_score = None
        for arm_id in allowed_arm_ids:
            arm = self.arms[arm_id]
            bonus = self.exploration * math.sqrt(math.log(total + 1) / max(arm["pulls"], 1))
            score = float(arm["mean_reward"]) + bonus
            if best_arm is None or score > best_score:
                best_arm = arm_id
                best_score = score
        return best_arm

    def update(self, arm_id: str, reward: float, meta: dict | None = None) -> None:
        self.ensure_arm(arm_id)
        arm = self.arms[arm_id]
        arm["pulls"] += 1
        arm["reward_sum"] += float(reward)
        arm["mean_reward"] = arm["reward_sum"] / max(arm["pulls"], 1)
        self.total_pulls += 1

    def apply_decay(self, factor: float) -> None:
        factor = float(factor)
        if factor <= 0 or factor > 1:
            return
        for arm in self.arms.values():
            arm["pulls"] = max(int(round(arm["pulls"] * factor)), 0)
            arm["reward_sum"] *= factor
            arm["mean_reward"] = arm["reward_sum"] / max(arm["pulls"], 1) if arm["pulls"] > 0 else 0.0
        self.total_pulls = max(sum(arm["pulls"] for arm in self.arms.values()), 0)

    def export_state(self) -> dict:
        return {"policy": "ucb", "exploration": self.exploration, "total_pulls": self.total_pulls, "arms": self.arms}

    @classmethod
    def from_state(cls, payload: dict):
        return cls(exploration=float(payload.get("exploration", 1.2) or 1.2), state=payload)
