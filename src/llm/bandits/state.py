from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_state_decay(payload: dict | None, decay: float, ttl_hours: int | None = None) -> dict | None:
    if not payload:
        return payload
    out = dict(payload)
    updated_at = payload.get("updated_at")
    if ttl_hours and updated_at:
        try:
            age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(updated_at)).total_seconds() / 3600.0
            if age_hours > ttl_hours:
                return None
        except Exception:
            pass
    for key in ("payload_bandit", "lane_bandit"):
        bandit = out.get(key)
        if not isinstance(bandit, dict):
            continue
        if bandit.get("policy") == "ucb":
            arms = bandit.get("arms", {}) or {}
            total = 0
            for arm in arms.values():
                pulls = max(int(round((arm.get("pulls", 0) or 0) * decay)), 0)
                reward_sum = float(arm.get("reward_sum", 0.0) or 0.0) * decay
                arm["pulls"] = pulls
                arm["reward_sum"] = reward_sum
                arm["mean_reward"] = reward_sum / pulls if pulls > 0 else 0.0
                total += pulls
            bandit["total_pulls"] = total
    out["updated_at"] = utc_now_iso()
    return out
