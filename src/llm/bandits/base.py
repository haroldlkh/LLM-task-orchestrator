class BanditPolicy:
    def select_arm(self, allowed_arm_ids: list[str], context: dict | None = None) -> str:
        raise NotImplementedError

    def update(self, arm_id: str, reward: float, meta: dict | None = None) -> None:
        raise NotImplementedError

    def export_state(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_state(cls, payload: dict):
        raise NotImplementedError
