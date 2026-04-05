def build_payload_arms(targets: list[int]) -> list[dict]:
    arms = []
    for target in targets:
        target = int(target)
        arms.append({"arm_id": f"payload_{target}", "target_request_tokens": target})
    return arms
