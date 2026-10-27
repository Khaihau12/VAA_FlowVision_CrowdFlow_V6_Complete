from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Decision:
    zone: str
    occupancy: int
    capacity: int
    utilization: float
    status: str
    action: str
    alternative: str | None
    explanation: str


DEFAULT_ADJACENCY = {
    "Zone A": ["Zone B"],
    "Zone B": ["Zone A", "Zone C"],
    "Zone C": ["Zone B"],
}


def classify(utilization: float, busy_threshold: float = 0.65, congested_threshold: float = 0.85) -> tuple[str, str]:
    if utilization >= congested_threshold:
        return "CONGESTED", "REDIRECT"
    if utilization >= busy_threshold:
        return "BUSY", "RESTRICT"
    return "NORMAL", "KEEP"


def evaluate_zone(
    zone: str,
    occupancy: int,
    capacities: dict[str, int],
    occupancies: dict[str, int],
    busy_threshold: float = 0.65,
    congested_threshold: float = 0.85,
    adjacency: dict[str, list[str]] | None = None,
) -> Decision:
    adjacency = adjacency or DEFAULT_ADJACENCY
    capacity = max(1, int(capacities.get(zone, 1)))
    utilization = float(occupancy) / capacity
    status, action = classify(utilization, busy_threshold, congested_threshold)

    alternative = None
    if action == "REDIRECT":
        candidates = adjacency.get(zone, [])
        if candidates:
            alternative = min(
                candidates,
                key=lambda z: float(occupancies.get(z, 0)) / max(1, int(capacities.get(z, 1))),
            )
    if action == "KEEP":
        explanation = f"{zone} đang dưới ngưỡng đông; giữ nguyên hướng di chuyển hiện tại."
    elif action == "RESTRICT":
        explanation = f"{zone} đang tiến gần sức chứa; hạn chế dòng người đi vào và tiếp tục theo dõi các mốc tiếp theo."
    elif alternative:
        explanation = f"{zone} vượt ngưỡng ùn tắc; ưu tiên điều hướng dòng người mới sang {alternative}, là khu lân cận có tỷ lệ tải thấp nhất."
    else:
        explanation = f"{zone} vượt ngưỡng ùn tắc; chưa cấu hình khu lân cận thay thế để điều hướng."

    return Decision(zone, int(occupancy), capacity, utilization, status, action, alternative, explanation)
