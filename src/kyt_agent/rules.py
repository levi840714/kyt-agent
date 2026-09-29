from kyt_agent.labels import LabelStore
from kyt_agent.models import HIGH_RISK_CATEGORIES, AddressNode, RiskLevel, RuleHit, max_risk


def screen(target: str, labels: LabelStore) -> list[RuleHit]:
    label = labels.get(target)
    if label is None or label.category != "sanctioned":
        return []
    return [
        RuleHit(
            rule="target_sanctioned",
            address=target,
            level="SEVERE",
            detail=f"{label.name}（來源 {label.source}）",
        )
    ]


def risk_floor(target: str, nodes: dict[str, AddressNode]) -> RiskLevel:
    levels: list[RiskLevel] = ["LOW"]
    for node in nodes.values():
        if node.label is None:
            continue
        if node.address == target.lower() and node.label.category == "sanctioned":
            levels.append("SEVERE")
        elif node.depth <= 1 and node.label.category in HIGH_RISK_CATEGORIES:
            levels.append("HIGH")
    return max_risk(*levels)
