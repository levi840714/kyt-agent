from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import interrupt
from pydantic import ValidationError

from kyt_agent import rules
from kyt_agent.audit import digest
from kyt_agent.graph import prompts
from kyt_agent.graph.budget import exhausted_reason
from kyt_agent.graph.deps import Deps
from kyt_agent.graph.state import CaseState
from kyt_agent.graph.tools import Investigator, ToolOutcome, label_evidence
from kyt_agent.models import AddressNode, Category, Evidence, Review
from kyt_agent.report import ReportDraft, finalize, unknown_evidence, write_case_files

Usage = dict[str, int]
# 新登記的地址帶有這些標籤時，視為發現風險跡象並重新計算收尾提示
_RISK_SIGNALS: frozenset[Category] = frozenset({"sanctioned", "hack", "mixer", "bridge"})


class ReportError(RuntimeError):
    """LLM 無法產出可解析的報告。"""


class CaseNodes:
    def __init__(self, deps: Deps) -> None:
        self._deps = deps
        self._investigator = Investigator(deps.chain, deps.labels, deps.tokens, deps.settings)

    def screen(self, state: CaseState) -> dict[str, Any]:
        target = state["target"]
        label = self._deps.labels.get(target)
        hits = rules.screen(target, self._deps.labels)
        self._audit(state, "case_opened", target=target, model=self._deps.model_name)
        self._audit(state, "rule_screen", hits=[hit.model_dump() for hit in hits])
        return {
            "rule_hits": hits,
            "nodes": {target: AddressNode(address=target, depth=0, label=label)},
            "evidence": label_evidence(label) if label else {},
        }

    def agent(self, state: CaseState) -> dict[str, Any]:
        response = self._deps.agent_model.invoke(state["messages"])
        usage = _usage(response)
        self._audit(
            state,
            "llm_call",
            node="agent",
            model=self._deps.model_name,
            tool_calls=[call["name"] for call in response.tool_calls],
            **usage,
        )
        return {"messages": [response], **_accumulate(state, usage)}

    def budget_guard(self, state: CaseState) -> dict[str, Any]:
        reason = exhausted_reason(state, self._deps.settings)
        if reason is None:
            return {"budget_note": None}
        self._audit(state, "budget_exhausted", reason=reason)
        skipped = [
            ToolMessage(content=f"未執行：{reason}", tool_call_id=call["id"] or "")
            for call in _last_ai(state).tool_calls
        ]
        return {"budget_note": reason, "messages": skipped}

    def tools(self, state: CaseState) -> dict[str, Any]:
        remaining = state["tool_call_limit"] - state["tool_calls"]
        nodes: dict[str, AddressNode] = {}
        evidence: dict[str, Evidence] = {}
        messages: list[BaseMessage] = []
        used = misses = 0
        # resume 舊版 checkpoint 時可能沒有這兩個欄位，缺省視為尚未累積收尾提示
        streak = state.get("calls_without_risk", 0)
        for call in _last_ai(state).tool_calls:
            call_id = call["id"] or ""
            if used >= remaining:
                messages.append(
                    ToolMessage(content="未執行：工具呼叫次數已達上限", tool_call_id=call_id)
                )
                continue
            known = {**state["nodes"], **nodes}
            known_evidence = {**state["evidence"], **evidence}
            outcome = self._investigator.execute(call["name"], call["args"], known, known_evidence)
            used += 1
            misses += outcome.miss
            streak = 0 if _reveals_risk(outcome, known) else streak + 1
            nodes.update(outcome.nodes)
            evidence.update(outcome.evidence)
            messages.append(ToolMessage(content=outcome.content, tool_call_id=call_id))
            self._audit(
                state,
                "tool_call",
                tool=call["name"],
                args=call["args"],
                result_sha256=digest(outcome.content),
                result_preview=outcome.content[:200],
                # 工具結果只有代號，記下新配的代號與 hash 對應，audit log 才能獨立追溯
                evidence={
                    item.id: item.ref
                    for item in outcome.evidence.values()
                    if item.kind == "tx" and item.id not in known_evidence
                },
                **({"resolved_tx": outcome.resolved_tx} if outcome.resolved_tx else {}),
            )
        hinted = state.get("wrap_up_hinted", False)
        hint = streak >= self._deps.settings.wrap_up_hint_after and not hinted
        if hint:
            # 放在所有 ToolMessage 之後，才不會拆開 tool_call 與 ToolMessage 的配對
            messages.append(HumanMessage(prompts.wrap_up_message(streak)))
            self._audit(state, "wrap_up_hint", calls_without_risk=streak)
        return {
            "messages": messages,
            "nodes": nodes,
            "evidence": evidence,
            "tool_calls": state["tool_calls"] + used,
            "snapshot_misses": state["snapshot_misses"] + misses,
            "calls_without_risk": streak,
            "wrap_up_hinted": hinted or hint,
        }

    def report(self, state: CaseState) -> dict[str, Any]:
        request = prompts.report_request(
            label_evidence=[
                f"{item.id}：{item.summary}"
                for item in state["evidence"].values()
                if item.kind == "label"
            ],
            rule_hits=[f"{hit.rule}：{hit.detail}" for hit in state["rule_hits"]],
            budget_note=state["budget_note"],
        )
        messages: list[BaseMessage] = [*state["messages"], HumanMessage(request)]
        draft, usage = self._draft(state, messages)
        invalid = unknown_evidence(draft, state["evidence"])
        if invalid:
            self._audit(state, "evidence_check", invalid=invalid, retry=True)
            messages += [
                AIMessage(draft.model_dump_json()),
                HumanMessage(prompts.evidence_retry_message(invalid)),
            ]
            draft, retry_usage = self._draft(state, messages)
            usage = _add(usage, retry_usage)
            invalid = unknown_evidence(draft, state["evidence"])
        self._audit(state, "evidence_check", invalid=invalid, retry=False)
        floor = rules.risk_floor(state["target"], state["nodes"])
        previous = state["report"]
        report = finalize(
            draft,
            evidence=state["evidence"],
            floor=floor,
            unverified=invalid,
            version=previous.version + 1 if previous else 1,
            labels=self._deps.labels,
        )
        self._audit(
            state,
            "report_generated",
            version=report.version,
            risk_level=report.risk_level,
            llm_risk_level=report.llm_risk_level,
            risk_floor=floor,
        )
        return {"report": report, **_accumulate(state, usage)}

    def review(self, state: CaseState) -> dict[str, Any]:
        report = state["report"]
        assert report is not None
        settings = self._deps.settings
        tokens_left = state["input_tokens"] + state["output_tokens"] < settings.max_tokens
        allow_more = state["review_round"] <= settings.max_review_rounds and tokens_left
        request = {
            "case_id": state["case_id"],
            "target": state["target"],
            "round": state["review_round"],
            "allow_more": allow_more,
            "report": report.model_dump(mode="json"),
        }
        review = self._await_review(request)
        self._audit(state, "human_decision", **review.model_dump())
        if review.decision == "request_more":
            limit = state["tool_call_limit"] + settings.supplement_tool_calls
            remaining = limit - state["tool_calls"]
            return {
                "reviews": [review],
                "review_round": state["review_round"] + 1,
                "tool_call_limit": limit,
                "budget_note": None,
                "calls_without_risk": 0,
                "wrap_up_hinted": False,
                "messages": [
                    HumanMessage(prompts.supplement_message(report, review.comment, remaining))
                ],
            }
        status = "approved" if review.decision == "approve" else "rejected"
        # eval 的自動核准不是真正結案，寫入 var/cases 只會混入大量測試報告
        if not self._deps.auto_approve:
            write_case_files(
                settings.var_dir / "cases" / state["case_id"],
                state["target"],
                report,
                [*state["reviews"], review],
            )
        self._audit(state, "case_closed", status=status)
        return {"reviews": [review], "status": status}

    def _await_review(self, request: dict[str, Any]) -> Review:
        if self._deps.auto_approve:
            return Review(
                decision="approve", comment="eval 自動核准", reviewer="eval", round=request["round"]
            )
        error: str | None = None
        while True:
            # 輸入無效時再次中斷而非拋錯，否則錯誤的 resume 值會卡死整個 thread
            payload = interrupt(request if error is None else {**request, "error": error})
            data = {**payload, "round": request["round"]} if isinstance(payload, dict) else payload
            try:
                review = Review.model_validate(data)
            except ValidationError as exc:
                error = "；".join(item["msg"] for item in exc.errors())
                continue
            if review.decision == "request_more" and not request["allow_more"]:
                error = "已達補查輪數或 token 預算上限，無法要求補查"
                continue
            return review

    def _draft(self, state: CaseState, messages: list[BaseMessage]) -> tuple[ReportDraft, Usage]:
        usage: Usage = {"input_tokens": 0, "output_tokens": 0}
        attempt = list(messages)
        for _ in range(2):
            result = self._deps.drafter.invoke(attempt)
            call_usage = _usage(result["raw"])
            usage = _add(usage, call_usage)
            self._audit(state, "llm_call", node="report", model=self._deps.model_name, **call_usage)
            if result["parsed"] is not None:
                return result["parsed"], usage
            attempt.append(HumanMessage(prompts.parse_retry_message(result["parsing_error"])))
        raise ReportError("報告格式解析失敗")

    def _audit(self, state: CaseState, event: str, **data: Any) -> None:
        self._deps.audit.record(state["case_id"], event, **data)


def route_after_screen(state: CaseState) -> str:
    return "report" if state["rule_hits"] else "agent"


def route_after_agent(state: CaseState) -> str:
    return "budget_guard" if _last_ai(state).tool_calls else "report"


def route_after_guard(state: CaseState) -> str:
    return "report" if state["budget_note"] else "tools"


def route_after_review(state: CaseState) -> str:
    return "agent" if state["status"] == "investigating" else END


def _reveals_risk(outcome: ToolOutcome, known: dict[str, AddressNode]) -> bool:
    return any(
        address not in known and node.label is not None and node.label.category in _RISK_SIGNALS
        for address, node in outcome.nodes.items()
    )


def _last_ai(state: CaseState) -> AIMessage:
    message = state["messages"][-1]
    assert isinstance(message, AIMessage)
    return message


def _usage(message: BaseMessage) -> Usage:
    metadata = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": metadata.get("input_tokens", 0),
        "output_tokens": metadata.get("output_tokens", 0),
    }


def _add(left: Usage, right: Usage) -> Usage:
    return {key: left[key] + right[key] for key in left}


def _accumulate(state: CaseState, usage: Usage) -> Usage:
    return {
        "input_tokens": state["input_tokens"] + usage["input_tokens"],
        "output_tokens": state["output_tokens"] + usage["output_tokens"],
    }
