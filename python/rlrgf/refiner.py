from __future__ import annotations

from .execution_state import AgentExecutionState, TaskClassification


class SafeRefiner:
    def build_refinement_prompt(self, state: AgentExecutionState, verifier_rationale: str) -> str:
        answer = state.candidate_answer.strip()
        guidance = verifier_rationale.strip() or "Revise the answer conservatively."
        prompt = (
            f"{state.prompt}\n\n"
            f"Current candidate answer:\n{answer}\n\n"
            f"Verifier guidance:\n{guidance}\n\n"
            "Revise the answer conservatively. Preserve supported content, reduce unsupported claims, "
            "and prefer explicit uncertainty over speculation."
        )
        if state.task_classification == TaskClassification.CODING:
            prompt += "\nIf the user asked for code, keep the answer in code form rather than replacing it with prose."
        if state.evidence_summary.strip():
            prompt += f"\nEvidence summary:\n{state.evidence_summary.strip()}"
        return prompt
