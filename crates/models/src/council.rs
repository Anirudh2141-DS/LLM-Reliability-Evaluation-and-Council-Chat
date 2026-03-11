use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Result from a single council evaluator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilVerdict {
    pub query_id: Uuid,
    pub evaluator_role: EvaluatorRole,
    pub critic_score: f32,
    pub grounding_score: f32,
    pub safety_flag: bool,
    pub leakage_risk: f32,
    pub confidence_score: f32,
    pub reasoning: String,
    pub claims: Vec<ClaimAssessment>,
}

/// Roles that evaluators can take in the council.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvaluatorRole {
    Generator,
    GroundingInspector,
    SafetyAuditor,
    Critic,
}

impl std::fmt::Display for EvaluatorRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvaluatorRole::Generator => write!(f, "generator"),
            EvaluatorRole::GroundingInspector => write!(f, "grounding_inspector"),
            EvaluatorRole::SafetyAuditor => write!(f, "safety_auditor"),
            EvaluatorRole::Critic => write!(f, "critic"),
        }
    }
}

/// Assessment of a single claim in the generated answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimAssessment {
    pub claim_text: String,
    pub supported: bool,
    pub supporting_chunk_id: Option<Uuid>,
    pub confidence: f32,
}

/// Aggregated result from all council members.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilAggregation {
    pub query_id: Uuid,
    pub verdicts: Vec<CouncilVerdict>,
    pub final_decision: Decision,
    pub aggregate_safety_flag: bool,
    pub aggregate_grounding_score: f32,
    pub disagreement_score: f32,
    pub supported_claim_ratio: f32,
}

/// Final decision after council deliberation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Accept,
    Abstain,
    Escalate,
    RequestRetrieval,
    AskClarification,
}

impl std::fmt::Display for Decision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Decision::Accept => write!(f, "accept"),
            Decision::Abstain => write!(f, "abstain"),
            Decision::Escalate => write!(f, "escalate"),
            Decision::RequestRetrieval => write!(f, "request_retrieval"),
            Decision::AskClarification => write!(f, "ask_clarification"),
        }
    }
}
