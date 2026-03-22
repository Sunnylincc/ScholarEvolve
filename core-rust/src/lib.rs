use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct Weights {
    semantic: f64,
    topic: f64,
    recency: f64,
    quality: f64,
    duplication: f64,
    overfit: f64,
    #[serde(default)]
    novelty: f64,
    #[serde(default)]
    exploration: f64,
}

#[derive(Debug, Deserialize)]
struct DiversityConfig {
    #[serde(default = "default_dup_threshold")]
    duplicate_threshold: f64,
    #[serde(default = "default_topic_target")]
    topic_concentration_target: f64,
    #[serde(default = "default_mmr_lambda")]
    mmr_lambda: f64,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    paper_id: String,
    semantic_similarity: f64,
    categories: Vec<String>,
    recency_bonus: f64,
    quality_proxy: f64,
    embedding: Vec<f32>,
    #[serde(default)]
    prior_impressions: f64,
    #[serde(default)]
    prior_clicks: f64,
}

#[derive(Debug, Deserialize, Default)]
struct UserState {
    #[serde(default)]
    topic_weights: HashMap<String, f64>,
    #[serde(default)]
    topic_impressions: HashMap<String, f64>,
    #[serde(default)]
    topic_clicks: HashMap<String, f64>,
    #[serde(default)]
    event_counter: u64,
    #[serde(default)]
    last_updated_ts: f64,
}

#[derive(Debug, Deserialize)]
struct RerankPayload {
    weights: Weights,
    diversity: DiversityConfig,
    candidates: Vec<Candidate>,
    #[serde(default)]
    user_state: UserState,
    top_k: usize,
}

#[derive(Debug, Serialize)]
struct RerankResult {
    paper_id: String,
    score: f64,
    reasons: HashMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct FeedbackConfig {
    #[serde(default = "default_click_delta")]
    click_delta: f64,
    #[serde(default = "default_bookmark_delta")]
    bookmark_delta: f64,
    #[serde(default = "default_skip_delta")]
    skip_delta: f64,
    #[serde(default = "default_half_life_days")]
    half_life_days: f64,
}

#[derive(Debug, Deserialize)]
struct FeedbackPayload {
    event_type: String,
    timestamp: f64,
    categories: Vec<String>,
    #[serde(default)]
    user_state: UserState,
    #[serde(default)]
    feedback_config: Option<FeedbackConfig>,
}

#[derive(Debug, Serialize)]
struct FeedbackResult {
    topic_weights: HashMap<String, f64>,
    topic_impressions: HashMap<String, f64>,
    topic_clicks: HashMap<String, f64>,
    event_counter: u64,
    last_updated_ts: f64,
}

fn default_dup_threshold() -> f64 {
    0.93
}
fn default_topic_target() -> f64 {
    0.35
}
fn default_mmr_lambda() -> f64 {
    0.75
}
fn default_click_delta() -> f64 {
    0.8
}
fn default_bookmark_delta() -> f64 {
    1.4
}
fn default_skip_delta() -> f64 {
    -0.3
}
fn default_half_life_days() -> f64 {
    30.0
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
}

fn norm(a: &[f32]) -> f64 {
    dot(a, a).sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let denom = norm(a) * norm(b);
    if denom <= 1e-12 {
        0.0
    } else {
        dot(a, b) / denom
    }
}

fn clamp01(v: f64) -> f64 {
    v.max(0.0).min(1.0)
}

fn calc_topic_match(categories: &[String], user_state: &UserState) -> (f64, f64) {
    if categories.is_empty() {
        return (0.0, 0.0);
    }

    let mut affinity_sum = 0.0;
    let mut uncertainty_sum = 0.0;
    for cat in categories {
        affinity_sum += *user_state.topic_weights.get(cat).unwrap_or(&0.0);
        let imps = *user_state.topic_impressions.get(cat).unwrap_or(&0.0);
        let clicks = *user_state.topic_clicks.get(cat).unwrap_or(&0.0);
        let ctr = (clicks + 1.0) / (imps + 2.0);
        uncertainty_sum += (ctr * (1.0 - ctr) / (imps + 1.0)).sqrt();
    }
    let denom = categories.len() as f64;
    (affinity_sum / denom, uncertainty_sum / denom)
}

fn calc_topic_concentration(categories: &[String], user_state: &UserState) -> f64 {
    if categories.is_empty() {
        return 0.0;
    }
    let mut running = 0.0;
    for cat in categories {
        running += *user_state.topic_impressions.get(cat).unwrap_or(&0.0);
    }
    running / (categories.len() as f64 * (user_state.event_counter.max(1) as f64))
}

fn base_score(candidate: &Candidate, payload: &RerankPayload) -> (f64, HashMap<String, f64>) {
    let (topic_match, uncertainty) = calc_topic_match(&candidate.categories, &payload.user_state);
    let topic_concentration = calc_topic_concentration(&candidate.categories, &payload.user_state);
    let duplication_penalty = (candidate.semantic_similarity - payload.diversity.duplicate_threshold).max(0.0);
    let overfit_penalty = (topic_concentration - payload.diversity.topic_concentration_target).max(0.0);

    let novelty = (1.0 - candidate.prior_impressions / 10.0).max(0.0);
    let exploration_bonus = uncertainty;

    let semantic_term = payload.weights.semantic * clamp01(candidate.semantic_similarity);
    let topic_term = payload.weights.topic * topic_match;
    let recency_term = payload.weights.recency * clamp01(candidate.recency_bonus);
    let quality_term = payload.weights.quality * clamp01(candidate.quality_proxy);
    let novelty_term = payload.weights.novelty * novelty;
    let exploration_term = payload.weights.exploration * exploration_bonus;
    let dup_term = -payload.weights.duplication * duplication_penalty;
    let overfit_term = -payload.weights.overfit * overfit_penalty;

    let total = semantic_term
        + topic_term
        + recency_term
        + quality_term
        + novelty_term
        + exploration_term
        + dup_term
        + overfit_term;

    let mut reasons = HashMap::new();
    reasons.insert("semantic".to_string(), semantic_term);
    reasons.insert("topic".to_string(), topic_term);
    reasons.insert("recency".to_string(), recency_term);
    reasons.insert("quality".to_string(), quality_term);
    reasons.insert("novelty".to_string(), novelty_term);
    reasons.insert("exploration".to_string(), exploration_term);
    reasons.insert("dup_penalty".to_string(), dup_term);
    reasons.insert("overfit_penalty".to_string(), overfit_term);
    (total, reasons)
}

fn mmr_select(payload: RerankPayload) -> Vec<RerankResult> {
    let n = payload.candidates.len();
    if n == 0 || payload.top_k == 0 {
        return vec![];
    }

    let mut remaining: Vec<usize> = (0..n).collect();
    let mut selected: Vec<usize> = Vec::with_capacity(payload.top_k.min(n));
    let mut base_scores: Vec<f64> = Vec::with_capacity(n);
    let mut reason_map: Vec<HashMap<String, f64>> = Vec::with_capacity(n);

    for cand in &payload.candidates {
        let (score, reasons) = base_score(cand, &payload);
        base_scores.push(score);
        reason_map.push(reasons);
    }

    while !remaining.is_empty() && selected.len() < payload.top_k.min(n) {
        let mut best_idx = remaining[0];
        let mut best_score = f64::NEG_INFINITY;

        for &idx in &remaining {
            let diversity_penalty = selected
                .iter()
                .map(|&sidx| cosine(&payload.candidates[idx].embedding, &payload.candidates[sidx].embedding))
                .fold(0.0_f64, f64::max)
                .max(0.0);

            let mmr_score = payload.diversity.mmr_lambda * base_scores[idx]
                - (1.0 - payload.diversity.mmr_lambda) * diversity_penalty;

            if mmr_score > best_score {
                best_score = mmr_score;
                best_idx = idx;
            }
        }

        selected.push(best_idx);
        remaining.retain(|&x| x != best_idx);
    }

    selected
        .into_iter()
        .map(|idx| {
            let mut reasons = reason_map[idx].clone();
            reasons.insert("final".to_string(), base_scores[idx]);
            RerankResult {
                paper_id: payload.candidates[idx].paper_id.clone(),
                score: base_scores[idx],
                reasons,
            }
        })
        .collect()
}

fn exp_decay_factor(delta_seconds: f64, half_life_days: f64) -> f64 {
    if delta_seconds <= 0.0 {
        return 1.0;
    }
    let half_life_seconds = half_life_days * 24.0 * 3600.0;
    2.0_f64.powf(-(delta_seconds / half_life_seconds))
}

#[pyfunction]
fn rerank_candidates_json(payload_json: &str) -> PyResult<String> {
    let payload: RerankPayload = serde_json::from_str(payload_json)
        .map_err(|e| PyValueError::new_err(format!("invalid rerank payload: {e}")))?;
    let results = mmr_select(payload);
    serde_json::to_string(&results)
        .map_err(|e| PyValueError::new_err(format!("serialize rerank result failed: {e}")))
}

#[pyfunction]
fn update_feedback_state_json(payload_json: &str) -> PyResult<String> {
    let payload: FeedbackPayload = serde_json::from_str(payload_json)
        .map_err(|e| PyValueError::new_err(format!("invalid feedback payload: {e}")))?;

    let cfg = payload.feedback_config.unwrap_or(FeedbackConfig {
        click_delta: default_click_delta(),
        bookmark_delta: default_bookmark_delta(),
        skip_delta: default_skip_delta(),
        half_life_days: default_half_life_days(),
    });

    let mut topic_weights = payload.user_state.topic_weights.clone();
    let mut topic_impressions = payload.user_state.topic_impressions.clone();
    let mut topic_clicks = payload.user_state.topic_clicks.clone();

    let delta = match payload.event_type.as_str() {
        "bookmark" => cfg.bookmark_delta,
        "click" => cfg.click_delta,
        "skip" => cfg.skip_delta,
        _ => 0.0,
    };

    let decay = if payload.user_state.last_updated_ts > 0.0 {
        exp_decay_factor(
            (payload.timestamp - payload.user_state.last_updated_ts).max(0.0),
            cfg.half_life_days,
        )
    } else {
        1.0
    };

    for cat in payload.categories {
        let current = topic_weights.get(&cat).copied().unwrap_or(0.0) * decay;
        topic_weights.insert(cat.clone(), current + delta);

        let imps = topic_impressions.get(&cat).copied().unwrap_or(0.0) * decay;
        topic_impressions.insert(cat.clone(), imps + 1.0);

        let clicks = topic_clicks.get(&cat).copied().unwrap_or(0.0) * decay;
        let click_increment = if payload.event_type == "click" || payload.event_type == "bookmark" {
            1.0
        } else {
            0.0
        };
        topic_clicks.insert(cat, clicks + click_increment);
    }

    let out = FeedbackResult {
        topic_weights,
        topic_impressions,
        topic_clicks,
        event_counter: payload.user_state.event_counter + 1,
        last_updated_ts: payload.timestamp,
    };

    serde_json::to_string(&out)
        .map_err(|e| PyValueError::new_err(format!("serialize feedback result failed: {e}")))
}

#[pymodule]
fn scholarevolve_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rerank_candidates_json, m)?)?;
    m.add_function(wrap_pyfunction!(update_feedback_state_json, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_decreases_with_time() {
        let short = exp_decay_factor(3600.0, 30.0);
        let long = exp_decay_factor(3600.0 * 24.0 * 60.0, 30.0);
        assert!(short > long);
    }
}
