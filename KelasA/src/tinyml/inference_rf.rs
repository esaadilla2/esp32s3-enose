//! Allocation-free Random Forest inference. Input features are raw features; no z-score needed.
use super::features::{FeatureVector, FEATURE_COUNT, FEATURE_SPEC};
use super::model_rf::{RandomForestModel, RfModelError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RfPrediction { pub class_index: usize, pub label: &'static str, pub score: f32 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RfPredictError { Model(RfModelError), NonFiniteInput, InvalidTree }

fn predict_tree(x: &FeatureVector, nodes: &[super::model_rf::RfNode], tree: super::model_rf::RfTree) -> Result<usize, RfPredictError> {
    let mut idx = tree.start as usize;
    // A valid binary tree with N nodes cannot require more than N traversals.
    for _ in 0..(tree.end - tree.start) {
        let n = nodes.get(idx).ok_or(RfPredictError::InvalidTree)?;
        if n.feature == -1 { return Ok(n.class as usize); }
        let value = *x.get(n.feature as usize).ok_or(RfPredictError::InvalidTree)?;
        idx = if value <= n.threshold { n.left as usize } else { n.right as usize };
        if idx < tree.start as usize || idx >= tree.end as usize { return Err(RfPredictError::InvalidTree); }
    }
    Err(RfPredictError::InvalidTree)
}

pub fn predict_rf(x: &FeatureVector, model: &RandomForestModel) -> Result<RfPrediction, RfPredictError> {
    model.validate(FEATURE_SPEC, FEATURE_COUNT).map_err(RfPredictError::Model)?;
    if !x.iter().all(|v| v.is_finite()) { return Err(RfPredictError::NonFiniteInput); }
    let mut votes = [0u16; super::model_rf::MAX_CLASSES_RF];
    for tree in model.trees {
        let class = predict_tree(x, model.nodes, *tree)?;
        let slot = votes.get_mut(class).ok_or(RfPredictError::InvalidTree)?;
        *slot = slot.saturating_add(1);
    }
    let mut best = 0usize;
    for i in 1..model.labels.len() { if votes[i] > votes[best] { best = i; } }
    Ok(RfPrediction { class_index: best, label: model.labels[best], score: votes[best] as f32 / model.trees.len() as f32 })
}
