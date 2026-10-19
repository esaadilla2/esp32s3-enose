//! Runtime data structures for exported Random Forest models (no_std, no allocation).

pub const MAX_CLASSES_RF: usize = 16;
pub const MAX_LABEL_LEN_RF: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RfModelKind { Trained, SyntheticFixture }

impl RfModelKind {
    pub const fn as_str(self) -> &'static str {
        match self { Self::Trained => "trained", Self::SyntheticFixture => "synthetic_fixture" }
    }
}

/// A decision-tree node. `feature == -1` means leaf; `class` is used only at leaves.
#[derive(Debug, Clone, Copy)]
pub struct RfNode {
    pub feature: i16,
    pub threshold: f32,
    pub left: u16,
    pub right: u16,
    pub class: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct RfTree { pub start: u16, pub end: u16 }

pub struct RandomForestModel {
    pub kind: RfModelKind,
    pub version: &'static str,
    pub feature_spec: &'static str,
    pub labels: &'static [&'static str],
    pub nodes: &'static [RfNode],
    pub trees: &'static [RfTree],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RfModelError { SpecMismatch, BadClassCount, EmptyForest, BadTreeRange, BadNode, BadLabel }

fn label_ok(s: &str) -> bool {
    !s.is_empty() && s.len() <= MAX_LABEL_LEN_RF && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

impl RandomForestModel {
    pub fn validate(
        &self,
        feature_spec: &str,
        feature_count: usize,
    ) -> Result<(), RfModelError> {
        if self.feature_spec != feature_spec {
            return Err(RfModelError::SpecMismatch);
        }

        if !(2..=MAX_CLASSES_RF).contains(&self.labels.len()) {
            return Err(RfModelError::BadClassCount);
        }

        if self.trees.is_empty() || self.nodes.is_empty() {
            return Err(RfModelError::EmptyForest);
        }

        if !self.labels.iter().all(|label| label_ok(label)) {
            return Err(RfModelError::BadLabel);
        }

        for tree in self.trees {
            if tree.start >= tree.end
                || (tree.end as usize) > self.nodes.len()
            {
                return Err(RfModelError::BadTreeRange);
            }

            for node in &self.nodes[tree.start as usize..tree.end as usize] {
                if node.feature == -1 {
                    if node.class as usize >= self.labels.len() {
                        return Err(RfModelError::BadNode);
                    }
                } else if node.feature < 0
                    || (node.feature as usize) >= feature_count
                    || (node.left as usize) < (tree.start as usize)
                    || (node.left as usize) >= (tree.end as usize)
                    || (node.right as usize) < (tree.start as usize)
                    || (node.right as usize) >= (tree.end as usize)
                    || !node.threshold.is_finite()
                {
                    return Err(RfModelError::BadNode);
                }
            }
        }

        Ok(())
    }
}
