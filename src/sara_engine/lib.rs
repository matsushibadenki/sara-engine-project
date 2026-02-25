/*
ディレクトリパス: src/sara_engine/lib.rs
ファイルの日本語タイトル: SARA Rustコア - 高速スパイクエンジン (型互換性修正版)
ファイルの目的や内容: Python側の SpikeSelfAttention (List[Dict]) との型互換性を確保し、大規模スパイク伝播とSTDP学習を高速化する。
*/

use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass]
pub struct SpikeEngine {
    // Python側の List[Dict[int, float]] に合わせ、Vec<HashMap<u32, f32>> に変更
    pub weights: Vec<HashMap<u32, f32>>,
}

#[pymethods]
impl SpikeEngine {
    #[new]
    fn new() -> Self {
        SpikeEngine {
            weights: Vec::new(),
        }
    }

    /// 高速なスパイク伝播 (Pythonの _sparse_propagate を代替)
    fn propagate(&self, active_spikes: Vec<u32>, threshold: f32, max_out: usize) -> PyResult<Vec<u32>> {
        let mut potentials: HashMap<u32, f32> = HashMap::new();

        for &pre in &active_spikes {
            // インデックスが範囲内にあるか確認
            if let Some(targets) = self.weights.get(pre as usize) {
                for (&post, &w) in targets {
                    let count = potentials.entry(post).or_insert(0.0);
                    *count += w;
                }
            }
        }

        let mut active: Vec<(u32, f32)> = potentials
            .into_iter()
            .filter(|&(_, p)| p > threshold)
            .collect();

        active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let result = active.into_iter()
            .take(max_out)
            .map(|(id, _)| id)
            .collect();

        Ok(result)
    }

    /// 高速なSTDP学習
    fn apply_stdp(&mut self, pre_spikes: Vec<u32>, post_spikes: Vec<u32>, lr: f32) -> PyResult<()> {
        for &pre in &pre_spikes {
            let pre_idx = pre as usize;
            // 必要に応じてベクタを拡張
            if pre_idx >= self.weights.len() {
                self.weights.resize_with(pre_idx + 1, HashMap::new);
            }
            
            let targets = &mut self.weights[pre_idx];
            for &post in &post_spikes {
                let w = targets.entry(post).or_insert(0.2);
                *w = (*w + lr).min(3.0);
            }
        }
        Ok(())
    }

    /// 荷重の取得 (List[Dict[int, float]] 形式で返す)
    fn get_weights(&self) -> Vec<HashMap<u32, f32>> {
        self.weights.clone()
    }

    /// 荷重の設定 (List[Dict[int, float]] 形式を受け取る)
    fn set_weights(&mut self, weights: Vec<HashMap<u32, f32>>) {
        self.weights = weights;
    }
}

#[pymodule]
fn sara_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SpikeEngine>()?;
    Ok(())
}