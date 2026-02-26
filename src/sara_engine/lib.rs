/*
ディレクトリパス: src/sara_engine/lib.rs
ファイルの日本語タイトル: SARA Rustコア - 分散並列スパイク処理版
ファイルの目的や内容: Phase 3に向けたマルチコアCPU極限最適化。Rayonを用いた並列マップリデュースにより、行列演算・GPUなしで大規模なスパイク伝播と重み正規化を高速化する。
*/

use pyo3::prelude::*;
use std::collections::HashMap;
use rayon::prelude::*;

#[pyclass]
pub struct SpikeEngine {
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

    /// Fast spike propagation using Parallel Map-Reduce (Multi-core CPU optimized)
    fn propagate(&self, active_spikes: Vec<u32>, threshold: f32, max_out: usize) -> PyResult<Vec<u32>> {
        // 並列処理によるスパイク電位の集計（ゼロ・コピー＆ロックフリー設計）
        let potentials = active_spikes
            .par_iter()
            .filter_map(|&pre| self.weights.get(pre as usize))
            .fold(
                || HashMap::new(),
                |mut acc: HashMap<u32, f32>, targets| {
                    for (&post, &w) in targets {
                        let count = acc.entry(post).or_insert(0.0);
                        *count += w;
                    }
                    acc
                }
            )
            .reduce(
                || HashMap::new(),
                |mut a, b| {
                    for (k, v) in b {
                        let count = a.entry(k).or_insert(0.0);
                        *count += v;
                    }
                    a
                }
            );

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

    /// Fast STDP learning
    fn apply_stdp(&mut self, pre_spikes: Vec<u32>, post_spikes: Vec<u32>, lr: f32) -> PyResult<()> {
        let max_pre = pre_spikes.iter().cloned().max().unwrap_or(0) as usize;
        if max_pre >= self.weights.len() {
            self.weights.resize_with(max_pre + 1, HashMap::new);
        }
        
        for &pre in &pre_spikes {
            let targets = &mut self.weights[pre as usize];
            for &post in &post_spikes {
                let w = targets.entry(post).or_insert(0.2);
                *w = (*w + lr).min(3.0);
            }
        }
        Ok(())
    }

    /// Parallel Synaptic scaling (Homeostatic Normalization)
    fn normalize_weights(&mut self, target_sum: f32) -> PyResult<()> {
        self.weights.par_iter_mut().for_each(|targets| {
            let sum: f32 = targets.values().sum();
            if sum > 0.0 {
                let scale = target_sum / sum;
                for w in targets.values_mut() {
                    *w *= scale;
                }
            }
        });
        Ok(())
    }

    fn get_weights(&self) -> Vec<HashMap<u32, f32>> {
        self.weights.clone()
    }

    fn set_weights(&mut self, weights: Vec<HashMap<u32, f32>>) {
        self.weights = weights;
    }
}

#[pymodule]
fn sara_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SpikeEngine>()?;
    Ok(())
}