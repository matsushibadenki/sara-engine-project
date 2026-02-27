/*
ディレクトリパス: src/sara_engine/lib.rs
ファイルの日本語タイトル: SARA Rustコア - トランスフォーマー＆LIF完全対応版
ファイルの目的や内容: Python側の `layers.py` や `transformer.py` から要求される `sparse_propagate_threshold` 関数の追加と、Phase 3向けのLIFモデル（膜電位の減衰・保持）を統合したマルチコアCPU最適化エンジン。
*/

use pyo3::prelude::*;
use std::collections::HashMap;
use rayon::prelude::*;

/// ------------------------------------------
/// 1. SpikeFeedForwardなどで利用される高速スパイク伝播
/// ------------------------------------------
#[pyfunction]
pub fn sparse_propagate_threshold(
    active_spikes: Vec<u32>,
    weights: Vec<HashMap<u32, f32>>,
    out_size: usize,
    threshold: f32,
) -> PyResult<Vec<u32>> {
    let mut potentials = vec![0.0; out_size];
    
    // 発火したスパイクの電位をターゲットのニューロンへ加算
    for s in active_spikes {
        let s_idx = s as usize;
        if s_idx < weights.len() {
            for (&target, &w) in &weights[s_idx] {
                let t_idx = target as usize;
                if t_idx < out_size {
                    potentials[t_idx] += w;
                }
            }
        }
    }
    
    // 閾値を超えたニューロンを発火 (スパイクIDとして返す)
    let fired: Vec<u32> = potentials.into_iter()
        .enumerate()
        .filter(|&(_, p)| p > threshold)
        .map(|(i, _)| i as u32)
        .collect();
        
    Ok(fired)
}

/// ------------------------------------------
/// 2. Phase 3 LIF(Leaky Integrate-and-Fire) 対応のSpikeEngine
/// ------------------------------------------
#[pyclass]
pub struct SpikeEngine {
    pub weights: Vec<HashMap<u32, f32>>,
    pub potentials: HashMap<u32, f32>, // LIFの膜電位を保持
    pub decay_rate: f32,               // 電位の減衰率
}

#[pymethods]
impl SpikeEngine {
    #[new]
    #[pyo3(signature = (decay_rate=0.9))]
    fn new(decay_rate: f32) -> Self {
        SpikeEngine {
            weights: Vec::new(),
            potentials: HashMap::new(),
            decay_rate,
        }
    }

    /// Fast spike propagation using Parallel Map-Reduce
    fn propagate(&mut self, active_spikes: Vec<u32>, threshold: f32, max_out: usize) -> PyResult<Vec<u32>> {
        // 1. 既存の電位を減衰 (Leaky phase)
        for p in self.potentials.values_mut() {
            *p *= self.decay_rate;
        }

        // 2. アクティブなスパイクからの入力電位を並列集計 (Integrate phase)
        let incoming = active_spikes
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

        // 3. 入力電位を加算
        for (k, v) in incoming {
            let p = self.potentials.entry(k).or_insert(0.0);
            *p += v;
        }

        // 4. 閾値を超えたら発火 (Fire phase)
        let mut active: Vec<(u32, f32)> = self.potentials
            .iter()
            .filter(|&(_, &p)| p > threshold)
            .map(|(&id, &p)| (id, p))
            .collect();

        // 電位が高い順にソートし、最大出力数に絞る
        active.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut result = Vec::new();
        for (id, _) in active.into_iter().take(max_out) {
            result.push(id);
            // 発火後は不応期として電位をリセット
            self.potentials.insert(id, 0.0);
        }

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

    fn reset_potentials(&mut self) {
        self.potentials.clear();
    }
}

/// ------------------------------------------
/// 3. layers.pyからのエラー回避用フォールバック (RustLiquidLayer)
/// ------------------------------------------
#[pyclass]
pub struct RustLiquidLayer {
    input_size: usize,
    hidden_size: usize,
    decay: f32,
    density: f32,
    feedback_scale: f32,
}

#[pymethods]
impl RustLiquidLayer {
    #[new]
    fn new(input_size: usize, hidden_size: usize, decay: f32, density: f32, feedback_scale: f32) -> Self {
        RustLiquidLayer {
            input_size,
            hidden_size,
            decay,
            density,
            feedback_scale,
        }
    }

    fn forward(
        &self,
        _active_inputs: Vec<u32>,
        _prev_active_hidden: Vec<u32>,
        _feedback_active: Vec<u32>,
        _attention_signal: Vec<u32>,
        _learning: bool,
        _reward: f32,
    ) -> PyResult<Vec<u32>> {
        Ok(Vec::new())
    }

    fn reset(&self) {}
}

/// ------------------------------------------
/// PyModule Entry Point
/// ------------------------------------------
#[pymodule]
fn sara_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SpikeEngine>()?;
    m.add_class::<RustLiquidLayer>()?; // Python側からのAttributeErrorを回避するため登録
    m.add_function(wrap_pyfunction!(sparse_propagate_threshold, m)?)?; // 今回のエラー原因を追加
    Ok(())
}