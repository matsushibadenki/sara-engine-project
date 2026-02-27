// ディレクトリパス: src/sara_engine/lib.rs
// ファイルの日本語タイトル: Rustハイブリッド SNNコア (復旧＋拡張版)
// ファイルの目的や内容: 誤って上書きしてしまった既存関数を復旧し、さらにPhase 3用のLIFとCausalSynapsesを統合した完全版。

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::{HashMap, HashSet};

// =====================================================================
// [1] 復旧部分: 既存のTransformer層などが依存していた関数
// =====================================================================

#[pyfunction]
fn sparse_propagate_threshold(
    active_spikes: Vec<usize>,
    weights: &PyAny, // 型の不一致エラーを回避するため、実行時に動的解釈する
    out_size: usize,
    threshold: f32,
) -> PyResult<Vec<usize>> {
    let mut potentials = vec![0.0; out_size];

    // weights が List であることを確認
    if let Ok(weights_list) = weights.downcast::<PyList>() {
        for &spike in &active_spikes {
            if spike < weights_list.len() {
                if let Ok(targets_obj) = weights_list.get_item(spike) {
                    // パターン1: weights が List[Dict[int, float]] の場合
                    if let Ok(targets_dict) = targets_obj.downcast::<PyDict>() {
                        for (k, v) in targets_dict.iter() {
                            if let (Ok(target_id), Ok(weight)) = (k.extract::<usize>(), v.extract::<f32>()) {
                                if target_id < out_size {
                                    potentials[target_id] += weight;
                                }
                            }
                        }
                    } 
                    // パターン2: weights が List[List[...]] の場合
                    else if let Ok(targets_list) = targets_obj.downcast::<PyList>() {
                        if targets_list.len() > 0 {
                            if let Ok(first_elem) = targets_list.get_item(0) {
                                // パターン2-a: Tupleのリスト List[List[Tuple[int, float]]]
                                if first_elem.is_instance_of::<PyTuple>() {
                                    for elem in targets_list.iter() {
                                        if let Ok(tuple) = elem.downcast::<PyTuple>() {
                                            if let (Ok(target_id), Ok(weight)) = (
                                                tuple.get_item(0).and_then(|x| x.extract::<usize>()),
                                                tuple.get_item(1).and_then(|x| x.extract::<f32>())
                                            ) {
                                                if target_id < out_size {
                                                    potentials[target_id] += weight;
                                                }
                                            }
                                        }
                                    }
                                } 
                                // パターン2-b: 単なる密行列のリスト List[List[float]]
                                else {
                                    for (target_id, elem) in targets_list.iter().enumerate() {
                                        if let Ok(weight) = elem.extract::<f32>() {
                                            if target_id < out_size {
                                                potentials[target_id] += weight;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err("weights must be a list"));
    }

    let mut fired = Vec::new();
    for (id, &pot) in potentials.iter().enumerate() {
        if pot >= threshold {
            fired.push(id);
        }
    }
    
    Ok(fired)
}


// =====================================================================
// [2] 新規部分: Phase 3 (v2.0.0) 用のLIFネットワークとSTDP学習コア
// =====================================================================

/// LIF (Leaky Integrate-and-Fire) ネットワークの状態を管理する構造体
#[pyclass]
pub struct LIFNetwork {
    potentials: HashMap<usize, f32>,
    decay_rate: f32,
    threshold: f32,
}

#[pymethods]
impl LIFNetwork {
    #[new]
    pub fn new(decay_rate: f32, threshold: f32) -> Self {
        LIFNetwork {
            potentials: HashMap::new(),
            decay_rate,
            threshold,
        }
    }

    pub fn reset(&mut self) {
        self.potentials.clear();
    }

    pub fn forward(&mut self, input_spikes: Vec<usize>) -> Vec<usize> {
        for val in self.potentials.values_mut() {
            *val *= self.decay_rate;
        }
        for &spike in input_spikes.iter() {
            *self.potentials.entry(spike).or_insert(0.0) += 1.0;
        }
        let mut fired = Vec::new();
        for (&neuron_id, val) in self.potentials.iter_mut() {
            if *val >= self.threshold {
                fired.push(neuron_id);
                *val = 0.0;
            }
        }
        fired
    }
}

/// SpikingCausalLM の重み学習と推論（電位計算）を担う構造体
#[pyclass]
pub struct CausalSynapses {
    weights: Vec<HashMap<usize, HashMap<usize, f32>>>,
    max_delay: usize,
}

#[pymethods]
impl CausalSynapses {
    #[new]
    pub fn new(max_delay: usize) -> Self {
        let mut weights = Vec::with_capacity(max_delay + 1);
        for _ in 0..=max_delay {
            weights.push(HashMap::new());
        }
        CausalSynapses {
            weights,
            max_delay,
        }
    }

    pub fn train_step(&mut self, spike_history: Vec<Vec<usize>>, next_token: usize, learning_rate: f32) {
        for (delay, active_spikes) in spike_history.iter().enumerate() {
            if delay > self.max_delay { break; }
            let eff_lr = learning_rate * (1.0 - (delay as f32) * 0.08);
            if eff_lr <= 0.0 { continue; }
            
            for &s in active_spikes.iter() {
                let targets = self.weights[delay].entry(s).or_insert_with(HashMap::new);
                for (t, w) in targets.iter_mut() {
                    if *t != next_token { *w *= 1.0 - eff_lr * 0.01; }
                }
                let old_w = *targets.get(&next_token).unwrap_or(&0.0);
                targets.insert(next_token, old_w + eff_lr * (1.0 - old_w));
                
                let total_w: f32 = targets.values().sum();
                if total_w > 5.0 {
                    let scale = 5.0 / total_w;
                    for w in targets.values_mut() { *w *= scale; }
                }
            }
        }
    }

    pub fn calculate_potentials(&self, spike_history: Vec<Vec<usize>>) -> HashMap<usize, f32> {
        let mut potentials: HashMap<usize, f32> = HashMap::new();
        let mut support_count: HashMap<usize, usize> = HashMap::new();

        for (delay, active_spikes) in spike_history.iter().enumerate() {
            if delay > self.max_delay { break; }
            let time_decay = (1.0 - (delay as f32) * 0.08).max(0.1);
            let mut supported = HashSet::new();

            for &s in active_spikes.iter() {
                if let Some(targets) = self.weights[delay].get(&s) {
                    for (&t_id, &weight) in targets.iter() {
                        *potentials.entry(t_id).or_insert(0.0) += weight * time_decay;
                        supported.insert(t_id);
                    }
                }
            }
            for t_id in supported {
                *support_count.entry(t_id).or_insert(0) += 1;
            }
        }
        for (t_id, pot) in potentials.iter_mut() {
            let count = *support_count.get(t_id).unwrap_or(&1) as f32;
            *pot *= count.powf(1.2);
        }
        potentials
    }
    
    pub fn get_token_fan_in(&self) -> HashMap<usize, f32> {
        let mut fan_in = HashMap::new();
        for delay_dict in &self.weights {
            for targets in delay_dict.values() {
                for (&t_id, &w) in targets.iter() {
                    *fan_in.entry(t_id).or_insert(0.0) += w;
                }
            }
        }
        fan_in
    }
}

// =====================================================================
// [3] モジュール登録
// =====================================================================

#[pymodule]
fn sara_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    // 復旧した関数を登録
    m.add_function(wrap_pyfunction!(sparse_propagate_threshold, m)?)?;
    
    // 新規のクラスを登録
    m.add_class::<LIFNetwork>()?;
    m.add_class::<CausalSynapses>()?;
    Ok(())
}