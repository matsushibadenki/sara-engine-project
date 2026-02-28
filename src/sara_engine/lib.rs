// ディレクトリパス: src/sara_engine/lib.rs
// ファイルの日本語タイトル: Rustハイブリッド SNNコア (ホメオスタシス対応版)
// ファイルの目的や内容: SpikeWTARouterに生物学的なSpike Frequency Adaptation(発火頻度適応)を導入し、特定のエキスパートが一人勝ちするのを防ぐ。

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::{HashMap, HashSet};

// =====================================================================
// [1] SpikeEngine & Sparse Propagation (Transformer / Attention Core)
// =====================================================================

#[pyfunction]
fn sparse_propagate_threshold(
    active_spikes: Vec<usize>,
    weights: &PyAny,
    out_size: usize,
    threshold: f32,
) -> PyResult<Vec<usize>> {
    let mut potentials = vec![0.0; out_size];

    if let Ok(weights_list) = weights.downcast::<PyList>() {
        for &spike in &active_spikes {
            if spike < weights_list.len() {
                if let Ok(targets_obj) = weights_list.get_item(spike) {
                    if let Ok(targets_dict) = targets_obj.downcast::<PyDict>() {
                        for (k, v) in targets_dict.iter() {
                            if let (Ok(target_id), Ok(weight)) = (k.extract::<usize>(), v.extract::<f32>()) {
                                if target_id < out_size { potentials[target_id] += weight; }
                            }
                        }
                    } else if let Ok(targets_list) = targets_obj.downcast::<PyList>() {
                        if targets_list.len() > 0 {
                            if let Ok(first_elem) = targets_list.get_item(0) {
                                if first_elem.is_instance_of::<PyTuple>() {
                                    for elem in targets_list.iter() {
                                        if let Ok(tuple) = elem.downcast::<PyTuple>() {
                                            if let (Ok(target_id), Ok(weight)) = (
                                                tuple.get_item(0).and_then(|x| x.extract::<usize>()),
                                                tuple.get_item(1).and_then(|x| x.extract::<f32>())
                                            ) {
                                                if target_id < out_size { potentials[target_id] += weight; }
                                            }
                                        }
                                    }
                                } else {
                                    for (target_id, elem) in targets_list.iter().enumerate() {
                                        if let Ok(weight) = elem.extract::<f32>() {
                                            if target_id < out_size { potentials[target_id] += weight; }
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
        if pot >= threshold { fired.push(id); }
    }
    Ok(fired)
}

#[pyclass]
pub struct SpikeEngine {
    weights: Vec<HashMap<usize, f32>>,
    potentials: HashMap<usize, f32>,
    decay_rate: f32,
}

#[pymethods]
impl SpikeEngine {
    #[new]
    #[pyo3(signature = (decay_rate=0.9))]
    pub fn new(decay_rate: f32) -> Self {
        SpikeEngine {
            weights: Vec::new(),
            potentials: HashMap::new(),
            decay_rate,
        }
    }

    pub fn set_weights(&mut self, weights: Vec<HashMap<usize, f32>>) {
        self.weights = weights;
    }

    pub fn get_weights(&self) -> Vec<HashMap<usize, f32>> {
        self.weights.clone()
    }

    pub fn reset_potentials(&mut self) {
        self.potentials.clear();
    }

    pub fn propagate(&mut self, active_spikes: Vec<usize>, threshold: f32, max_spikes: usize) -> Vec<usize> {
        for val in self.potentials.values_mut() { *val *= self.decay_rate; }
        
        for &spike in &active_spikes {
            if spike < self.weights.len() {
                for (&target, &w) in &self.weights[spike] {
                    *self.potentials.entry(target).or_insert(0.0) += w;
                }
            }
        }
        
        let mut fired: Vec<(usize, f32)> = self.potentials.iter()
            .filter(|&(_, &pot)| pot >= threshold)
            .map(|(&target, &pot)| (target, pot))
            .collect();
            
        fired.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut out_spikes = Vec::new();
        for (i, (target, _)) in fired.into_iter().enumerate() {
            if i >= max_spikes { break; }
            out_spikes.push(target);
            self.potentials.insert(target, 0.0);
        }
        out_spikes
    }

    pub fn apply_stdp(&mut self, pre_spikes: Vec<usize>, post_spikes: Vec<usize>, lr: f32) {
        let post_set: HashSet<usize> = post_spikes.into_iter().collect();
        for &pre in &pre_spikes {
            if pre < self.weights.len() {
                let targets = &mut self.weights[pre];
                let mut to_remove = Vec::new();
                for (&target, w) in targets.iter_mut() {
                    if post_set.contains(&target) {
                        *w = (*w + lr).min(3.0);
                    } else {
                        *w = (*w - lr * 0.05).max(0.0);
                        if *w < 0.01 { to_remove.push(target); }
                    }
                }
                for t in to_remove { targets.remove(&t); }
                for &post in &post_set {
                    if !targets.contains_key(&post) { targets.insert(post, 0.2); }
                }
            }
        }
    }

    pub fn normalize_weights(&mut self, max_weight: f32) {
        for targets in &mut self.weights {
            for w in targets.values_mut() {
                if *w > max_weight { *w = max_weight; }
            }
        }
    }
}

// =====================================================================
// [2] Cortical Columns (MoE) / Winner-Take-All Router with Homeostasis
// =====================================================================

#[pyclass]
pub struct SpikeWTARouter {
    weights: Vec<HashMap<usize, f32>>,
    num_experts: usize,
    top_k: usize,
    thresholds: Vec<f32>, // ホメオスタシス用の疲労度（閾値）
}

#[pymethods]
impl SpikeWTARouter {
    #[new]
    pub fn new(input_dim: usize, num_experts: usize, top_k: usize) -> Self {
        let mut weights = Vec::with_capacity(input_dim);
        for _ in 0..input_dim { weights.push(HashMap::new()); }
        SpikeWTARouter { 
            weights, 
            num_experts, 
            top_k,
            thresholds: vec![0.0; num_experts]
        }
    }

    pub fn set_weights(&mut self, weights: Vec<HashMap<usize, f32>>) {
        self.weights = weights;
    }

    pub fn get_weights(&self) -> Vec<HashMap<usize, f32>> {
        self.weights.clone()
    }
    
    pub fn get_thresholds(&self) -> Vec<f32> {
        self.thresholds.clone()
    }
    
    pub fn set_thresholds(&mut self, thresholds: Vec<f32>) {
        self.thresholds = thresholds;
    }

    pub fn route(&mut self, input_spikes: Vec<usize>, learning: bool) -> Vec<usize> {
        let mut potentials = vec![0.0; self.num_experts];
        for &spike in &input_spikes {
            if spike < self.weights.len() {
                for (&exp_id, &w) in &self.weights[spike] {
                    if exp_id < self.num_experts { potentials[exp_id] += w; }
                }
            }
        }
        
        // 疲労度（閾値）を引いてポテンシャルを調整
        let mut adjusted_potentials = potentials.clone();
        for i in 0..self.num_experts {
            adjusted_potentials[i] -= self.thresholds[i];
        }

        let mut sorted: Vec<(usize, f32)> = adjusted_potentials.into_iter().enumerate().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut winners = Vec::new();
        for (i, (exp_id, _pot)) in sorted.into_iter().enumerate() {
            if i >= self.top_k { break; }
            winners.push(exp_id);
        }

        // 学習時は勝者の閾値を上げ（疲れさせ）、全体を減衰させる
        if learning {
            for i in 0..self.num_experts {
                self.thresholds[i] *= 0.95; // 疲労の回復（減衰）
            }
            for &w_id in &winners {
                self.thresholds[w_id] += 2.0; // 勝つと疲労が溜まる
            }
        }

        winners
    }

    pub fn update_weights(&mut self, input_spikes: Vec<usize>, winners: Vec<usize>, lr: f32) {
        let winner_set: HashSet<usize> = winners.into_iter().collect();
        for &spike in &input_spikes {
            if spike < self.weights.len() {
                let targets = &mut self.weights[spike];
                let mut to_remove = Vec::new();
                for (&exp_id, w) in targets.iter_mut() {
                    if winner_set.contains(&exp_id) {
                        *w = (*w + lr).min(3.0);
                    } else {
                        *w = (*w - lr * 0.1).max(0.0);
                        if *w < 0.05 { to_remove.push(exp_id); }
                    }
                }
                for t in to_remove { targets.remove(&t); }
                for &exp_id in &winner_set {
                    if !targets.contains_key(&exp_id) { targets.insert(exp_id, 0.1); }
                }
            }
        }
    }
    
    pub fn decay_weights(&mut self, decay_rate: f32) {
        for targets in &mut self.weights {
            let mut to_remove = Vec::new();
            for (&exp_id, w) in targets.iter_mut() {
                *w *= decay_rate;
                if *w < 0.05 { to_remove.push(exp_id); }
            }
            for t in to_remove { targets.remove(&t); }
        }
    }
}

// =====================================================================
// [3] Phase 3 Core
// =====================================================================

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
        LIFNetwork { potentials: HashMap::new(), decay_rate, threshold }
    }
    pub fn reset(&mut self) { self.potentials.clear(); }
    pub fn forward(&mut self, input_spikes: Vec<usize>) -> Vec<usize> {
        for val in self.potentials.values_mut() { *val *= self.decay_rate; }
        for &spike in input_spikes.iter() { *self.potentials.entry(spike).or_insert(0.0) += 1.0; }
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
        for _ in 0..=max_delay { weights.push(HashMap::new()); }
        CausalSynapses { weights, max_delay }
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
            }
        }
    }
    pub fn calculate_potentials(&self, spike_history: Vec<Vec<usize>>) -> HashMap<usize, f32> {
        let mut potentials: HashMap<usize, f32> = HashMap::new();
        for (delay, active_spikes) in spike_history.iter().enumerate() {
            if delay > self.max_delay { break; }
            let time_decay = (1.0 - (delay as f32) * 0.08).max(0.1);
            for &s in active_spikes.iter() {
                if let Some(targets) = self.weights[delay].get(&s) {
                    for (&t_id, &weight) in targets.iter() {
                        *potentials.entry(t_id).or_insert(0.0) += weight * time_decay;
                    }
                }
            }
        }
        potentials
    }
    /// 各トークンIDへの受容結合重みの合計（ファンイン）を返す。
    /// ハブトークンの影響を正規化するために使用する。
    pub fn get_token_fan_in(&self) -> HashMap<usize, f32> {
        let mut fan_in: HashMap<usize, f32> = HashMap::new();
        for delay_weights in &self.weights {
            for (_src, targets) in delay_weights.iter() {
                for (&t_id, &weight) in targets.iter() {
                    *fan_in.entry(t_id).or_insert(0.0) += weight;
                }
            }
        }
        fan_in
    }
}

#[pymodule]
fn sara_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sparse_propagate_threshold, m)?)?;
    m.add_class::<SpikeEngine>()?;
    m.add_class::<SpikeWTARouter>()?;
    m.add_class::<LIFNetwork>()?;
    m.add_class::<CausalSynapses>()?;
    Ok(())
}