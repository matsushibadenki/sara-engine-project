// ディレクトリパス: src/sara_engine/lib.rs
// ファイルの英語タイトル: Rust Hybrid SNN Core (Complete Version)
// ファイルの目的や内容: SARA Engineのコアとなるスパイクニューラルネットワークの演算を高速化するためのRust拡張モジュール。フェーズ2の予測符号化、WTAルーター、スケーラブルなSDRメモリから、フェーズ3のコーパス直接シナプス結線（Direct Synaptic Wiring）や能動的推論に向けた報酬修飾型STDP（R-STDP）、さらにフェーズ4に向けたバッチSDR生成やホメオスタシス機能まで、すべての処理を統合した完全版。

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::{HashMap, HashSet};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

// =====================================================================
// [1] 基本演算 & Fuzzy Recall (Phase 2)
// =====================================================================

#[pyfunction]
fn calculate_sdr_overlap(sdr_a: Vec<usize>, sdr_b: Vec<usize>) -> PyResult<f32> {
    let set_a: HashSet<_> = sdr_a.into_iter().collect();
    let set_b: HashSet<_> = sdr_b.into_iter().collect();
    let intersect = set_a.intersection(&set_b).count();
    if set_a.is_empty() || set_b.is_empty() { return Ok(0.0); }
    Ok(intersect as f32 / (set_a.len().max(set_b.len()) as f32))
}

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

// =====================================================================
// [2] SpikeEngine (Transformer / Attention Core)
// =====================================================================

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

    pub fn set_weights(&mut self, weights: Vec<HashMap<usize, f32>>) { self.weights = weights; }
    pub fn get_weights(&self) -> Vec<HashMap<usize, f32>> { self.weights.clone() }
    pub fn reset_potentials(&mut self) { self.potentials.clear(); }

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
// [3] Cortical Columns (MoE) / Winner-Take-All Router with Homeostasis
// =====================================================================

#[pyclass]
pub struct SpikeWTARouter {
    weights: Vec<HashMap<usize, f32>>,
    num_experts: usize,
    top_k: usize,
    thresholds: Vec<f32>,
}

#[pymethods]
impl SpikeWTARouter {
    #[new]
    pub fn new(input_dim: usize, num_experts: usize, top_k: usize) -> Self {
        let mut weights = Vec::with_capacity(input_dim);
        for _ in 0..input_dim { weights.push(HashMap::new()); }
        SpikeWTARouter { weights, num_experts, top_k, thresholds: vec![0.0; num_experts] }
    }

    pub fn set_weights(&mut self, weights: Vec<HashMap<usize, f32>>) { self.weights = weights; }
    pub fn get_weights(&self) -> Vec<HashMap<usize, f32>> { self.weights.clone() }
    pub fn get_thresholds(&self) -> Vec<f32> { self.thresholds.clone() }
    pub fn set_thresholds(&mut self, thresholds: Vec<f32>) { self.thresholds = thresholds; }

    pub fn route(&mut self, input_spikes: Vec<usize>, learning: bool) -> Vec<usize> {
        let mut potentials = vec![0.0; self.num_experts];
        for &spike in &input_spikes {
            if spike < self.weights.len() {
                for (&exp_id, &w) in &self.weights[spike] {
                    if exp_id < self.num_experts { potentials[exp_id] += w; }
                }
            }
        }
        
        let mut adjusted_potentials = potentials.clone();
        for i in 0..self.num_experts { adjusted_potentials[i] -= self.thresholds[i]; }

        let mut sorted: Vec<(usize, f32)> = adjusted_potentials.into_iter().enumerate().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut winners = Vec::new();
        for (i, (exp_id, _pot)) in sorted.into_iter().enumerate() {
            if i >= self.top_k { break; }
            winners.push(exp_id);
        }

        if learning {
            for i in 0..self.num_experts { self.thresholds[i] *= 0.95; }
            for &w_id in &winners { self.thresholds[w_id] += 2.0; }
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
                    if winner_set.contains(&exp_id) { *w = (*w + lr).min(3.0); }
                    else { *w = (*w - lr * 0.1).max(0.0); if *w < 0.05 { to_remove.push(exp_id); } }
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
// [4] LIF & Predictive Synapses
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

    pub fn predict_and_learn(&mut self, spike_history: Vec<Vec<usize>>, actual_next_spikes: Vec<usize>, learning_rate: f32, threshold: f32) -> (Vec<usize>, f32) {
        let potentials = self.calculate_potentials(spike_history.clone());
        let mut predicted_set: HashSet<usize> = HashSet::new();
        for (&target, &pot) in potentials.iter() {
            if pot >= threshold { predicted_set.insert(target); }
        }
        
        let actual_set: HashSet<usize> = actual_next_spikes.into_iter().collect();
        let error_spikes: Vec<usize> = actual_set.difference(&predicted_set).cloned().collect();
        
        let error_rate = if actual_set.is_empty() { 
            0.0 
        } else { 
            error_spikes.len() as f32 / actual_set.len() as f32 
        };

        if !error_spikes.is_empty() {
            for (delay, active_spikes) in spike_history.iter().enumerate() {
                if delay > self.max_delay { break; }
                let eff_lr = learning_rate * (1.0 - (delay as f32) * 0.08);
                if eff_lr <= 0.0 { continue; }
                
                for &s in active_spikes.iter() {
                    let targets = self.weights[delay].entry(s).or_insert_with(HashMap::new);
                    for &err_spike in &error_spikes {
                        let old_w = *targets.get(&err_spike).unwrap_or(&0.0);
                        targets.insert(err_spike, old_w + eff_lr * (1.0 - old_w));
                    }
                }
            }
        }
        
        (error_spikes, error_rate)
    }
}

// =====================================================================
// [5] Scalable SDR Memory (Phase 3: Million-token LTM)
// =====================================================================

#[pyclass]
pub struct ScalableSDRMemory {
    records: Vec<(usize, HashSet<usize>)>, // (memory_id, sdr_set)
    threshold: f32,
}

#[pymethods]
impl ScalableSDRMemory {
    #[new]
    #[pyo3(signature = (threshold=0.1))]
    pub fn new(threshold: f32) -> Self {
        ScalableSDRMemory {
            records: Vec::new(),
            threshold,
        }
    }

    pub fn add_memory(&mut self, mem_id: usize, sdr: Vec<usize>) {
        let set: HashSet<usize> = sdr.into_iter().collect();
        self.records.push((mem_id, set));
    }

    pub fn search(&self, query_sdr: Vec<usize>, top_k: usize) -> Vec<(usize, f32)> {
        let query_set: HashSet<usize> = query_sdr.into_iter().collect();
        let query_len = query_set.len() as f32;
        if query_len == 0.0 { return Vec::new(); }

        let mut results = Vec::new();
        for (id, mem_set) in &self.records {
            let overlap = query_set.intersection(mem_set).count() as f32;
            let score = overlap / query_len;
            if score >= self.threshold {
                results.push((*id, score));
            }
        }

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().take(top_k).collect()
    }
    
    pub fn clear(&mut self) { self.records.clear(); }
    pub fn memory_count(&self) -> usize { self.records.len() }
}

// =====================================================================
// [6] Direct Synaptic Wiring (One-Shot Corpus Learning)
// =====================================================================

/// テキストコーパスから抽出された文字IDリスト（tokens）を走査し、
/// 遅延時間（Polychronization）とPMI（Pointwise Mutual Information）による重み正規化を用いて、
/// 高速にシナプス結線を構築します。
#[pyfunction]
fn build_direct_synapses(tokens: Vec<usize>, context_window: usize) -> PyResult<HashMap<usize, HashMap<usize, HashMap<usize, f32>>>> {
    // delay -> pre_token -> post_token -> count
    let mut co_occurrence: HashMap<usize, HashMap<usize, HashMap<usize, f64>>> = HashMap::new();
    let mut unigram_counts: HashMap<usize, usize> = HashMap::new();
    
    let total_tokens = tokens.len();
    
    // 1パス目: ウィンドウ内の遅延距離ごとの共起カウント
    for i in 0..total_tokens {
        let current = tokens[i];
        *unigram_counts.entry(current).or_insert(0) += 1;
        
        let end_idx = std::cmp::min(i + context_window + 1, total_tokens);
        for j in (i + 1)..end_idx {
            let delay = j - i;
            let next_token = tokens[j];
            
            let delay_map = co_occurrence.entry(delay).or_insert_with(HashMap::new);
            let targets = delay_map.entry(current).or_insert_with(HashMap::new);
            *targets.entry(next_token).or_insert(0.0) += 1.0;
        }
    }
    
    // 2パス目: カウントを確率的重みに正規化（PMI的アプローチ）
    let mut synapses: HashMap<usize, HashMap<usize, HashMap<usize, f32>>> = HashMap::new();
    for (delay, pre_dict) in co_occurrence.iter() {
        let mut delay_synapses = HashMap::new();
        for (pre, posts) in pre_dict.iter() {
            if let Some(&pre_count) = unigram_counts.get(pre) {
                let pre_count_f64 = pre_count as f64;
                let mut target_map = HashMap::new();
                
                for (post, count) in posts.iter() {
                    if let Some(&post_count) = unigram_counts.get(post) {
                        let post_count_f64 = post_count as f64;
                        // 無限ループを防ぐため、出現頻度の高い文字（空白など）への偏りを補正
                        let weight = count / (pre_count_f64 * post_count_f64).sqrt();
                        target_map.insert(*post, weight as f32);
                    }
                }
                delay_synapses.insert(*pre, target_map);
            }
        }
        synapses.insert(*delay, delay_synapses);
    }
    
    Ok(synapses)
}

// =====================================================================
// [7] Reward-Modulated STDP for Active Inference (Phase 3 Step 4)
// =====================================================================

#[pyclass]
pub struct RewardModulatedSTDP {
    weights: Vec<HashMap<usize, f32>>,
    eligibility_traces: Vec<HashMap<usize, f32>>,
    trace_decay: f32,
}

#[pymethods]
impl RewardModulatedSTDP {
    #[new]
    pub fn new(input_dim: usize, trace_decay: f32) -> Self {
        let mut weights = Vec::with_capacity(input_dim);
        let mut eligibility_traces = Vec::with_capacity(input_dim);
        for _ in 0..input_dim {
            weights.push(HashMap::new());
            eligibility_traces.push(HashMap::new());
        }
        RewardModulatedSTDP { weights, eligibility_traces, trace_decay }
    }

    pub fn update_trace(&mut self, pre_spikes: Vec<usize>, post_spikes: Vec<usize>) {
        let post_set: HashSet<usize> = post_spikes.into_iter().collect();
        for &pre in &pre_spikes {
            if pre < self.eligibility_traces.len() {
                let traces = &mut self.eligibility_traces[pre];
                for &post in &post_set {
                    *traces.entry(post).or_insert(0.0) += 1.0;
                }
            }
        }
        
        // Decay traces globally
        for traces in &mut self.eligibility_traces {
            let mut to_remove = Vec::new();
            for (&target, trace) in traces.iter_mut() {
                *trace *= self.trace_decay;
                if *trace < 0.01 {
                    to_remove.push(target);
                }
            }
            for t in to_remove {
                traces.remove(&t);
            }
        }
    }

    pub fn apply_reward(&mut self, reward: f32, learning_rate: f32) {
        for i in 0..self.weights.len() {
            let traces = &self.eligibility_traces[i];
            let w_map = &mut self.weights[i];
            for (&target, &trace) in traces.iter() {
                let w = w_map.entry(target).or_insert(0.1);
                *w += learning_rate * reward * trace;
                if *w < 0.0 { *w = 0.0; }
                if *w > 5.0 { *w = 5.0; }
            }
        }
    }
}

// =====================================================================
// [8] Batch Processing for Large Scale Training (Phase 4 Prep)
// =====================================================================

/// トークンのバッチを受け取り、それぞれをSDR（Sparse Distributed Representation）に変換します。
/// 大規模なコーパスの事前学習を高速化するためのユーティリティです。
#[pyfunction]
fn batch_tokens_to_sdr(
    batch_tokens: Vec<Vec<usize>>,
    vocab_size: usize,
    sdr_density: f32,
    seed: u64,
) -> PyResult<Vec<Vec<Vec<usize>>>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let sdr_size = (vocab_size as f32 * sdr_density).ceil() as usize;
    let sdr_size = sdr_size.max(1);

    // TODO: 将来的にRayonを導入して par_iter() で並列化可能
    let batch_sdrs: Vec<Vec<Vec<usize>>> = batch_tokens.into_iter().map(|seq| {
        seq.into_iter().map(|token| {
            // 簡易的なハッシュベースの擬似ランダムSDR生成
            // 実際の運用ではより洗練されたエンコーダを使用する
            let mut seq_rng = StdRng::seed_from_u64(seed ^ (token as u64));
            let mut sdr = Vec::with_capacity(sdr_size);
            for _ in 0..sdr_size {
                sdr.push(seq_rng.gen_range(0..vocab_size));
            }
            sdr.sort_unstable();
            sdr.dedup();
            sdr
        }).collect()
    }).collect();

    Ok(batch_sdrs)
}

// =====================================================================
// [9] Homeostatic Scaling
// =====================================================================

/// 大規模ネットワークでの発火率の爆発/減衰を防ぐためのホメオスタシス機能
#[pyfunction]
fn apply_homeostatic_scaling(
    mut weights: Vec<HashMap<usize, f32>>,
    firing_rates: Vec<f32>,
    target_rate: f32,
    learning_rate: f32,
) -> PyResult<Vec<HashMap<usize, f32>>> {
    for (i, rate) in firing_rates.iter().enumerate() {
        if i < weights.len() {
            let error = target_rate - rate;
            // 発火率が目標より高ければ重みを下げ、低ければ上げる
            let scaling_factor = 1.0 + (learning_rate * error);
            
            for val in weights[i].values_mut() {
                *val *= scaling_factor;
                if *val < 0.0 { *val = 0.0; }
                if *val > 5.0 { *val = 5.0; } // W_max
            }
        }
    }
    Ok(weights)
}

#[pymodule]
fn sara_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_sdr_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_propagate_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(build_direct_synapses, m)?)?;
    m.add_function(wrap_pyfunction!(batch_tokens_to_sdr, m)?)?;
    m.add_function(wrap_pyfunction!(apply_homeostatic_scaling, m)?)?;
    m.add_class::<SpikeEngine>()?;
    m.add_class::<SpikeWTARouter>()?;
    m.add_class::<LIFNetwork>()?;
    m.add_class::<CausalSynapses>()?;
    m.add_class::<ScalableSDRMemory>()?;
    m.add_class::<RewardModulatedSTDP>()?;
    Ok(())
}