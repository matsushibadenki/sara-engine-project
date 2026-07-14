#![allow(non_local_definitions)]

// Directory path: src/sara_engine/lib.rs
// English title: Rust Hybrid SNN Core
// Purpose: PyO3 extension for sparse, CPU-first SNN primitives used by SARA Engine.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::{HashMap, HashSet};
use rayon::prelude::*;

fn value_error(message: &str) -> PyErr {
    PyValueError::new_err(message.to_string())
}

fn validate_finite(name: &str, value: f32) -> PyResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(value_error(&format!("{name} must be finite")))
    }
}

fn validate_probability_like(name: &str, value: f32) -> PyResult<()> {
    validate_finite(name, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(value_error(&format!("{name} must be between 0.0 and 1.0")))
    }
}

// =====================================================================
// [1] Basic operations and fuzzy recall
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
    validate_finite("threshold", threshold)?;
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
// [2] SpikeEngine sparse propagation core
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
    pub fn new(decay_rate: f32) -> PyResult<Self> {
        validate_probability_like("decay_rate", decay_rate)?;
        Ok(SpikeEngine {
            weights: Vec::new(),
            potentials: HashMap::new(),
            decay_rate,
        })
    }

    pub fn set_weights(&mut self, weights: Vec<HashMap<usize, f32>>) { self.weights = weights; }
    pub fn get_weights(&self) -> Vec<HashMap<usize, f32>> { self.weights.clone() }
    pub fn reset_potentials(&mut self) { self.potentials.clear(); }

    pub fn propagate(&mut self, active_spikes: Vec<usize>, threshold: f32, max_spikes: usize) -> PyResult<Vec<usize>> {
        validate_finite("threshold", threshold)?;
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
        Ok(out_spikes)
    }

    pub fn apply_stdp(&mut self, pre_spikes: Vec<usize>, post_spikes: Vec<usize>, lr: f32) -> PyResult<()> {
        validate_finite("lr", lr)?;
        if lr < 0.0 {
            return Err(value_error("lr must be non-negative"));
        }
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
        Ok(())
    }

    pub fn normalize_weights(&mut self, max_weight: f32) -> PyResult<()> {
        validate_finite("max_weight", max_weight)?;
        if max_weight < 0.0 {
            return Err(value_error("max_weight must be non-negative"));
        }
        for targets in &mut self.weights {
            for w in targets.values_mut() {
                if *w > max_weight { *w = max_weight; }
            }
        }
        Ok(())
    }
}

// =====================================================================
// [3] Cortical columns / winner-take-all router with homeostasis
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
    pub fn new(input_dim: usize, num_experts: usize, top_k: usize) -> PyResult<Self> {
        if num_experts == 0 {
            return Err(value_error("num_experts must be positive"));
        }
        if top_k == 0 || top_k > num_experts {
            return Err(value_error("top_k must be between 1 and num_experts"));
        }
        let mut weights = Vec::with_capacity(input_dim);
        for _ in 0..input_dim { weights.push(HashMap::new()); }
        Ok(SpikeWTARouter { weights, num_experts, top_k, thresholds: vec![0.0; num_experts] })
    }

    pub fn set_weights(&mut self, weights: Vec<HashMap<usize, f32>>) { self.weights = weights; }
    pub fn get_weights(&self) -> Vec<HashMap<usize, f32>> { self.weights.clone() }
    pub fn get_thresholds(&self) -> Vec<f32> { self.thresholds.clone() }
    pub fn set_thresholds(&mut self, thresholds: Vec<f32>) -> PyResult<()> {
        if thresholds.len() != self.num_experts {
            return Err(value_error("thresholds length must equal num_experts"));
        }
        for value in &thresholds {
            validate_finite("threshold", *value)?;
        }
        self.thresholds = thresholds;
        Ok(())
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

    pub fn update_weights(&mut self, input_spikes: Vec<usize>, winners: Vec<usize>, lr: f32) -> PyResult<()> {
        validate_finite("lr", lr)?;
        if lr < 0.0 {
            return Err(value_error("lr must be non-negative"));
        }
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
        Ok(())
    }
    
    pub fn decay_weights(&mut self, decay_rate: f32) -> PyResult<()> {
        validate_probability_like("decay_rate", decay_rate)?;
        for targets in &mut self.weights {
            let mut to_remove = Vec::new();
            for (&exp_id, w) in targets.iter_mut() {
                *w *= decay_rate;
                if *w < 0.05 { to_remove.push(exp_id); }
            }
            for t in to_remove { targets.remove(&t); }
        }
        Ok(())
    }
}

// =====================================================================
// [4] LIF and predictive synapses
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
    pub fn new(decay_rate: f32, threshold: f32) -> PyResult<Self> {
        validate_probability_like("decay_rate", decay_rate)?;
        validate_finite("threshold", threshold)?;
        Ok(LIFNetwork { potentials: HashMap::new(), decay_rate, threshold })
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
    pub fn new(max_delay: usize) -> PyResult<Self> {
        let mut weights = Vec::with_capacity(max_delay + 1);
        for _ in 0..=max_delay { weights.push(HashMap::new()); }
        Ok(CausalSynapses { weights, max_delay })
    }

    pub fn train_step(&mut self, spike_history: Vec<Vec<usize>>, next_token: usize, learning_rate: f32) -> PyResult<()> {
        validate_finite("learning_rate", learning_rate)?;
        if learning_rate < 0.0 {
            return Err(value_error("learning_rate must be non-negative"));
        }
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
        Ok(())
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
// [5] Scalable SDR memory
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
    pub fn new(threshold: f32) -> PyResult<Self> {
        validate_probability_like("threshold", threshold)?;
        Ok(ScalableSDRMemory {
            records: Vec::new(),
            threshold,
        })
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
// [6] Direct synaptic wiring for one-shot corpus learning
// =====================================================================

/// Builds sparse delay-aware synapses from token IDs.
///
/// The weighting keeps the runtime sparse and CPU-first by using local
/// co-occurrence statistics instead of dense matrix training.
#[pyfunction]
fn build_direct_synapses(tokens: Vec<usize>, context_window: usize) -> PyResult<HashMap<usize, HashMap<usize, HashMap<usize, f32>>>> {
    if context_window == 0 {
        return Err(value_error("context_window must be positive"));
    }
    // delay -> pre_token -> post_token -> count
    let mut co_occurrence: HashMap<usize, HashMap<usize, HashMap<usize, f64>>> = HashMap::new();
    let mut unigram_counts: HashMap<usize, usize> = HashMap::new();
    
    let total_tokens = tokens.len();
    
    // First pass: count delay-specific co-occurrences inside the context window.
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
    
    // Second pass: normalize counts with a PMI-like sparse weighting scheme.
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
                        // Down-weight high-frequency tokens so common symbols do not dominate recall.
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
    pub fn new(input_dim: usize, trace_decay: f32) -> PyResult<Self> {
        validate_probability_like("trace_decay", trace_decay)?;
        let mut weights = Vec::with_capacity(input_dim);
        let mut eligibility_traces = Vec::with_capacity(input_dim);
        for _ in 0..input_dim {
            weights.push(HashMap::new());
            eligibility_traces.push(HashMap::new());
        }
        Ok(RewardModulatedSTDP { weights, eligibility_traces, trace_decay })
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

    pub fn apply_reward(&mut self, reward: f32, learning_rate: f32) -> PyResult<()> {
        validate_finite("reward", reward)?;
        validate_finite("learning_rate", learning_rate)?;
        if learning_rate < 0.0 {
            return Err(value_error("learning_rate must be non-negative"));
        }
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
        Ok(())
    }

    pub fn get_weights(&self) -> Vec<HashMap<usize, f32>> { self.weights.clone() }
    pub fn get_traces(&self) -> Vec<HashMap<usize, f32>> { self.eligibility_traces.clone() }
}

// =====================================================================
// [8] Batch processing for large-scale sparse encoding
// =====================================================================

/// Converts token batches into deterministic sparse distributed representations.
#[pyfunction]
fn batch_tokens_to_sdr(
    batch_tokens: Vec<Vec<usize>>,
    vocab_size: usize,
    sdr_density: f32,
    seed: u64,
) -> PyResult<Vec<Vec<Vec<usize>>>> {
    if vocab_size == 0 {
        return Err(value_error("vocab_size must be positive"));
    }
    validate_probability_like("sdr_density", sdr_density)?;
    if sdr_density <= 0.0 {
        return Err(value_error("sdr_density must be greater than 0.0"));
    }
    let sdr_size = (vocab_size as f32 * sdr_density).ceil() as usize;
    let sdr_size = sdr_size.max(1);

    // Rayon preserves collection order here; each token uses an independent seed.
    let batch_sdrs: Vec<Vec<Vec<usize>>> = batch_tokens.par_iter().map(|seq| {
        seq.par_iter().map(|token| {
            // The explicit LCG keeps Python and Rust fallback SDRs reproducible.
            let mut state = seed ^ (*token as u64);
            let mut sdr = Vec::with_capacity(sdr_size);
            for _ in 0..sdr_size {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                sdr.push(((state >> 32) as usize) % vocab_size);
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

/// Applies local homeostatic weight scaling to keep firing rates bounded.
#[pyfunction]
fn apply_homeostatic_scaling(
    mut weights: Vec<HashMap<usize, f32>>,
    firing_rates: Vec<f32>,
    target_rate: f32,
    learning_rate: f32,
) -> PyResult<Vec<HashMap<usize, f32>>> {
    validate_finite("target_rate", target_rate)?;
    validate_finite("learning_rate", learning_rate)?;
    if learning_rate < 0.0 {
        return Err(value_error("learning_rate must be non-negative"));
    }
    for rate in &firing_rates {
        validate_finite("firing_rate", *rate)?;
    }
    for (i, rate) in firing_rates.iter().enumerate() {
        if i < weights.len() {
            let error = target_rate - rate;
            let scaling_factor = 1.0 + (learning_rate * error);
            
            for val in weights[i].values_mut() {
                *val *= scaling_factor;
                if *val < 0.0 { *val = 0.0; }
                if *val > 5.0 { *val = 5.0; }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn calculate_sdr_overlap_uses_unique_sparse_indices() {
        assert_close(calculate_sdr_overlap(vec![1, 2, 2, 3], vec![2, 3, 4]).unwrap(), 2.0 / 3.0);
        assert_close(calculate_sdr_overlap(vec![], vec![1, 2]).unwrap(), 0.0);
    }

    #[test]
    fn spike_engine_propagates_decays_learns_and_resets() {
        let mut engine = SpikeEngine::new(0.5).unwrap();
        engine.set_weights(vec![HashMap::from([(1usize, 1.0f32), (2, 0.3)])]);

        assert_eq!(engine.propagate(vec![0], 0.8, 4).unwrap(), vec![1]);
        assert_eq!(engine.propagate(vec![], 0.4, 4).unwrap(), Vec::<usize>::new());

        engine.apply_stdp(vec![0], vec![2], 0.5).unwrap();
        let weights = engine.get_weights();
        assert!(weights[0][&2] > 0.3);
        assert!(weights[0][&1] < 1.0);

        engine.reset_potentials();
        assert_eq!(engine.propagate(vec![], 0.01, 4).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn wta_router_selects_top_k_adapts_thresholds_and_decays_weights() {
        let mut router = SpikeWTARouter::new(2, 3, 2).unwrap();
        router.set_weights(vec![
            HashMap::from([(0usize, 1.0f32), (1, 0.5)]),
            HashMap::from([(1usize, 0.7f32), (2, 0.2)]),
        ]);

        let winners = router.route(vec![0, 1], true);
        assert_eq!(winners, vec![1, 0]);
        let thresholds = router.get_thresholds();
        assert!(thresholds[1] > thresholds[2]);

        router.decay_weights(0.01).unwrap();
        assert!(router.get_weights()[0].is_empty());
    }

    #[test]
    fn lif_network_fires_at_threshold_and_resets() {
        let mut lif = LIFNetwork::new(0.5, 2.0).unwrap();
        assert_eq!(lif.forward(vec![7]), Vec::<usize>::new());
        assert_eq!(lif.forward(vec![7, 7]), vec![7]);
        lif.reset();
        assert_eq!(lif.forward(vec![7]), Vec::<usize>::new());
    }

    #[test]
    fn causal_synapses_learn_delay_aware_prediction_errors() {
        let mut synapses = CausalSynapses::new(2).unwrap();
        synapses.train_step(vec![vec![1], vec![2]], 9, 0.5).unwrap();

        let potentials = synapses.calculate_potentials(vec![vec![1], vec![2]]);
        assert!(potentials[&9] > 0.0);

        let (errors_before, rate_before) = synapses.predict_and_learn(vec![vec![3]], vec![8], 0.4, 0.1);
        assert_eq!(errors_before, vec![8]);
        assert_close(rate_before, 1.0);

        let (errors_after, rate_after) = synapses.predict_and_learn(vec![vec![3]], vec![8], 0.4, 0.1);
        assert!(errors_after.is_empty());
        assert_close(rate_after, 0.0);
    }

    #[test]
    fn scalable_sdr_memory_searches_top_k_and_handles_empty_query() {
        let mut memory = ScalableSDRMemory::new(0.34).unwrap();
        memory.add_memory(10, vec![1, 2, 3]);
        memory.add_memory(20, vec![1, 4, 5]);

        assert_eq!(memory.search(vec![], 2), Vec::<(usize, f32)>::new());
        let results = memory.search(vec![1, 2, 9], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 10);
        assert!(results[0].1 > 0.6);
    }

    #[test]
    fn reward_modulated_stdp_updates_traces_and_bounds_weights() {
        let mut learner = RewardModulatedSTDP::new(2, 0.5).unwrap();
        learner.update_trace(vec![0], vec![1]);
        assert!(learner.get_traces()[0][&1] > 0.0);

        learner.apply_reward(100.0, 1.0).unwrap();
        assert_close(learner.get_weights()[0][&1], 5.0);

        learner.apply_reward(-100.0, 1.0).unwrap();
        assert_close(learner.get_weights()[0][&1], 0.0);
    }

    #[test]
    fn direct_synapses_batch_sdr_and_homeostasis_stay_sparse_and_bounded() {
        let synapses = build_direct_synapses(vec![1, 2, 1, 3], 2).unwrap();
        assert!(synapses[&1][&1].contains_key(&2));
        assert!(synapses[&2][&2].contains_key(&3));

        let sdrs = batch_tokens_to_sdr(vec![vec![1, 2], vec![1]], 16, 0.125, 42).unwrap();
        assert_eq!(sdrs[0][0], sdrs[1][0]);
        assert!(sdrs.iter().flatten().flatten().all(|idx| *idx < 16));

        let scaled = apply_homeostatic_scaling(
            vec![HashMap::from([(1usize, 2.0f32)]), HashMap::from([(2usize, 2.0f32)])],
            vec![0.5, 2.0],
            1.0,
            0.5,
        )
        .unwrap();
        assert!(scaled[0][&1] > 2.0);
        assert!(scaled[1][&2] < 2.0);
        assert!(scaled.iter().flat_map(|m| m.values()).all(|w| (0.0..=5.0).contains(w)));
    }

    #[test]
    fn invalid_runtime_parameters_return_errors() {
        assert!(SpikeEngine::new(1.5).is_err());
        assert!(SpikeWTARouter::new(4, 0, 1).is_err());
        assert!(SpikeWTARouter::new(4, 2, 3).is_err());
        assert!(LIFNetwork::new(-0.1, 1.0).is_err());
        assert!(ScalableSDRMemory::new(1.5).is_err());
        assert!(RewardModulatedSTDP::new(2, 1.5).is_err());
        assert!(build_direct_synapses(vec![1, 2, 3], 0).is_err());
        assert!(batch_tokens_to_sdr(vec![vec![1]], 0, 0.1, 1).is_err());
        assert!(batch_tokens_to_sdr(vec![vec![1]], 16, 0.0, 1).is_err());
        assert!(apply_homeostatic_scaling(vec![HashMap::new()], vec![f32::NAN], 1.0, 0.1).is_err());
    }
}
