use pyo3::prelude::*;
use rand::prelude::*;
use std::collections::{HashSet, VecDeque};
use std::cmp::Ordering;

/// スパイクベースのAttention（Transformerの代替）
/// 行列演算を使わず、SDR（スパース分散表現）の共通集合サイズで類似度を計算する。
#[pyclass]
struct RustSpikeAttention {
    _input_size: usize, // [Fix] 未使用変数の警告抑制
    hidden_size: usize,
    num_heads: usize,
    memory_size: usize,
    
    // Projections
    w_query: Vec<Vec<usize>>,
    w_key: Vec<Vec<usize>>,
    w_value: Vec<Vec<usize>>,
    
    // Key-Value Memory
    memory_keys: VecDeque<Vec<Vec<usize>>>,
    memory_values: VecDeque<Vec<Vec<usize>>>,
}

// 内部ヘルパーメソッド
impl RustSpikeAttention {
    /// スパース射影 (Winner-Take-All)
    fn project(&self, input_spikes: Vec<usize>, mapping: &Vec<Vec<usize>>, sparsity: usize) -> Vec<usize> {
        let mut potentials = vec![0; self.hidden_size];
        let mut active_any = false;

        for &idx in &input_spikes {
            if idx < mapping.len() {
                for &target in &mapping[idx] {
                    if target < self.hidden_size {
                        potentials[target] += 1;
                        active_any = true;
                    }
                }
            }
        }

        if !active_any {
            return Vec::new();
        }

        // Top-K selection
        let mut candidates: Vec<(usize, i32)> = potentials.into_iter()
            .enumerate()
            .filter(|&(_, v)| v > 0)
            .collect();

        candidates.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        
        candidates.into_iter()
            .take(sparsity)
            .map(|(idx, _)| idx)
            .collect()
    }
}

#[pymethods]
impl RustSpikeAttention {
    #[new]
    fn new(input_size: usize, hidden_size: usize, num_heads: usize, memory_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        
        // スパースな射影重みを初期化
        let mut init_projection = |size_in: usize, size_out: usize| -> Vec<Vec<usize>> {
            let mut weights = Vec::with_capacity(size_in);
            for _ in 0..size_in {
                let n = (size_out as f32 * 0.05).max(1.0) as usize;
                let mut indices: Vec<usize> = (0..size_out).collect();
                indices.shuffle(&mut rng);
                weights.push(indices.into_iter().take(n).collect());
            }
            weights
        };

        RustSpikeAttention {
            _input_size: input_size,
            hidden_size,
            num_heads,
            memory_size,
            w_query: init_projection(input_size, hidden_size),
            w_key: init_projection(input_size, hidden_size),
            w_value: init_projection(input_size, hidden_size),
            memory_keys: VecDeque::with_capacity(memory_size),
            memory_values: VecDeque::with_capacity(memory_size),
        }
    }

    /// Attention機構のメイン計算 (Multi-Head)
    fn compute(&mut self, input_spikes: Vec<usize>) -> Vec<usize> {
        // 1. Current StepのQ, K, Vを生成
        let q_full = self.project(input_spikes.clone(), &self.w_query, self.hidden_size / 2);
        let k_full = self.project(input_spikes.clone(), &self.w_key, self.hidden_size / 2);
        let v_full = self.project(input_spikes.clone(), &self.w_value, self.hidden_size / 2);

        // 各ヘッドごとに分割
        let mut q_heads = vec![Vec::new(); self.num_heads];
        let mut k_heads = vec![Vec::new(); self.num_heads];
        let mut v_heads = vec![Vec::new(); self.num_heads];

        for &idx in &q_full { q_heads[idx % self.num_heads].push(idx); }
        for &idx in &k_full { k_heads[idx % self.num_heads].push(idx); }
        for &idx in &v_full { v_heads[idx % self.num_heads].push(idx); }

        // メモリに保存
        if self.memory_keys.len() >= self.memory_size {
            self.memory_keys.pop_front();
            self.memory_values.pop_front();
        }
        self.memory_keys.push_back(k_heads.clone());
        self.memory_values.push_back(v_heads);

        if self.memory_keys.len() < 2 {
            return Vec::new();
        }

        // 2. Attention Score Calculation (Intersection)
        let mut context_spikes = HashSet::new();

        for h in 0..self.num_heads {
            let q_vec: HashSet<usize> = q_heads[h].iter().cloned().collect();
            if q_vec.is_empty() { continue; }

            let mut scores: Vec<(usize, usize)> = Vec::with_capacity(self.memory_keys.len());

            for (t, past_keys) in self.memory_keys.iter().enumerate() {
                let k_vec = &past_keys[h];
                let overlap = k_vec.iter().filter(|&k| q_vec.contains(k)).count();
                if overlap > 0 {
                    scores.push((t, overlap));
                }
            }

            scores.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            
            for (t, _) in scores.iter().take(3) {
                if let Some(values) = self.memory_values.get(*t) {
                    for &v in &values[h] {
                        context_spikes.insert(v);
                    }
                }
            }
        }

        let mut result: Vec<usize> = context_spikes.into_iter().collect();
        result.sort_unstable();
        result
    }
    
    fn reset(&mut self) {
        self.memory_keys.clear();
        self.memory_values.clear();
    }
}

/// Rust版 Dynamic Liquid Layer (Feature: Homeostasis / LayerNorm equivalent)
#[pyclass]
struct RustLiquidLayer {
    size: usize,
    decay: f32,
    v: Vec<f32>,
    refractory: Vec<f32>,
    dynamic_thresh: Vec<f32>,
    base_thresh: f32,
    
    // Weights
    in_indices: Vec<Vec<usize>>,
    in_weights: Vec<Vec<f32>>,
    rec_indices: Vec<Vec<usize>>,
    rec_weights: Vec<Vec<f32>>,
    
    // Feedback weights
    feedback_weights: Vec<Vec<usize>>,
    feedback_scale: f32,
    
    // Homeostasis (Spike Layer Norm)
    activity_ma: Vec<f32>, // Moving Average of Activity
}

#[pymethods]
impl RustLiquidLayer {
    #[new]
    fn new(input_size: usize, hidden_size: usize, decay: f32, density: f32, feedback_scale: f32) -> Self {
        let mut rng = rand::thread_rng();
        let base_thresh = if decay < 0.5 { 1.1 } else if decay < 0.8 { 1.3 } else { 1.4 };
        
        let mut in_indices = Vec::with_capacity(input_size);
        let mut in_weights = Vec::with_capacity(input_size);
        for _ in 0..input_size {
            let n = (hidden_size as f32 * density) as usize;
            let mut indices: Vec<usize> = (0..hidden_size).collect();
            indices.shuffle(&mut rng);
            let selected = indices.into_iter().take(n).collect();
            let weights: Vec<f32> = (0..n).map(|_| rng.gen_range(-1.2..1.2)).collect();
            in_indices.push(selected);
            in_weights.push(weights);
        }

        let mut rec_indices = Vec::with_capacity(hidden_size);
        let mut rec_weights = Vec::with_capacity(hidden_size);
        let rec_density = 0.1;
        for i in 0..hidden_size {
            let n = (hidden_size as f32 * rec_density) as usize;
            let mut indices: Vec<usize> = (0..hidden_size).filter(|&x| x != i).collect();
            indices.shuffle(&mut rng);
            let selected = indices.into_iter().take(n).collect();
            let weights: Vec<f32> = (0..n).map(|_| rng.gen_range(-0.8..0.8)).collect();
            rec_indices.push(selected);
            rec_weights.push(weights);
        }

        let mut feedback_weights = Vec::with_capacity(hidden_size);
        for _ in 0..hidden_size {
            let n = (hidden_size as f32 * 0.05) as usize;
            let mut indices: Vec<usize> = (0..hidden_size).collect();
            indices.shuffle(&mut rng);
            feedback_weights.push(indices.into_iter().take(n).collect());
        }

        RustLiquidLayer {
            size: hidden_size,
            decay,
            v: vec![0.0; hidden_size],
            refractory: vec![0.0; hidden_size],
            dynamic_thresh: vec![base_thresh; hidden_size],
            base_thresh,
            in_indices,
            in_weights,
            rec_indices,
            rec_weights,
            feedback_weights,
            feedback_scale,
            activity_ma: vec![0.05; hidden_size],
        }
    }

    fn forward(
        &mut self, 
        active_inputs: Vec<usize>, 
        prev_active_hidden: Vec<usize>,
        feedback_active: Vec<usize>,
        attention_signal: Vec<usize>,
        learning: bool
    ) -> Vec<usize> {
        
        // 1. Decay & Refractory
        for i in 0..self.size {
            self.v[i] *= self.decay;
            if self.refractory[i] > 0.0 {
                self.refractory[i] -= 1.0;
            }
        }

        // 2. Integration
        for &pre_id in &active_inputs {
            if pre_id < self.in_indices.len() {
                for (&tgt, &w) in self.in_indices[pre_id].iter().zip(self.in_weights[pre_id].iter()) {
                    if tgt < self.size { self.v[tgt] += w; }
                }
            }
        }
        for &pre_id in &prev_active_hidden {
            if pre_id < self.rec_indices.len() {
                for (&tgt, &w) in self.rec_indices[pre_id].iter().zip(self.rec_weights[pre_id].iter()) {
                    if tgt < self.size { self.v[tgt] += w; }
                }
            }
        }
        for &fb_id in &feedback_active {
            if fb_id < self.feedback_weights.len() {
                for &tgt in &self.feedback_weights[fb_id] {
                    if tgt < self.size { self.v[tgt] += self.feedback_scale; }
                }
            }
        }
        let attn_scale = 1.5;
        for &idx in &attention_signal {
            if idx < self.size {
                self.v[idx] += attn_scale;
            }
        }

        // 3. Fire & Homeostasis
        let mut candidates = Vec::new();
        for i in 0..self.size {
            if self.v[i] >= self.dynamic_thresh[i] && self.refractory[i] <= 0.0 {
                candidates.push(i);
            }
        }

        let max_spikes = (self.size as f32 * 0.10) as usize;
        let fired_indices: Vec<usize>;
        
        if candidates.len() > max_spikes {
            candidates.sort_by(|&a, &b| self.v[b].partial_cmp(&self.v[a]).unwrap_or(Ordering::Equal));
            fired_indices = candidates.into_iter().take(max_spikes).collect();
        } else {
            fired_indices = candidates;
        }

        // 4. Update State & Homeostasis
        let mut rng = rand::thread_rng();
        let fired_set: HashSet<usize> = fired_indices.iter().cloned().collect();

        for i in 0..self.size {
            let is_fired = fired_set.contains(&i);
            
            if is_fired {
                self.v[i] = 0.0;
                self.refractory[i] = rng.gen_range(2.0..5.0);
                self.activity_ma[i] = 0.95 * self.activity_ma[i] + 0.05 * 1.0;
            } else {
                self.activity_ma[i] = 0.95 * self.activity_ma[i];
            }

            // Adaptive Threshold (Layer Norm Logic)
            let diff = self.activity_ma[i] - 0.05;
            self.dynamic_thresh[i] += diff * 0.1;

            if self.dynamic_thresh[i] < 0.3 { self.dynamic_thresh[i] = 0.3; }
            if self.dynamic_thresh[i] > 5.0 { self.dynamic_thresh[i] = 5.0; }
        }

        // 5. STDP
        if learning && !fired_indices.is_empty() && !prev_active_hidden.is_empty() {
            for &pre_id in &prev_active_hidden {
                if pre_id < self.rec_indices.len() {
                    let targets = &self.rec_indices[pre_id];
                    for i in 0..targets.len() {
                        let tgt = targets[i];
                        if fired_set.contains(&tgt) {
                             self.rec_weights[pre_id][i] += 0.02;
                             if self.rec_weights[pre_id][i] > 2.0 { self.rec_weights[pre_id][i] = 2.0; }
                        } else {
                             self.rec_weights[pre_id][i] -= 0.001;
                             if self.rec_weights[pre_id][i] < -2.0 { self.rec_weights[pre_id][i] = -2.0; }
                        }
                    }
                }
            }
        }

        fired_indices
    }

    fn reset(&mut self) {
        for x in &mut self.v { *x = 0.0; }
        for x in &mut self.refractory { *x = 0.0; }
        for x in &mut self.dynamic_thresh { *x = self.base_thresh; }
        for x in &mut self.activity_ma { *x = 0.05; }
    }
}

#[pymodule]
fn sara_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustLiquidLayer>()?;
    m.add_class::<RustSpikeAttention>()?;
    Ok(())
}