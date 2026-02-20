use pyo3::prelude::*;
use rand::prelude::*;
use std::collections::{HashSet, VecDeque};
use std::cmp::Ordering;

#[pyclass]
struct RustSpikeAttention {
    _input_size: usize,
    hidden_size: usize,
    num_heads: usize,
    memory_size: usize,
    w_query: Vec<Vec<usize>>,
    w_key: Vec<Vec<usize>>,
    w_value: Vec<Vec<usize>>,
    memory_keys: VecDeque<Vec<Vec<usize>>>,
    memory_values: VecDeque<Vec<Vec<usize>>>,
}

impl RustSpikeAttention {
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

        if !active_any { return Vec::new(); }

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

    fn compute(&mut self, input_spikes: Vec<usize>) -> Vec<usize> {
        let q_full = self.project(input_spikes.clone(), &self.w_query, self.hidden_size / 2);
        let k_full = self.project(input_spikes.clone(), &self.w_key, self.hidden_size / 2);
        let v_full = self.project(input_spikes.clone(), &self.w_value, self.hidden_size / 2);

        let mut q_heads = vec![Vec::new(); self.num_heads];
        let mut k_heads = vec![Vec::new(); self.num_heads];
        let mut v_heads = vec![Vec::new(); self.num_heads];

        for &idx in &q_full { q_heads[idx % self.num_heads].push(idx); }
        for &idx in &k_full { k_heads[idx % self.num_heads].push(idx); }
        for &idx in &v_full { v_heads[idx % self.num_heads].push(idx); }

        if self.memory_keys.len() >= self.memory_size {
            self.memory_keys.pop_front();
            self.memory_values.pop_front();
        }
        self.memory_keys.push_back(k_heads.clone());
        self.memory_values.push_back(v_heads);

        if self.memory_keys.len() < 2 { return Vec::new(); }

        let mut context_spikes = HashSet::new();

        for h in 0..self.num_heads {
            let q_vec: HashSet<usize> = q_heads[h].iter().cloned().collect();
            if q_vec.is_empty() { continue; }

            let mut scores: Vec<(usize, usize)> = Vec::with_capacity(self.memory_keys.len());

            for (t, past_keys) in self.memory_keys.iter().enumerate() {
                let k_vec = &past_keys[h];
                let overlap = k_vec.iter().filter(|&k| q_vec.contains(k)).count();
                if overlap > 0 { scores.push((t, overlap)); }
            }

            scores.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            
            for (t, _) in scores.iter().take(3) {
                if let Some(values) = self.memory_values.get(*t) {
                    for &v in &values[h] { context_spikes.insert(v); }
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

#[pyclass]
struct RustLiquidLayer {
    size: usize,
    decay: f32,
    v: Vec<f32>,
    refractory: Vec<f32>,
    dynamic_thresh: Vec<f32>,
    base_thresh: f32,
    in_indices: Vec<Vec<usize>>,
    in_weights: Vec<Vec<f32>>,
    rec_indices: Vec<Vec<usize>>,
    rec_weights: Vec<Vec<f32>>,
    feedback_weights: Vec<Vec<usize>>,
    feedback_scale: f32,
    activity_ma: Vec<f32>,
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

    pub fn apply_spatial_receptive_fields(&mut self, width: usize, height: usize, patch_sizes: Vec<usize>) {
        let mut rng = rand::thread_rng();
        let num_inputs = width * height;
        let mut new_in_indices: Vec<Vec<usize>> = vec![Vec::new(); num_inputs];
        let mut new_in_weights: Vec<Vec<f32>> = vec![Vec::new(); num_inputs];

        for hidden_idx in 0..self.size {
            let cx = rng.gen_range(0..width) as i32;
            let cy = rng.gen_range(0..height) as i32;
            let patch_size = if !patch_sizes.is_empty() { *patch_sizes.choose(&mut rng).unwrap() as i32 } else { 3 };
            let half_p = patch_size / 2;
            let pattern_type = rng.gen_range(0..7);
            
            for dy in -half_p..=half_p {
                for dx in -half_p..=half_p {
                    let x = cx + dx;
                    let y = cy + dy;
                    if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                        let inp_idx = (y * width as i32 + x) as usize;
                        let mut w: f32 = match pattern_type {
                            0 => rng.gen_range(-4.0..4.0),
                            1 => if dx*dx + dy*dy <= (half_p as f32 * 0.6).powi(2) as i32 { 6.0 } else { -3.0 },
                            2 => if dx*dx + dy*dy <= (half_p as f32 * 0.6).powi(2) as i32 { -6.0 } else { 3.0 },
                            3 => if dy < 0 { 6.0 } else { -6.0 },
                            4 => if dx < 0 { 6.0 } else { -6.0 },
                            5 => if dx > dy { 6.0 } else { -6.0 },
                            6 => if dx > -dy { 6.0 } else { -6.0 },
                            _ => 0.0,
                        };
                        w += rng.gen_range(-0.8..0.8);
                        w *= 1.5;
                        new_in_indices[inp_idx].push(hidden_idx);
                        new_in_weights[inp_idx].push(w);
                    }
                }
            }
        }
        self.in_indices = new_in_indices;
        self.in_weights = new_in_weights;
    }

    pub fn reset_potentials(&mut self) {
        for x in &mut self.v { *x = 0.0; }
        for x in &mut self.refractory { *x = 0.0; }
        for x in &mut self.dynamic_thresh { 
            if *x > 5.0 { *x = 5.0; }
        }
    }

    pub fn get_state(&self) -> (Vec<f32>, Vec<f32>) {
        (self.v.clone(), self.dynamic_thresh.clone())
    }

    fn forward(
        &mut self, 
        active_inputs: Vec<usize>, 
        prev_active_hidden: Vec<usize>,
        feedback_active: Vec<usize>,
        attention_signal: Vec<usize>,
        learning: bool
    ) -> Vec<usize> {
        
        for i in 0..self.size {
            self.v[i] *= self.decay;
            if self.refractory[i] > 0.0 { self.refractory[i] -= 1.0; }
        }

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
            if idx < self.size { self.v[idx] += attn_scale; }
        }

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
            let diff = self.activity_ma[i] - 0.05;
            self.dynamic_thresh[i] += diff * 0.1;
            if self.dynamic_thresh[i] < 0.3 { self.dynamic_thresh[i] = 0.3; }
            if self.dynamic_thresh[i] > 5.0 { self.dynamic_thresh[i] = 5.0; }
        }

        // ==========================================
        // 5. STDP (Spike-Timing-Dependent Plasticity)
        // ==========================================
        if learning && !fired_indices.is_empty() {
            // (A) 入力フィルターの乗法型STDP (Multiplicative STDP)
            if !active_inputs.is_empty() {
                let active_set: HashSet<usize> = active_inputs.iter().cloned().collect();
                for pre_id in 0..self.in_indices.len() {
                    let is_pre_active = active_set.contains(&pre_id);
                    for i in 0..self.in_indices[pre_id].len() {
                        let tgt = self.in_indices[pre_id][i];
                        if fired_set.contains(&tgt) {
                            let mut w = self.in_weights[pre_id][i];
                            if is_pre_active {
                                w *= 1.05; // LTP: 発火が一致した場合、現在の役割(興奮/抑制)を増幅
                            } else {
                                w *= 0.95; // LTD: 使われなかった場合は0に近づける(忘却)
                            }
                            // 暴走を防ぐクリッピング
                            if w > 10.0 { w = 10.0; }
                            if w < -10.0 { w = -10.0; }
                            self.in_weights[pre_id][i] = w;
                        }
                    }
                }
            }

            // (B) 隠れ層間のSTDP
            if !prev_active_hidden.is_empty() {
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