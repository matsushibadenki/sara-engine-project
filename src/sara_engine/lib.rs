use pyo3::prelude::*;
use rand::prelude::*;
use std::collections::HashSet;

/// Rust版 Dynamic Liquid Layer (Full Feature)
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
    
    // Feedback weights (fixed random projection)
    feedback_weights: Vec<Vec<usize>>,
    feedback_scale: f32,
}

#[pymethods]
impl RustLiquidLayer {
    #[new]
    fn new(input_size: usize, hidden_size: usize, decay: f32, density: f32, feedback_scale: f32) -> Self {
        let mut rng = rand::thread_rng();
        
        let base_thresh = if decay < 0.5 { 1.1 } else if decay < 0.8 { 1.3 } else { 1.4 };
        
        // 1. Input Weights
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

        // 2. Recurrent Weights
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

        // 3. Feedback Weights
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
        }
    }

    /// Forward pass with Feedback, Attention, and STDP
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

        // 2. Input Integration
        for &pre_id in &active_inputs {
            if pre_id < self.in_indices.len() {
                let targets = &self.in_indices[pre_id];
                let weights = &self.in_weights[pre_id];
                for (&tgt, &w) in targets.iter().zip(weights.iter()) {
                    if tgt < self.size { self.v[tgt] += w; }
                }
            }
        }

        // 3. Recurrent Integration
        for &pre_id in &prev_active_hidden {
            if pre_id < self.rec_indices.len() {
                let targets = &self.rec_indices[pre_id];
                let weights = &self.rec_weights[pre_id];
                for (&tgt, &w) in targets.iter().zip(weights.iter()) {
                    if tgt < self.size { self.v[tgt] += w; }
                }
            }
        }

        // 4. Feedback Integration
        for &fb_id in &feedback_active {
            if fb_id < self.feedback_weights.len() {
                for &tgt in &self.feedback_weights[fb_id] {
                    if tgt < self.size { self.v[tgt] += self.feedback_scale; }
                }
            }
        }

        // 5. Attention Integration
        let attn_scale = 1.5;
        for &idx in &attention_signal {
            if idx < self.size {
                self.v[idx] += attn_scale;
            }
        }

        // 6. Fire
        let mut candidates = Vec::new();
        for i in 0..self.size {
            if self.v[i] >= self.dynamic_thresh[i] && self.refractory[i] <= 0.0 {
                candidates.push(i);
            }
        }

        // Sparsity Control (Top-K)
        let max_spikes = (self.size as f32 * 0.10) as usize;
        let fired_indices: Vec<usize>;
        
        if candidates.len() > max_spikes {
            candidates.sort_by(|&a, &b| self.v[b].partial_cmp(&self.v[a]).unwrap());
            
            // --- [修正] ここで長さを保存しておく ---
            let candidate_len = candidates.len();
            
            fired_indices = candidates.into_iter().take(max_spikes).collect();
            
            // Penalty for excess activity
            let excess_ratio = (candidate_len as f32) / (max_spikes as f32);
            let penalty = 0.05 * excess_ratio.ln();
            for &idx in &fired_indices {
                self.dynamic_thresh[idx] += penalty;
            }
        } else {
            fired_indices = candidates;
        }

        // 7. Post-fire updates
        let mut rng = rand::thread_rng();
        for &idx in &fired_indices {
            self.v[idx] = 0.0;
            self.refractory[idx] = rng.gen_range(2.0..5.0);
            self.dynamic_thresh[idx] += 0.03;
        }

        // 8. Homeostasis (Recovery)
        let fired_set: HashSet<usize> = fired_indices.iter().cloned().collect();
        for i in 0..self.size {
            if !fired_set.contains(&i) {
                if self.dynamic_thresh[i] > 0.3 {
                    self.dynamic_thresh[i] -= 0.005;
                }
            }
        }
        for x in &mut self.dynamic_thresh {
            if *x > 5.0 { *x = 5.0; }
            if *x < 0.3 { *x = 0.3; }
        }

        // 9. STDP (Simple Hebbian)
        if learning && !fired_indices.is_empty() && !prev_active_hidden.is_empty() {
            for &pre_id in &prev_active_hidden {
                if pre_id < self.rec_indices.len() {
                    let targets = &self.rec_indices[pre_id];
                    for i in 0..targets.len() {
                        let tgt = targets[i];
                        if fired_set.contains(&tgt) {
                             self.rec_weights[pre_id][i] += 0.01;
                             if self.rec_weights[pre_id][i] > 2.0 { self.rec_weights[pre_id][i] = 2.0; }
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
    }
}

#[pymodule]
fn sara_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustLiquidLayer>()?;
    Ok(())
}