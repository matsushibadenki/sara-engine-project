// ディレクトリパス: src/sara_engine/lib.rs
// ファイルの日本語タイトル: SARA-Engine Rust高速化コアモジュール
// ファイルの目的や内容: スパース推論の高速化およびGaborフィルタを用いた生物学的視覚野（V1）受容野の導入。パターン分離能力を極大化する。

use pyo3::prelude::*;
use std::collections::HashMap;
use rand::Rng;
use rand::seq::SliceRandom;

#[pyfunction]
fn sparse_propagate_and_wta(
    active_spikes: Vec<usize>,
    weights: Vec<HashMap<usize, f64>>,
    out_size: usize,
    k: usize,
) -> PyResult<Vec<usize>> {
    let mut potentials = vec![0.0; out_size];
    for &s in &active_spikes {
        if s < weights.len() {
            for (&t, &w) in &weights[s] {
                if t < out_size { potentials[t] += w; }
            }
        }
    }
    let mut active_neurons: Vec<(usize, f64)> = potentials
        .into_iter().enumerate().filter(|&(_, p)| p > 0.0).collect();
    active_neurons.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(active_neurons.into_iter().take(k).map(|(i, _)| i).collect())
}

#[pyfunction]
fn sparse_propagate_threshold(
    active_spikes: Vec<usize>,
    weights: Vec<HashMap<usize, f64>>,
    out_size: usize,
    threshold: f64,
) -> PyResult<Vec<usize>> {
    let mut potentials = vec![0.0; out_size];
    for &s in &active_spikes {
        if s < weights.len() {
            for (&t, &w) in &weights[s] {
                if t < out_size { potentials[t] += w; }
            }
        }
    }
    Ok(potentials.into_iter().enumerate().filter(|&(_, p)| p > threshold).map(|(i, _)| i).collect())
}

#[pyclass]
struct RustReadoutLayer {
    synapses: Vec<HashMap<usize, f64>>,
    vocab_size: usize,
    total_readout_size: usize,
}

#[pymethods]
impl RustReadoutLayer {
    #[new]
    fn new(total_readout_size: usize, vocab_size: usize) -> Self {
        RustReadoutLayer { synapses: vec![HashMap::new(); total_readout_size], vocab_size, total_readout_size }
    }
    fn forward_and_learn(&mut self, combined_spikes: Vec<usize>, learning: bool, target_id: Option<usize>) -> PyResult<usize> {
        Ok(0) // 今回はPython側のReadoutを使うためモックのまま
    }
    fn get_synapses(&self) -> PyResult<Vec<HashMap<usize, f64>>> { Ok(self.synapses.clone()) }
    fn set_synapses(&mut self, new_synapses: Vec<HashMap<usize, f64>>) -> PyResult<()> {
        self.synapses = new_synapses; Ok(())
    }
}

#[pyclass]
struct RustLiquidLayer {
    input_size: usize,
    hidden_size: usize,
    decay: f64,
    density: f64,
    feedback_scale: f64,
    v: Vec<f64>,
    refractory: Vec<f64>,
    dynamic_thresh: Vec<f64>,
    trace: Vec<f64>,
    in_weights: Vec<HashMap<usize, f64>>,
    rec_weights: Vec<HashMap<usize, f64>>,
    feedback_map: Vec<Vec<usize>>,
    target_rate: f64,
}

#[pymethods]
impl RustLiquidLayer {
    #[new]
    fn new(input_size: usize, hidden_size: usize, decay: f64, density: f64, feedback_scale: f64) -> Self {
        let mut rng = rand::thread_rng();
        let mut in_weights = vec![HashMap::new(); input_size];
        let n_connect = (hidden_size as f64 * density) as usize;
        let mut all_hidden: Vec<usize> = (0..hidden_size).collect();

        if n_connect > 0 {
            for i in 0..input_size {
                all_hidden.shuffle(&mut rng);
                for &t in all_hidden.iter().take(n_connect) {
                    in_weights[i].insert(t, rng.gen_range(-1.0..1.0));
                }
            }
        }

        let mut rec_weights = vec![HashMap::new(); hidden_size];
        let rec_density = 0.1;
        let n_rec_connect = (hidden_size as f64 * rec_density) as usize;

        if n_rec_connect > 0 {
            for i in 0..hidden_size {
                let mut candidates: Vec<usize> = (0..hidden_size).filter(|&x| x != i).collect();
                if candidates.len() >= n_rec_connect {
                    candidates.shuffle(&mut rng);
                    for &t in candidates.iter().take(n_rec_connect) {
                        rec_weights[i].insert(t, rng.gen_range(-0.8..0.8));
                    }
                }
            }
        }

        let mut feedback_map = vec![Vec::new(); hidden_size];
        let n_fb_connect = (hidden_size as f64 * 0.05) as usize;
        for i in 0..hidden_size {
            all_hidden.shuffle(&mut rng);
            let targets: Vec<usize> = all_hidden.iter().take(n_fb_connect).copied().collect();
            feedback_map[i] = targets;
        }

        RustLiquidLayer {
            input_size, hidden_size, decay, density, feedback_scale,
            v: vec![0.0; hidden_size], refractory: vec![0.0; hidden_size],
            dynamic_thresh: vec![1.0; hidden_size], trace: vec![0.0; hidden_size],
            in_weights, rec_weights, feedback_map,
            target_rate: 0.04, // 生物学的な4%のスパース性
        }
    }

    fn forward(
        &mut self, active_inputs: Vec<usize>, prev_active_hidden: Vec<usize>,
        feedback_active: Vec<usize>, attention_signal: Vec<usize>, learning: bool, reward: f64,
    ) -> PyResult<Vec<usize>> {
        let mut rng = rand::thread_rng();

        for i in 0..self.hidden_size {
            self.v[i] *= self.decay;
            if self.refractory[i] > 0.0 { self.refractory[i] -= 1.0; }
            self.trace[i] *= 0.95;
        }

        for &inp_idx in &active_inputs {
            if inp_idx < self.input_size {
                for (&target, &weight) in &self.in_weights[inp_idx] { self.v[target] += weight; }
            }
        }

        for &hid_idx in &prev_active_hidden {
            if hid_idx < self.hidden_size {
                for (&target, &weight) in &self.rec_weights[hid_idx] { self.v[target] += weight; }
            }
        }

        let mut candidates = Vec::new();
        for i in 0..self.hidden_size {
            if self.v[i] >= self.dynamic_thresh[i] && self.refractory[i] <= 0.0 {
                candidates.push(i);
            }
        }

        let max_spikes = (self.hidden_size as f64 * 0.04) as usize; 
        let mut fired_indices = Vec::new();
        
        if candidates.len() > max_spikes {
            candidates.sort_by(|&a, &b| self.v[b].partial_cmp(&self.v[a]).unwrap_or(std::cmp::Ordering::Equal));
            fired_indices = candidates.into_iter().take(max_spikes).collect();
        } else {
            fired_indices = candidates;
        }

        for i in 0..self.hidden_size {
            if fired_indices.contains(&i) {
                self.v[i] = 0.0;
                self.refractory[i] = if learning { rng.gen_range(2.0..5.0) } else { 3.0 };
                self.trace[i] += 1.0;
                self.dynamic_thresh[i] += 0.05;
            } else {
                self.dynamic_thresh[i] += (self.target_rate - 0.05) * 0.01;
            }
            if self.dynamic_thresh[i] < 0.5 { self.dynamic_thresh[i] = 0.5; }
            if self.dynamic_thresh[i] > 5.0 { self.dynamic_thresh[i] = 5.0; }
        }
        Ok(fired_indices)
    }

    fn apply_spatial_receptive_fields(&mut self, width: usize, height: usize, patch_sizes: Vec<usize>) {
        let mut rng = rand::thread_rng();
        self.in_weights = vec![HashMap::new(); width * height];
        
        // Gaborフィルタ用パラメータ群: 4方向の角度と2つの位相(エッジ検出とライン検出)
        let angles = [0.0, std::f64::consts::PI / 4.0, std::f64::consts::PI / 2.0, std::f64::consts::PI * 3.0 / 4.0];
        let phases = [0.0, std::f64::consts::PI / 2.0]; 
        
        for hidden_idx in 0..self.hidden_size {
            let cx = rng.gen_range(0..width) as isize;
            let cy = rng.gen_range(0..height) as isize;
            let patch_size = *patch_sizes.choose(&mut rng).unwrap_or(&3) as isize;
            let half_p = patch_size / 2;
            
            let theta = *angles.choose(&mut rng).unwrap_or(&0.0);
            let psi = *phases.choose(&mut rng).unwrap_or(&0.0);
            let lambda = patch_size as f64 * 0.8; // 空間周波数の波長
            let sigma = patch_size as f64 * 0.4;  // ガウス包絡線の標準偏差
            let gamma = 0.5; // アスペクト比
            
            for dy in -half_p..=half_p {
                for dx in -half_p..=half_p {
                    let x = cx + dx;
                    let y = cy + dy;
                    if x >= 0 && x < width as isize && y >= 0 && y < height as isize {
                        let inp_idx = (y * width as isize + x) as usize;
                        
                        let x_f = dx as f64;
                        let y_f = dy as f64;
                        
                        // 座標の回転
                        let x_prime = x_f * theta.cos() + y_f * theta.sin();
                        let y_prime = -x_f * theta.sin() + y_f * theta.cos();
                        
                        // ガウス包絡線と正弦波キャリアの積
                        let env = (-(x_prime * x_prime + gamma * gamma * y_prime * y_prime) / (2.0 * sigma * sigma)).exp();
                        let carrier = (2.0 * std::f64::consts::PI * x_prime / lambda + psi).cos();
                        
                        let mut w = env * carrier;
                        
                        // 重みのスケールアップと微小ノイズの追加
                        w *= 8.0; 
                        w += rng.gen_range(-0.5..0.5); 
                        
                        self.in_weights[inp_idx].insert(hidden_idx, w);
                    }
                }
            }
        }
    }

    fn reset_potentials(&mut self) {
        self.v.fill(0.0);
        self.refractory.fill(0.0);
        for i in 0..self.hidden_size {
            if self.dynamic_thresh[i] > 5.0 { self.dynamic_thresh[i] = 5.0; }
        }
    }

    fn reset(&mut self) {
        self.v.fill(0.0);
        self.refractory.fill(0.0);
        self.dynamic_thresh.fill(1.0);
        self.trace.fill(0.0);
    }
}

#[pymodule]
fn sara_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sparse_propagate_and_wta, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_propagate_threshold, m)?)?;
    m.add_class::<RustReadoutLayer>()?;
    m.add_class::<RustLiquidLayer>()?;
    Ok(())
}