use pyo3::prelude::*;
use std::collections::HashMap;

/// プロパゲーションとK-Winner-Take-Allを高速に行う関数 (FFN用)
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
                if t < out_size {
                    potentials[t] += w;
                }
            }
        }
    }

    let mut active_neurons: Vec<(usize, f64)> = potentials
        .into_iter()
        .enumerate()
        .filter(|&(_, p)| p > 0.0)
        .collect();

    // 電位が高い順にソート (WTA)
    active_neurons.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let result: Vec<usize> = active_neurons.into_iter().take(k).map(|(i, _)| i).collect();

    Ok(result)
}

/// 閾値によるスパイク発火を高速に行う関数 (Attention用)
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
                if t < out_size {
                    potentials[t] += w;
                }
            }
        }
    }
    
    let result: Vec<usize> = potentials
        .into_iter()
        .enumerate()
        .filter(|&(_, p)| p > threshold)
        .map(|(i, _)| i)
        .collect();
        
    Ok(result)
}

/// 予測とSTDP学習を同時に行うリードアウト層のRust実装
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
        RustReadoutLayer {
            synapses: vec![HashMap::new(); total_readout_size],
            vocab_size,
            total_readout_size,
        }
    }

    /// 電位計算・予測WTA・STDP学習をRust側で一括して高速実行する
    fn forward_and_learn(
        &mut self,
        combined_spikes: Vec<usize>,
        learning: bool,
        target_id: Option<usize>,
    ) -> PyResult<usize> {
        let mut out_potentials = vec![0.0; self.vocab_size];

        // 1. スパースな電位計算
        for &s in &combined_spikes {
            if s < self.total_readout_size {
                for (&v_idx, &w) in &self.synapses[s] {
                    if v_idx < self.vocab_size {
                        out_potentials[v_idx] += w;
                    }
                }
            }
        }

        // 2. Winner-Take-All (最大値の取得)
        let mut max_p = 0.0;
        let mut predicted_id = 32; // フォールバック(Space)
        for (i, &p) in out_potentials.iter().enumerate() {
            if p > max_p {
                max_p = p;
                predicted_id = i;
            }
        }

        if max_p <= 0.1 {
            predicted_id = 32;
        }

        // 3. 予測誤差に基づくSTDP学習
        if learning {
            if let Some(t_id) = target_id {
                for &s in &combined_spikes {
                    if s < self.total_readout_size {
                        // LTP: 正解ルートの増強
                        let current_w = *self.synapses[s].get(&t_id).unwrap_or(&0.0);
                        let new_w = if current_w + 1.5 > 15.0 { 15.0 } else { current_w + 1.5 };
                        self.synapses[s].insert(t_id, new_w);

                        // LTD: 誤予測ルートの抑圧
                        if predicted_id != t_id {
                            if let std::collections::hash_map::Entry::Occupied(mut o) = self.synapses[s].entry(predicted_id) {
                                *o.get_mut() -= 0.1;
                                if *o.get() <= 0.0 {
                                    o.remove_entry(); // シナプスの刈り込み
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(predicted_id)
    }

    /// JSON保存用にシナプス重みをPython側へ返す
    fn get_synapses(&self) -> PyResult<Vec<HashMap<usize, f64>>> {
        Ok(self.synapses.clone())
    }

    /// JSONから復元したシナプス重みをRust側へセットする
    fn set_synapses(&mut self, new_synapses: Vec<HashMap<usize, f64>>) -> PyResult<()> {
        self.synapses = new_synapses;
        Ok(())
    }
}

#[pymodule]
fn sara_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sparse_propagate_and_wta, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_propagate_threshold, m)?)?;
    m.add_class::<RustReadoutLayer>()?;
    Ok(())
}