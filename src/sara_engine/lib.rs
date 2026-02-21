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

#[pymodule]
fn sara_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sparse_propagate_and_wta, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_propagate_threshold, m)?)?;
    Ok(())
}