// ディレクトリパス: src/sara_engine/lib.rs
// ファイルの日本語タイトル: Rustハイブリッド SNNコア
// ファイルの目的や内容: LIFモデルのシミュレーションと、CausalLMのSTDP学習をRust側で超高速化するためのモジュール。

use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

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

    /// スパイクを入力し、膜電位を更新して、閾値を超えたニューロンのID（発火）を返す
    pub fn forward(&mut self, input_spikes: Vec<usize>) -> Vec<usize> {
        // 1. 漏れ (Leak) - 膜電位を減衰させる
        for val in self.potentials.values_mut() {
            *val *= self.decay_rate;
        }

        // 2. 統合 (Integrate) - 入力スパイクを加算する
        for &spike in input_spikes.iter() {
            *self.potentials.entry(spike).or_insert(0.0) += 1.0;
        }

        // 3. 発火 (Fire) - 閾値を超えたら発火し、電位をリセットする
        let mut fired = Vec::new();
        for (&neuron_id, val) in self.potentials.iter_mut() {
            if *val >= self.threshold {
                fired.push(neuron_id);
                *val = 0.0; // Reset after firing
            }
        }

        fired
    }
}

/// SpikingCausalLM の重み学習と推論（電位計算）を担う構造体
#[pyclass]
pub struct CausalSynapses {
    // delay -> pre_neuron -> post_token -> weight
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

    /// STDPに基づくシナプス重みの更新（Pythonの二重ループを完全にRust化）
    pub fn train_step(&mut self, spike_history: Vec<Vec<usize>>, next_token: usize, learning_rate: f32) {
        for (delay, active_spikes) in spike_history.iter().enumerate() {
            if delay > self.max_delay {
                break;
            }
            let eff_lr = learning_rate * (1.0 - (delay as f32) * 0.08);
            if eff_lr <= 0.0 {
                continue;
            }
            
            for &s in active_spikes.iter() {
                let targets = self.weights[delay].entry(s).or_insert_with(HashMap::new);
                
                // 他のターゲットへの重みを減衰（忘却）
                for (t, w) in targets.iter_mut() {
                    if *t != next_token {
                        *w *= 1.0 - eff_lr * 0.01;
                    }
                }
                
                // ターゲットへの重みを強化
                let old_w = *targets.get(&next_token).unwrap_or(&0.0);
                targets.insert(next_token, old_w + eff_lr * (1.0 - old_w));
                
                // 容量上限による正規化 (Homeostasis)
                let total_w: f32 = targets.values().sum();
                if total_w > 5.0 {
                    let scale = 5.0 / total_w;
                    for w in targets.values_mut() {
                        *w *= scale;
                    }
                }
            }
        }
    }

    /// 各トークンの発火ポテンシャルを計算
    pub fn calculate_potentials(&self, spike_history: Vec<Vec<usize>>) -> HashMap<usize, f32> {
        let mut potentials: HashMap<usize, f32> = HashMap::new();
        let mut support_count: HashMap<usize, usize> = HashMap::new();

        for (delay, active_spikes) in spike_history.iter().enumerate() {
            if delay > self.max_delay {
                break;
            }
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

        // 複数遅延からの支持ボーナス
        for (t_id, pot) in potentials.iter_mut() {
            let count = *support_count.get(t_id).unwrap_or(&1) as f32;
            *pot *= count.powf(1.2);
        }

        potentials
    }
    
    /// ハブトークンペナルティ用のファンイン計算
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

/// Pythonモジュール定義
#[pymodule]
fn sara_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LIFNetwork>()?;
    m.add_class::<CausalSynapses>()?;
    Ok(())
}