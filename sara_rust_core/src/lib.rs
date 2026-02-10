// sara_rust_core/src/lib.rs
use pyo3::prelude::*;
use rand::prelude::*;
use sha2::{Sha256, Digest};
use std::collections::HashMap;

/// SDRエンコーダー: 文字列をスパースなビット列に変換
#[pyclass]
#[derive(Clone)]
struct SDREncoder {
    input_size: usize,
    density: f32,
    cache: HashMap<String, Vec<usize>>,
}

#[pymethods]
impl SDREncoder {
    #[new]
    fn new(input_size: usize, density: f32) -> Self {
        SDREncoder {
            input_size,
            density,
            cache: HashMap::new(),
        }
    }

    fn encode(&mut self, text: String) -> Vec<usize> {
        if let Some(indices) = self.cache.get(&text) {
            return indices.clone();
        }

        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let result = hasher.finalize();
        
        // ハッシュ値をシードにして決定論的にインデックスを選択
        // (簡易実装としてu64シードを使用)
        let seed = u64::from_be_bytes(result[0..8].try_into().unwrap());
        let mut rng = StdRng::seed_from_u64(seed);
        
        let n_active = (self.input_size as f32 * self.density) as usize;
        let mut indices: Vec<usize> = (0..self.input_size).collect();
        indices.shuffle(&mut rng);
        
        let active_indices: Vec<usize> = indices.into_iter().take(n_active).collect();
        self.cache.insert(text, active_indices.clone());
        active_indices
    }
}

/// リキッド層: スパイクニューラルネットワークの1層
#[pyclass]
struct LiquidLayer {
    size: usize,
    decay: f32,
    v: Vec<f32>,              // 膜電位
    refractory: Vec<i32>,      // 不応期カウンタ
    thresh: Vec<f32>,         // 発火しきい値
    
    // スパース結合 (Adjacency List): index -> [(target, weight)]
    in_weights: Vec<Vec<(usize, f32)>>,
    rec_weights: Vec<Vec<(usize, f32)>>,
    
    // 短期可塑性・ゲーティング (Associative Memory / Gating)
    stp_gains: Vec<Vec<f32>>, 
}

impl LiquidLayer {
    fn new(input_size: usize, hidden_size: usize, decay: f32, density: f32, input_scale: f32, rec_scale: f32) -> Self {
        let mut rng = thread_rng();
        
        // 入力結合の初期化
        let mut in_weights = vec![Vec::new(); input_size];
        for i in 0..input_size {
            // 各入力ニューロンからランダムに投影
            let fan_out = (hidden_size as f32 * density) as usize;
            for _ in 0..fan_out {
                let target = rng.gen_range(0..hidden_size);
                let w = rng.gen_range(0.0..1.0) * input_scale;
                in_weights[i].push((target, w));
            }
        }

        // リカレント結合の初期化
        let mut rec_weights = vec![Vec::new(); hidden_size];
        let mut stp_gains = vec![Vec::new(); hidden_size];
        for i in 0..hidden_size {
            let fan_out = (hidden_size as f32 * density) as usize;
            for _ in 0..fan_out {
                let target = rng.gen_range(0..hidden_size);
                if i == target { continue; } // 自己結合回避
                let w = (rng.gen_range(-1.0..1.0) as f32) * rec_scale;
                rec_weights[i].push((target, w));
                stp_gains[i].push(1.0); // 初期ゲイン
            }
        }

        LiquidLayer {
            size: hidden_size,
            decay,
            v: vec![0.0; hidden_size],
            refractory: vec![0; hidden_size],
            thresh: vec![1.0; hidden_size], // 可変しきい値も実装可能
            in_weights,
            rec_weights,
            stp_gains,
        }
    }

    fn forward(&mut self, active_inputs: &[usize], prev_active_hidden: &[usize]) -> Vec<usize> {
        // 電位減衰
        for x in self.v.iter_mut() { *x *= self.decay; }
        for r in self.refractory.iter_mut() { *r = (*r - 1).max(0); }

        // 入力スパイクの統合
        for &pre_id in active_inputs {
            if let Some(conns) = self.in_weights.get(pre_id) {
                for (target, w) in conns {
                    self.v[*target] += w;
                }
            }
        }

        // リカレントスパイクの統合 (with STP/Gating)
        for &pre_id in prev_active_hidden {
            if let Some(conns) = self.rec_weights.get(pre_id) {
                let gains = &mut self.stp_gains[pre_id];
                
                for (i, (target, w)) in conns.iter().enumerate() {
                    let gain = gains[i];
                    self.v[*target] += w * gain;
                    
                    // STP: 使われた経路を強化 (Facilitation)
                    gains[i] = (gains[i] + 0.1).min(3.0);
                }
            }
        }

        // STPゲインの自然減衰
        for gains in self.stp_gains.iter_mut() {
            for g in gains.iter_mut() {
                *g += (1.0 - *g) * 0.1;
            }
        }

        // 発火判定
        let mut fired = Vec::new();
        for i in 0..self.size {
            if self.v[i] >= self.thresh[i] && self.refractory[i] <= 0 {
                self.v[i] -= self.thresh[i]; // リセット
                self.refractory[i] = 2;      // 不応期
                fired.push(i);
            }
        }
        fired
    }
    
    fn reset(&mut self) {
        self.v.fill(0.0);
        self.refractory.fill(0);
        for gains in self.stp_gains.iter_mut() {
            gains.fill(1.0);
        }
    }
}

/// メインエンジン: Pythonから呼び出されるクラス
#[pyclass]
struct SaraRustBrain {
    sdr_size: usize,
    encoder: SDREncoder,
    l1: LiquidLayer,
    l2: LiquidLayer,
    l3: LiquidLayer,
    
    // Readout Weights: Index -> [(HiddenNeuronIdx, Weight)]
    // 全結合ではなくスパース結合
    readout_map: Vec<Vec<usize>>, 
    readout_w: Vec<Vec<f32>>,
    
    // 状態管理
    prev_spikes_l1: Vec<usize>,
    prev_spikes_l2: Vec<usize>,
    prev_spikes_l3: Vec<usize>,
    readout_refractory: Vec<i32>,
    
    lr: f32,
}

#[pymethods]
impl SaraRustBrain {
    #[new]
    fn new(sdr_size: usize) -> Self {
        let h_size = 2000;
        
        // 階層型リザーバ (Stacked LSM)
        let l1 = LiquidLayer::new(sdr_size, h_size, 0.3, 0.05, 3.0, 1.2);
        let l2 = LiquidLayer::new(h_size, h_size, 0.6, 0.05, 2.0, 1.5);
        let l3 = LiquidLayer::new(h_size, h_size, 0.95, 0.05, 1.5, 2.0);
        
        // Readout層の初期化 (全層からのランダム接続)
        let total_hidden = h_size * 3;
        let mut readout_map = vec![Vec::new(); sdr_size];
        let mut readout_w = vec![Vec::new(); sdr_size];
        let mut rng = thread_rng();
        
        for i in 0..sdr_size {
            let fan_in = 400;
            for _ in 0..fan_in {
                readout_map[i].push(rng.gen_range(0..total_hidden));
                readout_w[i].push(0.0);
            }
        }

        SaraRustBrain {
            sdr_size,
            encoder: SDREncoder::new(sdr_size, 0.02),
            l1, l2, l3,
            readout_map, readout_w,
            prev_spikes_l1: Vec::new(),
            prev_spikes_l2: Vec::new(),
            prev_spikes_l3: Vec::new(),
            readout_refractory: vec![0; sdr_size],
            lr: 0.05,
        }
    }

    fn reset(&mut self) {
        self.l1.reset();
        self.l2.reset();
        self.l3.reset();
        self.prev_spikes_l1.clear();
        self.prev_spikes_l2.clear();
        self.prev_spikes_l3.clear();
        self.readout_refractory.fill(0);
    }

    /// 順伝播 (1ステップ)
    fn forward_step(&mut self, text_input: Option<String>, training: bool) -> (Vec<usize>, Vec<usize>) {
        // 1. エンコード
        let input_sdr = if let Some(text) = text_input {
            self.encoder.encode(text)
        } else {
            Vec::new() // 無入力（自己想起中など）
        };

        // 2. 階層処理
        let spikes_1 = self.l1.forward(&input_sdr, &self.prev_spikes_l1);
        let spikes_2 = self.l2.forward(&spikes_1, &self.prev_spikes_l2);
        let spikes_3 = self.l3.forward(&spikes_2, &self.prev_spikes_l3);

        // 3. 全層スパイクの統合 (Offset加算)
        let h_size = 2000;
        let mut all_spikes = Vec::with_capacity(spikes_1.len() + spikes_2.len() + spikes_3.len());
        all_spikes.extend_from_slice(&spikes_1);
        for &x in &spikes_2 { all_spikes.push(x + h_size); }
        for &x in &spikes_3 { all_spikes.push(x + h_size * 2); }

        // 4. Readout (Spike Decoder)
        let mut potentials = vec![0.0; self.sdr_size];
        
        // 高速なスパース内積
        // Note: Python版より圧倒的に速い
        for (out_idx, (inputs, weights)) in self.readout_map.iter().zip(&self.readout_w).enumerate() {
            if self.readout_refractory[out_idx] > 0 {
                potentials[out_idx] = -999.0;
                continue;
            }
            
            // RustのHashSetを使うか、ソート済み配列で二分探索するとさらに速いが
            // ここでは簡易的に全探索（スパイク数が少ないので十分高速）
            let mut sum_p = 0.0;
            for (in_idx, w) in inputs.iter().zip(weights) {
                if all_spikes.contains(in_idx) { // TODO: Optimize contains
                    sum_p += w;
                }
            }
            potentials[out_idx] = sum_p;
        }

        // 不応期更新
        for r in self.readout_refractory.iter_mut() { *r = (*r - 1).max(0); }

        // しきい値判定 & 上位選択
        let mut predicted_sdr = Vec::new();
        let threshold = 0.5;
        
        // 候補の収集
        let mut candidates: Vec<(usize, f32)> = potentials.iter()
            .enumerate()
            .filter(|(_, &p)| p > threshold)
            .map(|(i, &p)| (i, p))
            .collect();
        
        if !candidates.is_empty() {
            // 上位N個を選択
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let n_predict = (self.sdr_size as f32 * 0.02) as usize;
            for (idx, _) in candidates.into_iter().take(n_predict) {
                predicted_sdr.push(idx);
            }
        } else if !training {
             // Fallback (Force Output)
             // 学習中でなければ、しきい値以下でも一番強いものを出す
             let max_p = potentials.iter().fold(-f32::INFINITY, |a, &b| a.max(b));
             if max_p > -100.0 {
                 if let Some(idx) = potentials.iter().position(|&x| x == max_p) {
                     predicted_sdr.push(idx);
                 }
             }
        }

        // 不応期セット
        if !training {
            for &idx in &predicted_sdr {
                if idx < self.readout_refractory.len() {
                    self.readout_refractory[idx] = 2;
                }
            }
        }

        // 状態更新
        self.prev_spikes_l1 = spikes_1;
        self.prev_spikes_l2 = spikes_2;
        self.prev_spikes_l3 = spikes_3;

        (predicted_sdr, all_spikes)
    }

    /// 学習ステップ (Competitive Hebbian)
    fn train_step(&mut self, input_text: String, target_text: String) {
        let target_sdr = self.encoder.encode(target_text);
        
        // 予測実行
        let (predicted_sdr, active_spikes) = self.forward_step(Some(input_text), true);
        
        if active_spikes.is_empty() { return; }

        // 重み更新 (Rustの並列イテレータ Rayon を使うとさらに高速化可能)
        for out_idx in 0..self.sdr_size {
            let is_target = target_sdr.contains(&out_idx);
            let is_predicted = predicted_sdr.contains(&out_idx);
            
            if is_target || is_predicted {
                let inputs = &self.readout_map[out_idx];
                let weights = &mut self.readout_w[out_idx];
                
                for (i, &in_idx) in inputs.iter().enumerate() {
                    if active_spikes.contains(&in_idx) {
                        if is_target {
                            // LTP
                            weights[i] += self.lr * (5.0 - weights[i]);
                        } else if is_predicted {
                            // LTD
                            weights[i] -= self.lr * 1.5 * weights[i];
                        }
                    }
                }
            }
        }
    }
    
    // 他、generateメソッドやsave/loadメソッドを実装...
}

/// Pythonモジュール定義
#[pymodule]
fn sara_rust_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SDREncoder>()?;
    m.add_class::<SaraRustBrain>()?;
    Ok(())
}