use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {

        let get_tensor = |name: &str| {
            let stensor = safetensor.tensor(name).expect("tensor is not exists");
            let shape = stensor.shape();
            // println!("{}", stensor.data().len());
            let data = stensor.data()
                                        .chunks(4)
                                        .map(|chunk| {
                                            let res = f32::from_ne_bytes(chunk.try_into().unwrap());
                                            return res;
                                        }).collect();
            let s = Vec::from(shape);
            return Tensor::new(data, &s);
        };

        let paras = LLamaParams { 
            w_gate: vec![get_tensor("model.layers.0.mlp.gate_proj.weight"), get_tensor("model.layers.1.mlp.gate_proj.weight")],
            w_up: vec![get_tensor("model.layers.0.mlp.up_proj.weight"), get_tensor("model.layers.1.mlp.up_proj.weight")],
            w_down: vec![get_tensor("model.layers.0.mlp.down_proj.weight"), get_tensor("model.layers.1.mlp.down_proj.weight")],
            lm_head: get_tensor("lm_head.weight"),
            wq: vec![get_tensor("model.layers.0.self_attn.q_proj.weight"), get_tensor("model.layers.1.self_attn.q_proj.weight")],
            wk: vec![get_tensor("model.layers.0.self_attn.k_proj.weight"), get_tensor("model.layers.1.self_attn.k_proj.weight")],
            wv: vec![get_tensor("model.layers.0.self_attn.v_proj.weight"), get_tensor("model.layers.1.self_attn.v_proj.weight")],
            wo: vec![get_tensor("model.layers.0.self_attn.o_proj.weight"), get_tensor("model.layers.1.self_attn.o_proj.weight")],

            rms_att_w: vec![get_tensor("model.layers.0.input_layernorm.weight"), get_tensor("model.layers.1.input_layernorm.weight")],
            rms_ffn_w: vec![get_tensor("model.layers.0.post_attention_layernorm.weight"), get_tensor("model.layers.1.post_attention_layernorm.weight")],
            rms_out_w: get_tensor("model.norm.weight"),
            embedding_table: get_tensor("lm_head.weight"),
        };
        return paras;
    }
}