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

impl<T: Copy + Clone + Default> LLamaParams<T> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| {
            let stensor = safetensor.tensor(name).expect("tensor is not exists");
            let shape = stensor.shape();
            // println!("{}", stensor.data().len());
            let n_blocks = stensor.data().len() / std::mem::size_of::<T>();
            let data = unsafe { std::slice::from_raw_parts(stensor.data().as_ptr() as *const T, n_blocks) };
            let s = Vec::from(shape);
            return Tensor::new(data.to_vec(), &s);
        };

        let get_tensors = |template: &str, number: usize| {
            let mut result: Vec<Tensor<T>> = Vec::new();
            for i in 0..number {
                let name = format!("{}", template.replace("{}", &i.to_string()));
                result.push(get_tensor(&name));
            }
            result
        };

        let paras = LLamaParams {
            embedding_table: get_tensor("model.embed_tokens.weight"),
            wq: get_tensors("model.layers.{}.self_attn.q_proj.weight", config.num_hidden_layers),
            wk: get_tensors("model.layers.{}.self_attn.k_proj.weight", config.num_hidden_layers),
            wv: get_tensors("model.layers.{}.self_attn.v_proj.weight", config.num_hidden_layers),
            wo: get_tensors("model.layers.{}.self_attn.o_proj.weight", config.num_hidden_layers),
            rms_att_w: get_tensors("model.layers.{}.input_layernorm.weight", config.num_hidden_layers),
            w_gate: get_tensors("model.layers.{}.mlp.gate_proj.weight", config.num_hidden_layers),
            w_up: get_tensors("model.layers.{}.mlp.up_proj.weight", config.num_hidden_layers),
            w_down: get_tensors("model.layers.{}.mlp.down_proj.weight", config.num_hidden_layers),
            rms_ffn_w: get_tensors("model.layers.{}.post_attention_layernorm.weight", config.num_hidden_layers),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        };
        return paras;
    }
}