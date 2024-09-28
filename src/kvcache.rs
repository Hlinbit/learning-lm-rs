use std::{usize, vec};

use crate::tensor::Tensor;
use std::path::{PathBuf, Path};
use bytemuck::Pod;
use num_traits::Float;
use half::f16;
use std::fs::{File, OpenOptions};
use std::io::{self, Error, Read, Seek, SeekFrom, Write};

const BLOCK_SIZE: usize = 128;
const PREFIX: &str = "/tmp/engine";


fn write_head(filepath: &str, length: usize, offset: &mut usize) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)       // 允许写入
        .create(true)      // 如果文件不存在则创建
        .open(filepath)?;  // 打开文件

    file.seek(SeekFrom::Start(*offset as u64))?;
    let data_len_bytes = length.to_le_bytes();  // Little-endian representation
    *offset += data_len_bytes.len();
    file.write_all(&data_len_bytes)?;

    Ok(())
}

fn write_data<T: Pod>(filepath: &str, data_type: &str, data: &[T], offset: &mut usize) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)       // 允许写入
        .create(true)      // 如果文件不存在则创建
        .open(filepath)?;  // 打开文件

    file.seek(SeekFrom::Start(*offset as u64))?;
    // 写入 'T' 字符
    file.write_all(b"T")?;
    *offset += 1;

    // 写入类型标识符 (如 "f16" 或 "f32")，以 '\0' 结尾
    let mut type_bytes = data_type.as_bytes().to_vec();
    type_bytes.push(0);  // Append '\0'
    *offset += type_bytes.len();
    file.write_all(&type_bytes)?;

    // 写入 'L' 字符
    file.write_all(b"L")?;
    *offset += 1;

    // 写入数据长度 (4 字节的无符号整数)
    let data_len = (data.len() * std::mem::size_of::<T>()) as u32;
    let data_len_bytes = data_len.to_le_bytes();  // Little-endian representation
    *offset += data_len_bytes.len();
    file.write_all(&data_len_bytes)?;

    // 写入实际的二进制数据
    let data_bytes = bytemuck::cast_slice(data);  // 将泛型数据转换为字节流
    *offset += data_bytes.len();
    file.write_all(data_bytes)?;

    Ok(())
}

fn read_head(filepath: &str, offset: &mut usize) -> io::Result<usize> {
    // 读取数据长度 (4 字节的无符号整数)
    let mut file = File::open(filepath)?;
    file.seek(SeekFrom::Start(*offset as u64))?;
    let mut data_len_bytes = [0u8; std::mem::size_of::<usize>()];
    file.read_exact(&mut data_len_bytes)?;
    let data_len = usize::from_le_bytes(data_len_bytes);
    *offset += data_len_bytes.len();

    return Ok(data_len);

}

fn read_data<T: Pod>(filepath: &str, buffer: &mut [T], offset: &mut usize) -> io::Result<usize> {
    let mut file = File::open(filepath)?;
    
    file.seek(SeekFrom::Start(*offset as u64))?;
    // 读取 'T' 字符
    let mut t = [0u8; 1];
    file.read_exact(&mut t)?;
    if t[0] != b'T' {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Expected 'T'"));
    }
    *offset += 1;

    // 读取类型标识符，直到遇到 '\0'
    let mut type_bytes = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        file.read_exact(&mut byte)?;
        if byte[0] == 0 {
            type_bytes.push(byte[0]);
            break;
        }
        type_bytes.push(byte[0]);
    }
    *offset += type_bytes.len();
    let data_type = String::from_utf8(type_bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid UTF-8 in type"))?;

    // 读取 'L' 字符
    let mut l = [0u8; 1];
    file.read_exact(&mut l)?;
    if l[0] != b'L' {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Expected 'L'"));
    }
    *offset += 1;

    // 读取数据长度 (4 字节的无符号整数)
    let mut data_len_bytes = [0u8; 4];
    file.read_exact(&mut data_len_bytes)?;
    let data_len = u32::from_le_bytes(data_len_bytes) as usize;
    *offset += data_len_bytes.len();

    let buffer_size_in_bytes = buffer.len() * std::mem::size_of::<T>();
    if data_len > buffer_size_in_bytes {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "Data length {} exceeds buffer size {}",
                data_len, buffer_size_in_bytes
            ),
        ));
    }

    // 如果数据长度小于缓冲区大小，只填充前 data_len 字节
    let buffer_as_u8: &mut [u8] = bytemuck::cast_slice_mut(buffer);
    file.read_exact(&mut buffer_as_u8[..data_len])?;
    *offset += data_len;

    Ok(data_len / std::mem::size_of::<T>())
}

fn file_exists(filepath: &str) -> bool {
    Path::new(filepath).exists()
}

fn cache_exist(name: &str) -> bool {
    let k_store_dir = PathBuf::from(PREFIX).join(format!("{}{}", name, "_kcache.bin"));
    let v_store_dir = PathBuf::from(PREFIX).join(format!("{}{}", name, "_vcache.bin"));
    file_exists(k_store_dir.to_str().unwrap()) && file_exists(v_store_dir.to_str().unwrap())
}

pub struct CacheManager<T>{
    pub cache: Option<KVCache<T>>,
    pub current_name: String,
    layer: usize,
    max_seq_len: usize,
    dim: usize
}

impl<T: Default + Copy + Pod + Float> CacheManager<T> {

    pub fn new(layer: usize, max_seq_len: usize, dim: usize) -> Self {
        CacheManager {
            cache: None,
            current_name: "".to_string(),
            layer: layer,
            max_seq_len: max_seq_len,
            dim: dim
        }
    }

    pub fn set_name(&mut self, name: &str) {
        self.current_name = String::from(name);
    }

    pub fn store(&self, data_type: &str) -> io::Result<()>
    {
        let k_store_dir = PathBuf::from(PREFIX).join(format!("{}{}", self.current_name, "_kcache.bin"));
        let v_store_dir = PathBuf::from(PREFIX).join(format!("{}{}", self.current_name, "_vcache.bin"));

        let mut k_offset = 0usize;
        let mut v_offset = 0usize;

        if let Some(cache) = self.cache.as_ref() {
            let k_tensors = cache.all_k_cache();
            let v_tensors = cache.all_v_cache();
            write_head(k_store_dir.to_str().unwrap(), cache.length, &mut k_offset)?;
            write_head(v_store_dir.to_str().unwrap(), cache.length, &mut v_offset)?;
            for i in 0..self.layer {
                let data = k_tensors[i].data();
                write_data(k_store_dir.to_str().unwrap(), data_type, data, &mut k_offset)?;

                let data = v_tensors[i].data();
                write_data(v_store_dir.to_str().unwrap(), data_type, data, &mut v_offset)?;
            }
            Ok(())
        } else {
            Err(io::Error::new(io::ErrorKind::InvalidData, "cache is None"))
        }

        
    }

    pub fn load(&mut self) -> io::Result<()> 
    {
        let mut k_offset = 0usize;
        let mut v_offset = 0usize;

        let mut k_tensors: Vec<Tensor<T>> = Vec::new();
        let mut v_tensors: Vec<Tensor<T>> = Vec::new();
        if  self.cache.is_none() {
            self.cache = Some(KVCache::empty(self.max_seq_len, self.dim, 0))
        }

        let k_store_dir = PathBuf::from(PREFIX).join(format!("{}{}", self.current_name, "_kcache.bin"));
        let v_store_dir = PathBuf::from(PREFIX).join(format!("{}{}", self.current_name, "_vcache.bin"));

        let cache = self.cache.as_mut().unwrap();

        if cache_exist(&self.current_name) {
            cache.length = read_head(k_store_dir.to_str().unwrap(), &mut k_offset)?;
            read_head(v_store_dir.to_str().unwrap(), &mut v_offset)?;
            for _ in 0..self.layer {
                let mut k_buffer = vec![T::zero(); self.max_seq_len * self.dim];
                read_data(k_store_dir.to_str().unwrap(), &mut k_buffer, &mut k_offset)?;
                k_tensors.push(Tensor::new(k_buffer, &vec![self.max_seq_len, self.dim]));

                let mut v_buffer = vec![T::zero(); self.max_seq_len * self.dim];
                read_data(v_store_dir.to_str().unwrap(), &mut v_buffer, &mut v_offset)?;
                v_tensors.push(Tensor::new(v_buffer, &vec![self.max_seq_len, self.dim]));
            }
            cache.k_cache = k_tensors;
            cache.v_cache = v_tensors;
        } else {
            cache.set_k_cache((0..self.layer)
                        .map(|_| Tensor::default(&vec![self.max_seq_len, self.dim]))
                        .collect());
            cache.set_v_cache((0..self.layer)
                        .map(|_| Tensor::default(&vec![self.max_seq_len, self.dim]))
                        .collect());
        }
        Ok(())
    }
}

pub struct KVCache<T> {
    k_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    #[allow(unused)]
    max_seq_len: usize,
    dim: usize,
    length: usize, // length of the current sequence
}

impl<T: Default + Copy> KVCache<T> {
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            max_seq_len: max_seq_len,
            dim: dim,
            length: init_len,
        }
    }

    pub fn empty(max_seq_len: usize, dim: usize, length: usize) -> Self {
        KVCache {
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            max_seq_len: max_seq_len,
            dim: dim,
            length: length,
        }
    }

    pub fn set_k_cache(&mut self, cache: Vec<Tensor<T>>) {
        self.k_cache = cache;
    }

    pub fn set_v_cache(&mut self, cache: Vec<Tensor<T>>) {
        self.v_cache = cache;
    }

    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.k_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.v_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    pub fn all_k_cache(&self) -> &Vec<Tensor<T>> {
        self.k_cache.as_ref()
    }

    pub fn all_v_cache(&self) -> &Vec<Tensor<T>> {
        self.v_cache.as_ref()
    }

    pub fn dorp_block(&mut self) {
        let offset = self.dim * BLOCK_SIZE;
        let total_size = (self.length * self.dim) as i32;
        let move_size = total_size - offset as i32;
        if move_size < 0 {
            self.length = 0;
            return;
        }
        for l in 0..self.k_cache.len() {
            let k_data = unsafe {
                self.k_cache[l].data_mut()
            };
            let v_data = unsafe {
                self.v_cache[l].data_mut()
            };
            for i in 0..move_size as usize {
                k_data[i] = k_data[offset + i];
                v_data[i] = v_data[offset + i]
            }
        }
        self.length -= BLOCK_SIZE;

    }

    pub fn increment(&mut self, seq_len: usize){
        while self.length + seq_len > self.max_seq_len {
            self.dorp_block();
        }
        self.length += seq_len;
    }

    pub fn len(&self) -> usize {
        self.length
    }
}
