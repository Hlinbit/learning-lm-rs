use std::{clone, cmp::Ordering};

#[derive(Clone, Debug, Copy)]
pub struct PPP<K, V> {
    pub prob: K,
    pub position: V,
}

impl<K: PartialOrd, V> PartialOrd for PPP<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.prob.partial_cmp(&other.prob)
    }
}

impl<K: Ord, V> Ord for PPP<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.prob.cmp(&other.prob)
    }
}

impl<K: PartialEq, V> PartialEq for PPP<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.prob == other.prob
    }
}

impl<K: Eq, V> Eq for PPP<K, V> {}



pub struct MinHeap<T> {
    data: Vec<T>,
    max_length: u32,
}

impl<K: PartialOrd + Copy, V> MinHeap<PPP<K, V>>  {
    // 创建一个新的空堆
    pub fn new(max: u32) -> Self {
        MinHeap { data: Vec::new(), max_length: max }
    }

    // 获取父节点的索引
    fn parent_index(i: usize) -> usize {
        (i - 1) / 2
    }

    // 获取左子节点的索引
    fn left_child_index(i: usize) -> usize {
        2 * i + 1
    }

    // 获取右子节点的索引
    fn right_child_index(i: usize) -> usize {
        2 * i + 2
    }

    // 插入新元素到堆中
    pub fn push(&mut self, value: PPP<K, V>) {
        if self.len() >= self.max_length as usize {
            // 覆盖最大长度位置的元素
            self.data[self.max_length as usize - 1] = value;
            self.heapify_up();
        } else {
            self.data.push(value); // 将新元素添加到数组末尾
            self.heapify_up(); // 调整堆以保持小顶堆的性质
        }
    }

    // 自底向上调整堆（堆化）
    fn heapify_up(&mut self) {
        let mut index = self.data.len() - 1;
        while index > 0 {
            let parent = MinHeap::<PPP<K, V>>::parent_index(index);
            if self.data[index] < self.data[parent] {
                self.data.swap(index, parent);
                index = parent;
            } else {
                break;
            }
        }
    }

    // 删除并返回堆顶元素（最小值）
    pub fn pop(&mut self) -> Option<PPP<K, V>> {
        if self.data.is_empty() {
            return None;
        }
        let last_index = self.data.len() - 1;
        self.data.swap(0, last_index); // 将堆顶元素与最后一个元素交换
        let min_value = self.data.pop(); // 移除并保存最后一个元素（之前的堆顶）
        self.heapify_down(); // 自顶向下调整堆
        min_value
    }

    // 自顶向下调整堆（堆化）
    fn heapify_down(&mut self) {
        let mut index = 0;
        let len = self.data.len();
        loop {
            let left = MinHeap::<PPP<K, V>>::left_child_index(index);
            let right = MinHeap::<PPP<K, V>>::right_child_index(index);
            let mut smallest = index;

            if left < len && self.data[left] < self.data[smallest] {
                smallest = left;
            }
            if right < len && self.data[right] < self.data[smallest] {
                smallest = right;
            }
            if smallest != index {
                self.data.swap(index, smallest);
                index = smallest;
            } else {
                break;
            }
        }
    }

    // 获取堆中的元素数量
    fn len(&self) -> usize {
        self.data.len()
    }

    // 检查堆是否为空
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.max_length == self.data.len() as u32
    }

    pub fn peek(&self) -> Option<&PPP<K, V>> {
        if self.data.is_empty() {
            return None;
        }
        return self.data.first();
    }

    pub fn prob(&self) -> Vec<K>{
        let mut result = vec![];
        for pair in &self.data {
            result.push(pair.prob);
        }
        return result;
    }
}
