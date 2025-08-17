use candle_core::{DType, Device, Module, Result, Tensor, Var};
use candle_nn::{Init, Linear, Optimizer, SGD, VarBuilder, VarMap, linear, loss, ops};
use rand::prelude::SliceRandom;
use std::cmp;

fn print_varmap(varmap: &VarMap) {
    let data = varmap.data().lock().unwrap();
    let mut param_count = 0;
    for (key, value) in data.iter() {
        println!("{:?}", key);
        println!("{}", value);
        param_count += value.elem_count();
    }
    println!("param_count: {}", param_count);
}

#[test]
fn varmap_() -> Result<()> {
    let device = Device::Cpu;
    // 用于保存命名变量
    let varmap = VarMap::new();
    {
        let mut data = varmap.data().lock().unwrap();
        // 变量就是可变张量
        let var = Var::randn(0.0f32, 1.0f32, (2, 4), &device)?;
        data.insert("var1".to_string(), var);
        let var = Var::zeros((4, 3), DType::F32, &device)?;
        data.insert("var2".to_string(), var);
    }
    print_varmap(&varmap);
    // 命令变量的保存
    varmap.save("test.safetensors")?;
    Ok(())
}

#[test]
fn varmap_load() -> Result<()> {
    let device = Device::Cpu;
    let mut varmap = VarMap::new();
    {
        let mut data = varmap.data().lock().unwrap();
        let var = Var::randn(0.0f32, 1., (2, 4), &device)?;
        data.insert("var1".to_string(), var);
        let var = Var::randn(0.0f32, 1.0f32, (4, 3), &device)?;
        data.insert("var2".to_string(), var);
    }
    // 必须具有同等规模才能进行加载
    varmap.load("test.safetensors")?;
    print_varmap(&varmap);
    Ok(())
}

#[test]
fn varbuilder_() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    {
        let mut data = varmap.data().lock().unwrap();
        let var = Var::zeros((2, 4), DType::F32, &device)?;
        data.insert("weight".to_string(), var);
    }
    // 变量的构建器，用于在指定容器放置变量
    let mut vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    // 容器中有指定名称的变量就返回，没有就根据hints创建
    let tensor = vb.get_with_hints(
        (2, 4),
        "weight",
        Init::Randn {
            mean: 0.,
            stdev: 1.,
        },
    )?;
    print_varmap(&varmap);
    // 没有就生成
    vb.get((1, 4), "bias")?;
    print_varmap(&varmap);
    // println!("tensor dims: {:?}", tensor.get_on_dim(1,1).unwrap());
    Ok(())
}

struct SimpleModel {
    linear1: Linear,
    linear2: Linear,
    linear3: Linear,
}
impl SimpleModel {
    fn new(vb: VarBuilder, in_dim: usize, hidden_dim: usize, out_dim: usize) -> Result<Self> {
        let linear1 = linear(in_dim, hidden_dim, vb.pp("linear1"))?;
        let linear2 = linear(hidden_dim, hidden_dim, vb.pp("linear2"))?;
        let linear3 = linear(hidden_dim, out_dim, vb.pp("linear3"))?;
        Ok(Self {
            linear1,
            linear2,
            linear3,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.relu()?;
        let x = self.linear2.forward(&x)?;
        let x = x.relu()?;
        let x = self.linear3.forward(&x)?;
        Ok(x)
    }
}

trait Dataset {
    fn len(&self) -> Result<usize>;
    fn get_batch(&self, start: usize, end: usize) -> Result<(Tensor, Tensor)>;
    fn shuffle(&mut self) -> Result<()>;
}

pub struct DemoDataset {
    inputs: Tensor,
    targets: Tensor,
}

impl DemoDataset {
    pub fn new(inputs: Tensor, targets: Tensor) -> Result<Self> {
        Ok(Self { inputs, targets })
    }

    fn get_inx(&self, inx: usize) -> Result<(Tensor, Tensor)> {
        let x_inx = self.inputs.narrow(0, inx, 1)?;
        let y_inx = self.targets.narrow(0, inx, 1)?;
        Ok((x_inx, y_inx))
    }
}

impl Dataset for DemoDataset {
    fn len(&self) -> Result<usize> {
        Ok(self.inputs.dim(0)?)
    }
    fn get_batch(&self, start: usize, end: usize) -> Result<(Tensor, Tensor)> {
        let x_inx = self.inputs.narrow(0, start, end - start)?;
        let y_inx = self.targets.narrow(0, start, end - start)?;
        Ok((x_inx, y_inx))
    }
    // 打乱顺序
    fn shuffle(&mut self) -> Result<()> {
        let len = self.len()?;
        let mut indices: Vec<u32> = (0..len).map(|i| i as u32).collect();
        let mut rng = rand::rng();
        indices.shuffle(&mut rng);
        let indices_tensor = Tensor::from_vec(indices, (len,), self.inputs.device())?;
        self.inputs = self.inputs.index_select(&indices_tensor, 0)?;
        self.targets = self.targets.index_select(&indices_tensor, 0)?;
        Ok(())
    }
}

struct DatasetLoader<'a> {
    dataset: Box<dyn Dataset + 'a>,
    batch_size: usize,
    current_index: usize,
    shuffle: bool,
}
impl<'a> DatasetLoader<'a> {
    fn new<D: Dataset + 'a>(dataset: D, batch_size: usize, shuffle: bool) -> Result<Self> {
        Ok(Self {
            dataset: Box::new(dataset),
            batch_size,
            current_index: 0,
            shuffle,
        })
    }
    fn reset(&mut self) -> Result<()> {
        self.current_index = 0;
        if self.shuffle {
            self.dataset.shuffle()?;
        }
        Ok(())
    }
}

impl<'a> Iterator for DatasetLoader<'a> {
    type Item = Result<(Tensor, Tensor)>;
    fn next(&mut self) -> Option<Self::Item> {
        let start = self.current_index * self.batch_size;
        let end = cmp::min(start + self.batch_size, self.dataset.len().ok()?);
        if start >= end {
            return None;
        }
        let batch = self.dataset.get_batch(start, end).ok()?;
        self.current_index += 1;
        Some(Ok(batch))
    }
}

#[test]
fn linear_() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let mut vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = SimpleModel::new(vb, 2, 9, 2)?;
    // print_varmap(&varmap);
    let x_vec = vec![-1.2f32, 3.1, -0.9, 2.9, -0.5, 2.6, 2.3, -1.1, 2.7, -1.5];
    let x_train = Tensor::from_vec(x_vec, (5, 2), &device)?;
    let y_vec = vec![0u32, 0, 0, 1, 1];
    let y_train = Tensor::from_vec(y_vec, 5, &device)?;
    // let y_predict = model.forward(&x_train)?;
    // print_varmap(&varmap);
    let mut train_dataset = DemoDataset::new(x_train, y_train)?;
    let mut dataload = DatasetLoader::new(train_dataset, 1, true)?;
    dataload.reset()?;
    for (inx, batch) in dataload.enumerate() {
        println!("inx: {}", inx);
        let (x, y) = batch?;
        println!("batch: {}", x);
    }
    Ok(())
}

#[test]
fn train_() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let mut vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = SimpleModel::new(vb, 2, 20, 2)?;
    let x_train = Tensor::from_vec(
        vec![-1.2f32, 3.1, -0.9, 2.9, -0.5, 2.6, 2.3, -1.1, 2.7, -1.5],
        (5, 2),
        &device,
    )?;
    let y_train = Tensor::from_vec(vec![0u32, 0, 0, 1, 1], 5, &device)?;
    let train_dataset = DemoDataset::new(x_train, y_train)?;
    let x_val = Tensor::from_vec(vec![-0.8f32, 2.8, 2.6, -1.6], (2, 2), &device)?;
    let y_val = Tensor::from_vec(vec![0u32, 1], 2, &device)?;
    let val_dataset = DemoDataset::new(x_val, y_val)?;
    let mut train_dataload = DatasetLoader::new(train_dataset, 2, true)?;
    let mut val_dataload = DatasetLoader::new(val_dataset, 2, false)?;
    let mut sgd = SGD::new(varmap.all_vars(), 0.01)?;
    let epochs = 3;
    for epoch in 0..epochs {
        train_dataload.reset()?;
        val_dataload.reset()?;
        for batch in &mut train_dataload {
            let (x, y) = batch?;
            let predict = model.forward(&x)?;
            let loss = loss::cross_entropy(&predict, &y)?;
            sgd.backward_step(&loss)?;
            println!("epoch:{} train loss: {}", epoch, loss);
        }
        for batch in &mut val_dataload {
            let (x, y) = batch?;
            let predict = model.forward(&x)?;
            let loss = loss::cross_entropy(&predict, &y)?;
            println!("epoch:{} val loss: {}", epoch, loss);
        }
    }
    train_dataload.reset()?;
    val_dataload.reset()?;
    for batch in &mut train_dataload {
        let (x, y) = batch?;
        let predict = model.forward(&x)?;
        let softmax = ops::softmax(&predict, 1)?;
        let label = softmax.argmax(1)?;
        println!("train label: {}", label);
        println!("train targets:{}", y);
        println!(
            "train acc:{}",
            label
                .eq(&y)?
                .sum(0)?
                .to_dtype(DType::F32)?
                .affine(1.0 / (x.dim(0)? as f64), 0.0)?
        );
    }
    for batch in &mut val_dataload {
        let (x, y) = batch?;
        let predict = model.forward(&x)?;
        let softmax = ops::softmax(&predict, 1)?;
        let label = softmax.argmax(1)?;
        println!("val label: {}", label);
        println!("val targets:{}", y);
        println!(
            "val acc:{}",
            label
                .eq(&y)?
                .sum(0)?
                .to_dtype(DType::F32)?
                .affine(1.0 / (x.dim(0)? as f64), 0.0)?
        );
    }

    varmap.save("model.safetensors");

    Ok(())
}

#[test]
fn load_train_() -> Result<()> {
    let device = Device::Cpu;
    let mut varmap = VarMap::new();
    let mut vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = SimpleModel::new(vb, 2, 20, 2)?;
    let x_train = Tensor::from_vec(
        vec![-1.2f32, 3.1, -0.9, 2.9, -0.5, 2.6, 2.3, -1.1, 2.7, -1.5],
        (5, 2),
        &device,
    )?;
    let y_train = Tensor::from_vec(vec![0u32, 0, 0, 1, 1], 5, &device)?;
    let train_dataset = DemoDataset::new(x_train, y_train)?;
    let x_val = Tensor::from_vec(vec![-0.8f32, 2.8, 2.6, -1.6], (2, 2), &device)?;
    let y_val = Tensor::from_vec(vec![0u32, 1], 2, &device)?;
    let val_dataset = DemoDataset::new(x_val, y_val)?;
    let mut train_dataload = DatasetLoader::new(train_dataset, 5, true)?;
    let mut val_dataload = DatasetLoader::new(val_dataset, 2, false)?;
    varmap.load("model.safetensors");
    train_dataload.reset()?;
    val_dataload.reset()?;
    for batch in &mut train_dataload {
        let (x, y) = batch?;
        let predict = model.forward(&x)?;
        let softmax = ops::softmax(&predict, 1)?;
        let label = softmax.argmax(1)?;
        println!("train label: {}", label);
        println!("train targets:{}", y);
        println!(
            "train acc:{}",
            label
                .eq(&y)?
                .sum(0)?
                .to_dtype(DType::F32)?
                .affine(1.0 / (x.dim(0)? as f64), 0.0)?
        );
    }
    for batch in &mut val_dataload {
        let (x, y) = batch?;
        let predict = model.forward(&x)?;
        let softmax = ops::softmax(&predict, 1)?;
        let label = softmax.argmax(1)?;
        println!("val label: {}", label);
        println!("val targets:{}", y);
        println!(
            "val acc:{}",
            label
                .eq(&y)?
                .sum(0)?
                .to_dtype(DType::F32)?
                .affine(1.0 / (x.dim(0)? as f64), 0.0)?
        );
    }

    Ok(())
}
