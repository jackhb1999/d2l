use candle_core::{DType, Device, Module, Result, Tensor, Var};
use candle_nn::{Init, Linear, VarBuilder, VarMap, linear};

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

#[test]
fn linear_() -> Result<()> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let mut vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = SimpleModel::new(vb, 2, 9, 1)?;
    print_varmap(&varmap);
    // let x = Tensor::randn(0.0f32, 1.0f32, (1, 2), &device)?;
    // let y = model.forward(&x)?;
    Ok(())
}
