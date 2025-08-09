use candle_core::{DType, Device, Result, Var};
use candle_nn::VarMap;

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
        let var = Var::zeros((4, 3),DType::F32, &device)?;
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
