use polars::prelude::*;
use std::{error::Error, fs::File, io::Write, path::Path};

/// ## 2.2.1 读取数据集
///
/// 举一个例子，我们首先创建一个人工数据集，并存储在CSV（逗号分隔值）文件 house_tiny.csv中。
/// 以其他格式存储的数据也可以通过类似的方式进行处理。
/// 下面我们将数据集按行写入CSV文件中。
///
///```rust
///    use std::{error::Error, fs::File, path::Path,io::Write};
///    let path = Path::new("output_std.csv");
///    let mut file = match File::create(&path) {
///        Err(why) => {
///            let display = path.display();
///            panic!("couldn't create {}: {:?}", display, why)
///        }
///        Ok(file) => file,
///    };
///    writeln!(file, "NumRooms,Alley,Price");
///    writeln!(file, ",Pave,127500");
///    writeln!(file, "2,,106000");
///    writeln!(file, "4,,178100");
///    writeln!(file, ",,140000");
///    file.flush()?;
/// ```
///
/// <br>
/// 默认按行写入，但如果数据来源按列写入更方便的话，建议使用polars的文件创建。
///
///```rust
///     use polars::prelude::*;
///     use std::{error::Error, fs::File, path::Path,io::Write};
///     let mut df: DataFrame = df!(
///        "NumRooms" => vec![None, Some(2), Some(4), None],
///        "Alley" => vec![Some("Pave"), None, None, None],
///        "Price" => [127500,106000 , 178100, 140000],
///    )
///    .unwrap();
///    let mut file = File::create("output.csv").expect("could not create file");
///    CsvWriter::new(&mut file)
///        .include_header(true)
///        .with_separator(b',')
///        .finish(&mut df)?;
/// ```
///
/// <br>
/// 要从创建的CSV文件中加载原始数据集，我们导入polars包(use polars::prelude::*;)并调用CsvReadOptions组织读取。
/// 该数据集有四行三列。其中每行描述了房间数量（“NumRooms”）、巷子类型（“Alley”）和房屋价格（“Price”）。
#[test]
pub fn reading_the_dataset() -> Result<(), Box<dyn Error>> {

    let mut df: DataFrame = df!(
        "NumRooms" => vec![None, Some(2), Some(4), None],
        "Alley" => vec![Some("Pave"), None, None, None],
        "Price" => [127500,106000 , 178100, 140000],
    )
    .unwrap();
    println!("{df}");
    let mut file = File::create("output.csv").expect("could not create file");
    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df)?;
    let df_csv = CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(CsvParseOptions::default().with_try_parse_dates(true))
        .try_into_reader_with_file_path(Some("output.csv".into()))?
        .finish()?;
    let df_csv = df_csv
        .clone()
        .lazy()
        .select([col("NumRooms")])
        .collect()?;
  

    
    
    println!("{df_csv}");


    Ok(())
}
