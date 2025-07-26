
本库行文逻辑、展示示例完全来自[《动手学深度学习》](https://zh.d2l.ai/index.html)，请对照原书进行学习。

本库使用 cargo doc 进行组织，相关章节函数只为组织层级，均无意义。
如有必要可能使用 jupyter notebook 进行重构，但目前考虑到在其 rust 的内核适配和生态问题尚不清晰，故暂时不做。

开始学习：
```shell
cargo doc --open
```


本教程使用库与python库的对照

| python库 | rust库 |
| --- | --- |
| numpy | ndarray |
| pandas | polars |
| PyTorch | candle |