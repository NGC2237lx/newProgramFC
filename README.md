# newProgramFC
- ## 项目说明
- 本项目是基于ACL 2023 Paper ["Fact-Checking Complex Claims with Program-Guided Reasoning"](https://arxiv.org/abs/2305.12744)的代码复现与改进，主要目的是通过生成推理程序来验证复杂声明的真伪和对潜在传播源的判定。
- ## 项目结构总览
```shell
.
│  
├─code
│   ├─  Analyze_emo_and_nt.py # 情感叙事分析
│   ├─  baseline1.py # 基线方法
│   ├─  getlabel.py # 获取Label
│   ├─  program_execution.py # 执行推理程序
│   └─  v1.0program_generator .py # 生成推理程序
│      
├─datasets
│   └─  weibo.json # 原始数据集
│      
└─result
│   ├─  accuracy_comparison.json # 判定Label与正确结果对比
│   ├─  baseline_results.json # 基线结果
│   ├─  emotion_narrative_analysis.json # 情感叙事分析
│   ├─  execute_program.json # 生成的推理程序
│   ├─  potential_propagators.json # 潜在传播源
│   └─  result.json # 执行结果
└─README.md # 说明文档
```

- ## 实验结果

| Model               | Accuracy | Real News Accuracy | Fake News Accuracy |
|---------------------|----------|---------------------|---------------------|
| BaseLine            | 57.32%  | 43.21%             | 72.68%             |
| newProgramFC (α=0.5, β=0.5) | 63.28%  | 57.54%             | 69.19%             |
