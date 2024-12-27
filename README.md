# ProgramFC with emotion_narrative_analysis
- ## 自然语言处理期末大作业代码仓库
- ## 项目说明
 本项目是基于ACL 2023 Paper ["Fact-Checking Complex Claims with Program-Guided Reasoning"](https://arxiv.org/abs/2305.12744)的代码复现与改进，主要目的是通过生成推理程序来验证复杂声明的真伪和对潜在传播源的判定。
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

---

**实验检测正确与错误分类数量**  
| 模型                 | 正确真实新闻数 | 正确虚假新闻数 | 错误真实新闻数 | 错误虚假新闻数 |
|----------------------|----------------|----------------|----------------|----------------|
| BaseLine            | 2720           | 2386           | 2052           | 2242           |
| newProgramFC (α=0.5, β=0.5) | 2746           | 3202           | 2026           | 1426           |

---

**分类准确率对比**  
| 模型                 | 整体准确率 | 真实新闻准确率 | 虚假新闻准确率 |
|----------------------|------------|----------------|----------------|
| BaseLine            | 54.32%     | 57.00%         | 51.56%         |
| newProgramFC (α=0.5, β=0.5) | 63.28%     | 57.54%         | 69.19%         |

---

**真实新闻性能指标**  
| 模型                 | 精确率（Precision） | 召回率（Recall） | F1 分数（F1 Score） |
|----------------------|--------------------|------------------|---------------------|
| BaseLine            | 57.00%            | 54.82%          | 55.89%             |
| newProgramFC (α=0.5, β=0.5) | 57.54%            | 65.82%          | 61.40%             |

---

**虚假新闻性能指标**  
| 模型                 | 精确率（Precision） | 召回率（Recall） | F1 分数（F1 Score） |
|----------------------|--------------------|------------------|---------------------|
| BaseLine            | 51.56%            | 53.76%          | 52.64%             |
| newProgramFC (α=0.5, β=0.5) | 69.19%            | 61.25%          | 64.98%             |  

