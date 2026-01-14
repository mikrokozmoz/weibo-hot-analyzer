# 基于Qwen的热门微博分析工具 | *Qwen-Based Weibo Hot Post Analyzer*

本工具利用开源的Qwen大模型对热门微博进行分析。具体功能包括爬虫结果的读取和预处理、生成AI批量推理所需的背景知识Prompt、随机抽取博文内容进行本地API测试、生成百炼批量处理所需的JSON文件、以及下载并分析批量处理结果文件。更多功能仍在研发中。参考研究项目正在进行中。

*You can use this Qwen-based tool to analyze hot Weibo posts. Features include loading and pre-processing posts, generating background information prompt for batch labeling, testing with local API on randomnized post samples, generating the batch JSON file for batch inference, and downloading and analyzing inference results. More features are in progress, as well as an example analysis project.*

## 1. 项目结构 | *Project Structure*

```
weibo-hot-analyzer/
├── analyzer/                          # AI 驱动分析模块 / AI Analysis Module
│   ├── __init__.py
│   ├── settings.py                   # AI 分析参数配置 / AI parameters
│   ├── summary.py                    # 两阶段分析脚本 / Two-stage analysis script
│   ├── labeling_testing.py           # 打标测试工具 / Labeling testing tool
│   ├── batch_generator.py            # 批量推理文件生成器 / Batch inference file generator
│   ├── result_download_and_conversion.py  # 批量推理结果处理 / Batch result processor
│   ├── data/                         # 数据目录 / Data directory
│   │   ├── batch_example.jsonl       # 批量推理示例文件 / Batch inference example
│   │   ├── batch_list.jsonl          # 生成的批量请求文件 / Generated batch requests
│   │   ├── batch_results_raw.jsonl   # 原始批量推理结果 / Raw batch results
│   │   ├── batch_results_final.csv   # 处理后的三列结果 / Processed three-column results
│   │   └── batch_results_final_expanded.csv  # 标签展开版本 / Expanded label version
│   └── prompts/                      # Prompt 模板文件 / Prompt templates
│       ├── sys_prompt.txt            # 系统 prompt / System prompt
│       ├── keyword_prompt.txt        # 关键词分析 prompt / Keyword analysis prompt
│       ├── correlation_prompt.txt    # 关联分析 prompt / Correlation analysis prompt
│       └── labeling_prompt.txt       # 打标 prompt / Labeling prompt
│
├── utils/                             # 数据处理模块 / Data Processing Module
│   ├── __init__.py
│   ├── __main__.py                   # 命令行接口 / CLI interface
│   ├── settings.py                   # 处理参数配置 / Processing parameters
│   └── data_processing.py            # 入口函数 / Entry function
│
├── processing/                        # Submodule: 数据处理库
│   └── post_analysis/
│       ├── __init__.py
│       ├── pre_processing.py         # 数据加载、去重、话题提取
│       └── corpus_analysis.py        # 分词、词频、词云     
│
├── requirements.txt                   # 项目依赖 / Dependencies
├── README.md                          # 项目说明（本文件）
└── .gitmodules                        # Submodule 配置
```

---

## 2. 工具安装 | *Installation*

### 2.a 克隆项目 | *Clone The Repository*

```bash
git clone --recursive https://github.com/mikrokozmoz/weibo-hot-analyzer
cd weibo-hot-analyzer
```

### 2.b 创建虚拟环境（可选但推荐） | *Create Virtual Environment (Optional But Recommended)*

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 2.c 安装依赖 | *Install Requirements*

```bash
pip install -r requirements.txt
```

---

## 3. 数据预处理 | *Data Pre-Processing*

### 3.a 功能说明 | *Feature Explanation*
- 自动判定结果文件为嵌套目录结构还是单独文件、读取、并合并
- 根据用户输入的判定规则去重
- 根据互动热度和内容长度提取关键博文，以便生成背景知识提示词

### 3.b 在Jupyter Notebook中进行 | *In Jupyter Notebook*

```python
# 加载工具包
# load library
import numpy as np 
import pandas as pd

from processing.post_analysis import load_posts_from_folder
from processing.post_analysis import dedupe_posts
from processing.post_analysis.pre_processing import get_context_posts
from analyzer.labeling_testing import update_labeling_prompt

# 加载并另存数据
# load and save a copy of the combined data
# 注意替换路径为你自己的目录和文件路径
# remember to replace the placeholder with your own directory and file path
df = load_posts_from_folder('your_file_path', keyword_column='关键词')
df.to_csv('your_path_and_file.csv', index=False, encoding='utf-8-sig')

# 博文内容去重并另存数据
# deduplicate posts and save a copy of the results
df_deduped = dedupe_posts(df, 
                          keyword_col='关键词',         # 标识关键词所在列名 | locate the column with keywords
                          text_col='微博正文_cleaned',  # 标识微博正文所在列 | locate the column with posts
                          time_col='发布时间',          # 标识发布时间所在列 | locate the column with posting time
                          sum_cols=None,               # 标识具体需要的去重后求和列，选择None则自动选择所有转赞评列 | locate the column to sum the interactions; default is all reposts, likes, and comments if left none
                          similarity_threshold=0.88,   # 输入想要的相似度，说明见下文 | desired similarity threshold, see explanations below
                          min_len_for_similarity=6,    # 输入判定相似需要的最小文本长度，说明见下文 | input minimum length for posts to be identified as similar, see explanations below
                          debug=False,                 # 是否需要开启debug模式以获取运行日志 | whether to turn on debug mode for detailed logs
                          debug_pairs_path=None,        
                          auto_clean=True              # 是否开启自动文本清理，说明见下文 | whether to turn on the auto post cleaning, see explanations below
                          )

# 注意替换路径为你自己的目录和文件路径
# remember to replace the placeholder with your own directory and file path
df_deduped.to_csv('your_path_and_file.csv', index=False, encoding='utf-8-sig')

# 提取生成背景知识提示词所需的博文
# extract posts needed to generate labeling backgound prompt
df_context = get_context_posts(df_deduped,                  
                               keyword_col='关键词',        # 标识关键词所在列名 | locate the column with keywords
                               interaction_col='互动总数',  # 标识博文热度所在列名 | locate the column for interactions as basis
                               text_col='微博正文',         # 标识微博正文所在列 | locate the column with posts
                               # 这里推荐不要用清洗后的 | recommend to use posts NOT cleaned
                               top_n=20                     # 输入你需要的每个关键词下需要的博文条数 | input the number of posts needed for your prompt generation
                               )

# 注意替换路径为你自己的目录和文件路径
# remember to replace the placeholder with your own directory and file path
df_context.loc[:,['id', '微博正文', '关键词']].to_csv('your_path_and_file.csv', index=False, encoding='utf-8-sig')
```

---

## 4. 使用百炼的批量推理工具 | *Use Bailian Batch Inference*

### 4.a 调用Qwen API生成批量推理所需要的背景信息 | *Use Qwen API to Generate Background Information for Batch Inference*

在终端中运行，结果文件会存储为: | *run in terminal and the ouput files will be saved as:*
- `analyzer/data/stage1_keyword_analysis.csv`
- `analyzer/data/stage2_correlation_analysis_report.md`
- `analyzer/data/final_context_knowledge_base.txt`
- 你可以自己指定存储路径，具体请更新`analyzer/settings.py`中的`OUTPUT_STAGE1_CSV`、`OUTPUT_STAGE2_MD`、`OUTPUT_FINAL_CONTEXT`
- *You can also have your own path; for details, please update `OUTPUT_STAGE1_CSV`, `OUTPUT_STAGE2_MD`, and `OUTPUT_FINAL_CONTEXT` in `analyzer/settings.py`*

```bash
# 记得在analyzer/settings.py里面更新你的上一步存储的文件路径
# remember to update the file you saved in the previous step
# 具体参数名字为INPUT_FILE
# parameter name is INPUT_FILE
python -m analyzer.summary
```

### 4.b 更新批量推理所需的提示词，在Notebook中进行 | *Update Prompts for Batch Inference, Run in Jupyter Notebook*

```python
# 更新生成背景知识所需的提示词 
# update prompt to generate background information
# 如果你有另存的文件路径记得更新 
# update the path and file name if you have saved your own
update_labeling_prompt(context_file='analyzer/data/stage2_correlation_analysis_report.md',
                        prompt_file='analyzer/prompts/labeling_prompt.txt')

# 另存测试和批量测试所需的数据节选
# save a subset of dataset needed for testing and batch inference
post_list = df_deduped.loc[:, ['id', '微博正文']]
post_list.to_csv('your_path_and_file.csv', index=False, encoding='utf-8-sig')
```

### 4.c 抽样测试批量推理 | *Testing Batch Inference with Samples*

在终端中运行，结果文件会存储为： | *run in terminal and the ouput files will be saved as:*
- `labeling_test_results.csv` 
- `labeling_test_results_expanded.csv`

```bash
# --n 30 表示你要随机抽取30条来进行测试 | means you will sample randomly 30 posts for testing
# --random_seed 101 随机种子，如果你需要抽取不同的博文进行测试，请使用不同的种子 | please use different seeds if you want a new sampling
python -m analyzer.labeling_testing --n 30 --random_seed 101 

# 你也可以在analyzer/settings.py的TEST_SAMPLE_SIZE里面更新样本数量 | you can also specify your own sample size as TEST_SAMPLE_SIZE in analyzer/settings.py
# 更新后可以只运行 | you can run only the following if you update the size in settings.py
python -m analyzer.labeling_testing --random_seed 101 
```

### 4.d 生成批量推理所需的JSON文件 | *Generate JSON File Needed for Batch Inference*

```bash
# 在终端中运行 | run in terminal
python -m analyzer.batch_generator

# 结果会存储为analyzer/data/batch_list.jsonl
# result will be saved as analyzer/data/batch_list.jsonl
```

### 4.e 上传文件到百炼批量推理并进行推理工作 | *Create Batch Inference Task with the File*
