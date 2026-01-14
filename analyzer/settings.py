# -*- coding: utf-8 -*-
import os

# =======================================================
# 🔐 核心配置 (Secrets & API)
# Core Configuration (Secrets & API)
# =======================================================

# [必须修改] 你的阿里云百炼 API Key
# [MUST MODIFY] Your Alibaba Cloud Bailian API Key
# 还没有 Key？请访问: https://bailian.console.aliyun.com/
# No Key yet? Visit: https://bailian.console.aliyun.com/
API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 

# API 接入点
# API Endpoint
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# [模型选择]
# [Model Selection]
# 推荐使用 qwen-plus (性价比高)，或者 qwen-max (逻辑更强)
# Recommended: qwen-plus (cost-effective) or qwen-max (stronger logic)
MODEL_NAME = "qwen-max"


# =======================================================
# ⚙️ 分析参数 (Analysis Parameters)
# =======================================================

# 阶段一 (关键词事实提取) 的随机性
# Stage 1 (Keyword Fact Extraction) temperature
TEMP_STAGE_1 = 0.3 

# 阶段二 (关键词关联分析) 的随机性
# Stage 2 (Correlation Analysis) temperature
TEMP_STAGE_2 = 0.5 

# 单次请求的最大文本长度限制 (防止超长/省钱)
# Maximum text length per request (prevent overflow/save cost)
MAX_TEXT_LENGTH = 25000 

# API 调用失败时的重试次数
# Retry count on API call failure
MAX_RETRIES = 3


# =======================================================
# 📂 文件路径配置 (File Paths)
# =======================================================

# [输入] 待分析的数据文件 (支持 CSV)
# [INPUT] Data file to analyze (CSV format)
# 请确保你的 CSV 文件名和这里一致
# Ensure your CSV filename matches here
INPUT_FILE = 'analyzer/data/context_posts.csv'

# [输入] Prompt 模板路径
# [INPUT] Prompt template path
PROMPT_DIR = 'analyzer/prompts'
PROMPT_SYSTEM_FILE = os.path.join(PROMPT_DIR, 'sys_prompt.txt')
PROMPT_STAGE1_FILE = os.path.join(PROMPT_DIR, 'keyword_prompt.txt')
PROMPT_STAGE2_FILE = os.path.join(PROMPT_DIR, 'correlation_prompt.txt')

# [输出] 结果保存路径
# [OUTPUT] Result save path
OUTPUT_STAGE1_CSV = 'analyzer/data/stage1_keyword_analysis.csv'        # 阶段一结果 / Stage 1 results
OUTPUT_STAGE2_MD = 'analyzer/data/stage2_correlation_analysis_report.md'   # 阶段二报告 / Stage 2 report
OUTPUT_FINAL_CONTEXT = 'analyzer/data/final_context_knowledge_base.txt' # 最终生成的背景库 / Final knowledge base

TEST_POST_LIST = 'analyzer/data/post_list.csv'  # 用于打标测试的微博列表 / Post list for labeling test
TEST_SAMPLE_SIZE = 30  # 抽取的测试条数 / Number of test samples

# [批量推理结果] 百炼批量推理的结果下载 URL（带临时访问凭证）
# [Batch Results] Batch inference result URL from Alibaba Qwen (with temporary credentials)
RESULT_URL = "your_url_here"  # 替换为你的批量推理结果 URL / Replace with your batch result URL