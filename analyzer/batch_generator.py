"""
微博打标系统 - 批量请求生成器
Weibo Labeling System - Batch Request Generator

该脚本生成 batch_list.jsonl 文件，用于阿里云百炼 API 的批量推理服务。
This script generates batch_list.jsonl for Alibaba Cloud Bailian batch inference API.

输出格式：每行是一个 JSON 对象，包含 OpenAI 兼容的请求格式
Output format: Each line is a JSON object in OpenAI-compatible batch format
"""

import pandas as pd
import json
import os
from . import settings


def generate_batch_list(csv_file=None, output_file='analyzer/data/batch_list.jsonl',
                        prompt_file='analyzer/prompts/labeling_prompt.txt',
                        api_key=None, model_name=None):
    """
    从 post_list.csv 生成批量推理请求文件 batch_list.jsonl
    Generate batch_list.jsonl from post_list.csv for batch inference
    
    参数 Parameters:
    - csv_file: CSV 文件路径（两列：id 和 微博正文）/ CSV file path (two columns: id and post content)
    - output_file: 输出 JSONL 文件路径 / Output JSONL file path
    - prompt_file: 标注 prompt 文件路径 / Path to labeling prompt file
    - api_key: 阿里云百炼 API Key（默认从 settings.py 读取）/ Alibaba Cloud API Key
    - model_name: 模型名称（默认从 settings.py 读取）/ Model name
    """
    
    # 使用 settings 中的配置作为默认值 / Use settings defaults if parameters not provided
    if api_key is None:
        api_key = settings.API_KEY
    if model_name is None:
        model_name = settings.MODEL_NAME
    if csv_file is None:
        csv_file = settings.TEST_POST_LIST
    
    try:
        # 读取 CSV 文件 / Read CSV file
        df = pd.read_csv(csv_file)
        
        # 验证列名 / Verify column names
        if 'id' not in df.columns or '微博正文' not in df.columns:
            raise ValueError(f"CSV 文件必须包含 'id' 和 '微博正文' 列 / CSV must have 'id' and '微博正文' columns")
        
        print(f"正在读取数据..." )
        print(f"Reading data from {csv_file}...")
        print(f"原始数据行数 / Original rows: {len(df)}")
        
        # 按 id 去重，保留第一次出现的行（保留正文） / Deduplicate by id, keep first occurrence
        df_deduped = df.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
        
        print(f"去重后行数 / Rows after deduplication: {len(df_deduped)}")
        print(f"删除重复数 / Duplicates removed: {len(df) - len(df_deduped)}")
        
        # 读取 system prompt / Read system prompt
        with open(prompt_file, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        print(f"总共 {len(df_deduped)} 条微博需要打标")
        print(f"Total {len(df_deduped)} posts to label")
        
        # 输出目录检查 / Check output directory
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成 batch_list.jsonl 文件 / Generate batch_list.jsonl
        print(f"\n正在生成批量推理请求文件...")
        print(f"Generating batch inference request file...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, (_, row) in enumerate(df_deduped.iterrows(), 1):
                post_id = str(row['id']).strip()
                content = str(row['微博正文']).strip()
                
                # 构造请求对象 / Construct request object (OpenAI-compatible format)
                # 格式参考 labeling_testing.py 中的逻辑
                request_body = {
                    "custom_id": post_id,  # 用于关联结果 / For result association
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": f"请对以下微博进行打标：\n{content}"
                            }
                        ],
                        "temperature": 0.3
                    }
                }
                
                # 写入 JSONL（每行一个完整的 JSON 对象）
                f.write(json.dumps(request_body, ensure_ascii=False) + '\n')
                
                if idx % 100 == 0:
                    print(f"  已生成 {idx} 条请求 / Generated {idx} requests")
        
        print(f"\n[OK] 批量请求文件已生成")
        print(f"[OK] Batch request file generated")
        print(f"  输出路径 / Output: {output_file}")
        print(f"  总请求数 / Total requests: {len(df_deduped)}")
        print(f"\n下一步 / Next step:")
        print(f"  1. 上传 {output_file} 到阿里云百炼批量推理服务")
        print(f"  2. Upload {output_file} to Alibaba Bailian batch service")
        print(f"  3. 等待任务完成后，运行 result_download_and_conversion.py")
        print(f"  4. After completion, run result_download_and_conversion.py")
        
        return len(df_deduped)
    
    except Exception as e:
        print(f"[ERROR] 生成失败 / Generation failed: {e}")
        print(f"[ERROR] {e}")
        return None


if __name__ == '__main__':
    # 示例使用 / Example usage
    print("=" * 60)
    print("微博打标系统 - 批量请求生成")
    print("Weibo Labeling System - Batch Request Generation")
    print("=" * 60)
    
    # 生成 batch_list.jsonl / Generate batch_list.jsonl
    count = generate_batch_list()
    
    if count:
        print(f"\n✓ 成功生成 {count} 条批量推理请求")
        print(f"✓ Successfully generated {count} batch inference requests")
