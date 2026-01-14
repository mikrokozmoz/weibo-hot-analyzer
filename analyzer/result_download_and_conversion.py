"""
微博打标系统 - 批量推理结果下载与转换
Weibo Labeling System - Batch Inference Result Download & Conversion

从阿里云百炼批量推理服务下载结果 JSONL，与原始数据关联，
生成 id、content、label 三列 CSV，并支持 JSON 拆解为单独列。
Download batch inference results from Alibaba Bailian, associate with original data,
generate CSV with id, content, label, and support JSON parsing to separate columns.
"""

import pandas as pd
import json
import requests
import os
from . import settings


def download_batch_results(result_url=None, output_file='analyzer/data/batch_results_raw.jsonl'):
    """
    从阿里云百炼批量推理 URL 下载结果 JSONL 文件
    Download batch inference result JSONL from Alibaba Bailian URL
    
    参数 Parameters:
    - result_url: 批量推理结果 URL（默认从 settings.RESULT_URL 读取）/ Batch result URL
    - output_file: 保存的原始 JSONL 文件路径 / Path to save raw JSONL
    
    返回 Returns:
    - 下载成功返回 True，否则返回 False / True if successful, False otherwise
    """
    
    if result_url is None:
        result_url = settings.RESULT_URL
    
    try:
        print("正在从阿里云百炼下载批量推理结果...")
        print("Downloading batch inference results from Alibaba Bailian...")
        
        # 输出目录检查 / Check output directory
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 下载 JSONL 文件 / Download JSONL file
        response = requests.get(result_url, timeout=30)
        response.raise_for_status()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # 统计行数 / Count lines
        with open(output_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        print(f"[OK] 下载完成，包含 {line_count} 条结果")
        print(f"[OK] Download complete with {line_count} results")
        print(f"  保存路径 / Saved to: {output_file}")
        
        return True
    
    except Exception as e:
        print(f"[ERROR] 下载失败 / Download failed: {e}")
        return False


def process_batch_results(raw_jsonl_file='analyzer/data/batch_results_raw.jsonl',
                         csv_file=None,
                         output_csv='analyzer/data/batch_results_final.csv'):
    """
    处理批量推理结果 JSONL，与去重后的原始 CSV 关联，生成最终的 id、content、label 三列 CSV
    Process batch inference results JSONL, associate with deduplicated original CSV,
    generate final CSV with id, content, label columns
    
    参数 Parameters:
    - raw_jsonl_file: 从百炼下载的原始 JSONL 文件 / Raw JSONL from Alibaba
    - csv_file: 原始 post_list.csv 路径（包含 id 和 微博正文）/ Original post_list.csv
    - output_csv: 输出 CSV 文件路径 / Output CSV file path
    
    返回 Returns:
    - 处理成功返回 DataFrame，否则返回 None / DataFrame if successful, None otherwise
    """
    
    if csv_file is None:
        csv_file = settings.TEST_POST_LIST
    
    try:
        print("\n正在处理批量推理结果...")
        print("Processing batch inference results...")
        
        # 读取原始 CSV / Read original CSV
        df_original = pd.read_csv(csv_file)
        if 'id' not in df_original.columns or '微博正文' not in df_original.columns:
            raise ValueError("CSV 文件必须包含 'id' 和 '微博正文' 列")
        
        print(f"  原始 CSV 行数 / Original CSV rows: {len(df_original)}")
        
        # 按 id 去重，保留第一次出现的行（与 batch_generator 保持一致）
        # Deduplicate by id, keep first occurrence (consistent with batch_generator)
        df_deduped = df_original.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
        
        print(f"  去重后行数 / Deduplicated rows: {len(df_deduped)}")
        print(f"  删除重复数 / Duplicates removed: {len(df_original) - len(df_deduped)}")
        
        # 建立 id->content 映射 / Build id->content mapping
        id_content_map = dict(zip(df_deduped['id'].astype(str), df_deduped['微博正文'].astype(str)))
        
        # 读取百炼返回的 JSONL 结果 / Read Alibaba results JSONL
        # 百炼返回格式：{"custom_id": "...", "result": {"message": {"content": "..."}}, ...}
        results = []
        error_count = 0
        
        # 先读第一行看看结构 / Read first line to check structure
        with open(raw_jsonl_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                try:
                    first_data = json.loads(first_line)
                    print(f"\n  样本数据结构（第1行）/ Sample structure (line 1):")
                    print(f"    顶级键 / Top keys: {list(first_data.keys())}")
                    if 'result' in first_data:
                        print(f"    result 键 / result keys: {list(first_data['result'].keys()) if isinstance(first_data['result'], dict) else 'non-dict'}")
                    if 'body' in first_data:
                        print(f"    body 键 / body keys: {list(first_data['body'].keys()) if isinstance(first_data['body'], dict) else 'non-dict'}")
                except:
                    pass
        
        with open(raw_jsonl_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    
                    # 提取 custom_id (对应原始的 post_id) 和标注结果 / Extract custom_id and labeling result
                    custom_id = str(data.get('custom_id', ''))
                    
                    # 尝试多个可能的路径来提取标注结果 / Try multiple possible paths to extract label
                    message_content = ''
                    
                    # 路径 1: response.body.choices[0].message.content（百炼的实际结构）
                    # Path 1: response.body.choices[0].message.content (Alibaba actual structure)
                    if 'response' in data and isinstance(data['response'], dict):
                        response_obj = data['response']
                        if 'body' in response_obj and isinstance(response_obj['body'], dict):
                            body_obj = response_obj['body']
                            if 'choices' in body_obj and isinstance(body_obj['choices'], list) and len(body_obj['choices']) > 0:
                                choice = body_obj['choices'][0]
                                if isinstance(choice, dict) and 'message' in choice:
                                    message_obj = choice['message']
                                    if isinstance(message_obj, dict) and 'content' in message_obj:
                                        message_content = message_obj['content']
                    
                    # 路径 2: result.message.content（备用）/ Path 2: result.message.content (backup)
                    if not message_content and 'result' in data and isinstance(data['result'], dict):
                        result_obj = data['result']
                        if 'message' in result_obj and isinstance(result_obj['message'], dict):
                            message_content = result_obj['message'].get('content', '')
                    
                    # 从映射表中获取原始微博内容 / Get original post content from mapping
                    content = id_content_map.get(custom_id, '')
                    
                    results.append({
                        'id': custom_id,
                        'content': content,
                        'label': message_content  # 保存原始 JSON 字符串 / Save raw JSON string
                    })
                    
                    if idx <= 3:
                        print(f"  样本结果 {idx}: custom_id={custom_id}, label_length={len(message_content)}")
                    
                    if idx % 100 == 0:
                        print(f"  已处理 {idx} 条结果 / Processed {idx} results")
                
                except json.JSONDecodeError as e:
                    error_count += 1
                    print(f"  行 {idx} JSON 格式错误 / Line {idx} JSON decode error: {e}")
                    continue
        
        if not results:
            raise ValueError("没有成功解析的结果 / No successful results parsed")
        
        # 转换为 DataFrame / Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # 输出目录检查 / Check output directory
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存结果 CSV（与 labeling_test_results.csv 格式一致）
        # Save results CSV (same format as labeling_test_results.csv)
        results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        
        print(f"\n[OK] 结果已保存到 {output_csv}")
        print(f"[OK] Results saved to {output_csv}")
        print(f"  总行数 / Total rows: {len(results_df)}")
        print(f"  成功处理 / Successful: {len(results)} 条")
        print(f"  失败 / Failed: {error_count} 条")
        
        return results_df
    
    except Exception as e:
        print(f"[ERROR] 处理失败 / Processing failed: {e}")
        return None


def expand_labels(csv_file='analyzer/data/batch_results_final.csv',
                  output_expanded='analyzer/data/batch_results_final_expanded.csv'):
    """
    将 label 列的 JSON 拆解为单独的列（与 labeling_testing.py 逻辑一致）
    Parse JSON in label column and generate expanded version (same as labeling_testing.py)
    
    参数 Parameters:
    - csv_file: 包含 label 列的 CSV 文件 / CSV file with label column
    - output_expanded: 输出展开后的 CSV 文件 / Output expanded CSV file
    
    返回 Returns:
    - 展开成功返回 DataFrame，否则返回 None / DataFrame if successful, None otherwise
    """
    
    try:
        print("\n正在拆解标签 JSON...")
        print("Expanding label JSON to separate columns...")
        
        # 读取结果 CSV / Read results CSV
        df = pd.read_csv(csv_file)
        
        if 'label' not in df.columns:
            raise ValueError("CSV 文件必须包含 'label' 列")
        
        # 解析 JSON 并展开 / Parse JSON and expand
        df['label_parsed'] = df['label'].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
        df_expanded = pd.json_normalize(df['label_parsed'])
        
        # 合并 id、content 和展开的标签列 / Merge id, content, and expanded label columns
        df_final = pd.concat([df[['id', 'content']], df_expanded], axis=1)
        
        # 输出目录检查 / Check output directory
        output_dir = os.path.dirname(output_expanded)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存展开版本 / Save expanded version
        df_final.to_csv(output_expanded, index=False, encoding='utf-8-sig')
        
        print(f"[OK] 展开版本已保存到 {output_expanded}")
        print(f"[OK] Expanded version saved to {output_expanded}")
        print(f"  列数 / Columns: {len(df_final.columns)}")
        
        return df_final
    
    except Exception as e:
        print(f"[ERROR] 拆解失败 / Expansion failed: {e}")
        return None


if __name__ == '__main__':
    # 执行流程 / Execution flow
    print("=" * 70)
    print("微博打标系统 - 批量推理结果处理")
    print("Weibo Labeling System - Batch Inference Result Processing")
    print("=" * 70)
    
    # 1. 下载结果 / Step 1: Download results
    success = download_batch_results()
    if not success:
        print("无法继续，请检查 RESULT_URL 是否正确")
        print("Cannot continue, please check if RESULT_URL is correct")
        exit(1)
    
    # 2. 处理结果并与原始数据关联 / Step 2: Process results and associate with original data
    results_df = process_batch_results()
    if results_df is None:
        print("无法继续，请检查结果格式")
        print("Cannot continue, please check result format")
        exit(1)
    
    # 3. 展开 label JSON（可选）/ Step 3: Expand label JSON (optional)
    expanded_df = expand_labels()
    
    print("\n" + "=" * 70)
    print("处理完成！")
    print("Processing complete!")
    print("=" * 70)
