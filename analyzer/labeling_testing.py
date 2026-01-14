"""
微博打标系统 - 自动更新背景知识 & 测试打标效果
Weibo Labeling System - Auto-update background knowledge & test labeling effects
"""

import pandas as pd
from openai import OpenAI
import json
import re
import argparse
from . import settings


# ======================
# 函数 1: 更新标注 Prompt 中的背景知识
# Function 1: Update background knowledge in labeling prompt
# ======================

def update_labeling_prompt(context_file='analyzer/data/stage2_correlation_analysis_report.md',
                          prompt_file='analyzer/prompts/labeling_prompt.txt'):
    """
    将 stage2_correlation_analysis_report.md 的内容替换 labeling_prompt.txt 中
    @@@背景信息开始 和 背景信息结束@@@ 之间的内容。
    Replace the content between @@@背景信息开始 and 背景信息结束@@@ in labeling_prompt.txt 
    with stage2_correlation_analysis_report.md content.
    
    参数 Parameters:
    - context_file: 背景知识文件路径 / Path to background knowledge file
    - prompt_file: 标注 prompt 文件路径 / Path to labeling prompt file
    """
    try:
        # 读取背景知识文件 / Read background knowledge file
        with open(context_file, 'r', encoding='utf-8') as f:
            context_content = f.read()
        
        # 读取原始 prompt 文件 / Read original prompt file
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read()
        
        # 使用正则表达式替换内容 / Use regex to replace content between markers
        pattern = r'(@@@背景信息开始)\n(.*?)\n(背景信息结束@@@)'
        replacement = rf'\1\n{context_content}\n\3'
        
        new_prompt_text = re.sub(pattern, replacement, prompt_text, flags=re.DOTALL)
        
        # 写回文件 / Write back to file
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(new_prompt_text)
        
        print(f"✓ 已更新 {prompt_file} 的背景知识部分")
        print(f"✓ Updated background knowledge section in {prompt_file}")
    
    except Exception as e:
        print(f"✗ 更新失败: {e}")
        print(f"✗ Update failed: {e}")


# ======================
# 函数 2: 测试打标效果
# Function 2: Test labeling effectiveness
# ======================
def test_labeling(csv_file=None, n=None, api_key=None, model_name=None, random_seed=None,
                 prompt_file='analyzer/prompts/labeling_prompt.txt',
                 output_file='analyzer/data/labeling_test_results.csv'):
    """
    随机抽取 n 条微博进行打标测试。
    Randomly sample n posts for labeling testing.
    
    参数 Parameters:
    - csv_file: CSV 文件路径（两列：id 和 微博正文，可无列标题）/ CSV file path (two columns: id and post content, may have no header)
    - n: 测试条数 / Number of samples to test
    - api_key: 阿里云百炼 API Key（默认从 settings.py 读取）/ Alibaba Cloud API Key (defaults from settings.py)
    - model_name: 模型名称（默认从 settings.py 读取）/ Model name (defaults from settings.py)
    - prompt_file: 标注 prompt 文件路径 / Path to labeling prompt file
    - output_file: 输出结果文件 / Output results file
    """
    
    # 使用 settings 中的配置作为默认值 / Use settings defaults if parameters not provided
    if api_key is None:
        api_key = settings.API_KEY
    if model_name is None:
        model_name = settings.MODEL_NAME
    if n is None:
        n = settings.TEST_SAMPLE_SIZE
    if csv_file is None:
        csv_file = settings.TEST_POST_LIST
    
    try:
        # 读取 CSV 文件 / Read CSV file
        df = pd.read_csv(csv_file, header=None)
        
        # 自动处理列名（如果没有标题）/ Auto-handle column names if no header
        if len(df.columns) == 2:
            df.columns = ['id', 'content']
        
        # 随机抽取 n 条 / Randomly sample n rows
        sample_df = df.sample(n=min(n, len(df)), random_state=random_seed).reset_index(drop=True)
        
        # 读取 prompt / Read prompt file
        with open(prompt_file, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        # 初始化客户端 / Initialize OpenAI client
        client = OpenAI(api_key=api_key, base_url=settings.BASE_URL)
        
        results = []
        print(f"\n开始测试 {len(sample_df)} 条微博...")
        print(f"\nTesting {len(sample_df)} posts...")
        
        for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
            post_id = str(row['id'])
            content = str(row['content'])
            
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"请对以下微博进行打标：\n{content}"}
            ]
            
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.3
                )
                label_result = response.choices[0].message.content
                results.append({
                    'id': post_id,
                    'content': content,
                    'label': label_result
                })
                print(f"  [{idx}/{len(sample_df)}] {post_id[:20]}: ✓")
            
            except Exception as e:
                error_msg = str(e)[:30]
                results.append({
                    'id': post_id,
                    'content': content,
                    'label': f"Error: {error_msg}"
                })
                print(f"  [{idx}/{len(sample_df)}] {post_id[:20]}: ✗")
        
        # 保存结果 / Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        # 拆解 label 列的 JSON，生成展开后的版本 / Parse JSON in label column and generate expanded version
        try:
            results_df['label_parsed'] = results_df['label'].apply(lambda x: json.loads(x) if isinstance(x, str) else {})
            results_expanded = pd.json_normalize(results_df['label_parsed'])
            results_expanded_df = pd.concat([results_df[['id', 'content']], results_expanded], axis=1)
            
            # 生成展开文件名 / Generate expanded file name
            expanded_file = output_file.replace('.csv', '_expanded.csv')
            results_expanded_df.to_csv(expanded_file, index=False, encoding='utf-8-sig')
            
            print(f"✓ 展开版本已保存到 {expanded_file}")
            print(f"✓ Expanded version saved to {expanded_file}")
        except Exception as e:
            print(f"⚠ 拆解 JSON 失败: {e}")
            print(f"⚠ Failed to parse JSON: {e}")
        
        print(f"\n✓ 测试完成，结果已保存到 {output_file}")
        print(f"✓ Testing complete, results saved to {output_file}")
        return results_df
    
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        print(f"✗ Testing failed: {e}")
        return None


if __name__ == '__main__':
    # 命令行参数解析 / Command-line argument parsing
    parser = argparse.ArgumentParser(description='微博打标测试工具 / Weibo Labeling Test Tool')
    parser.add_argument('--n', type=int, default=20, help='要测试的微博数量 / Number of posts to test (default: 20)')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子 / Random seed (default: 42)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("微博打标测试工具 / Weibo Labeling Test Tool")
    print("=" * 50)
    print(f"\n参数 / Parameters:")
    print(f"  - 样本数量 (Sample size): {args.n}")
    print(f"  - 随机种子 (Random seed): {args.random_seed}")
    print(f"\n正在启动打标测试 / Starting labeling test...\n")
    
    test_labeling(n=args.n, random_seed=args.random_seed)
