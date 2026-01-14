import pandas as pd
from openai import OpenAI
import time
import os
from . import settings  # 导入配置文件 / Import configuration file

# --- 初始化客户端 ---
# --- Initialize Client ---
client = OpenAI(
    api_key=settings.API_KEY, 
    base_url=settings.BASE_URL
)

# --- 辅助函数 ---
# --- Utility Functions ---

def load_prompt(filename):
    """读取 Prompt 模板文件"""
    """Load Prompt template file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {filename}，请检查路径。")
        print(f"❌ Error: File not found {filename}, please check the path.")
        return ""

def get_completion(messages, temperature):
    """封装 API 调用，带重试机制"""
    """Wrap API call with retry mechanism"""
    for i in range(settings.MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=settings.MODEL_NAME,
                messages=messages,
                temperature=temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"⚠️ API调用波动 (第{i+1}/{settings.MAX_RETRIES}次): {e}")
            print(f"⚠️ API call fluctuation (Attempt {i+1}/{settings.MAX_RETRIES}): {e}")
            time.sleep(2)
    return None

# --- 主逻辑 ---
# --- Main Logic ---

def main():
    print(f"🚀 任务启动... 使用模型: {settings.MODEL_NAME}")
    print(f"🚀 Task started... Using model: {settings.MODEL_NAME}")

    # A. 加载 Prompt 模板
    # A. Load Prompt templates
    sys_prompt = load_prompt(settings.PROMPT_SYSTEM_FILE)
    stage1_prompt = load_prompt(settings.PROMPT_STAGE1_FILE)
    stage2_prompt = load_prompt(settings.PROMPT_STAGE2_FILE)

    if not (sys_prompt and stage1_prompt and stage2_prompt):
        print("❌ 缺少 Prompt 模板文件，程序终止。")
        print("❌ Missing Prompt template files, program terminated.")
        return 

    # B. 读取数据
    # B. Read data
    try:
        df = pd.read_csv(settings.INPUT_FILE)
        # 兼容性处理：防止列名不对
        # Compatibility handling: prevent incorrect column names
        if '微博正文' not in df.columns:
            # 假设第2列是正文，自动重命名
            # Assume 2nd column is content, auto rename
            df.rename(columns={df.columns[1]: '微博正文'}, inplace=True)
            print("⚠️ 已自动将第二列重命名为 '微博正文'")
            print("⚠️ Automatically renamed second column to '微博正文'")
    except Exception as e:
        print(f"❌ 读取数据文件失败: {e}")
        print(f"❌ Failed to read data file: {e}")
        return

    # C. 预处理：按关键词合并文本
    # C. Preprocessing: Merge text by keyword
    separator = "\n\n【---下一条微博---】\n\n"
    # 确保内容转为字符串并合并
    # Ensure content is converted to string and merged
    grouped = df.groupby('关键词')['微博正文'].apply(lambda x: separator.join(x.astype(str))).reset_index()
    
    stage1_results_dict = {}
    
    print(f"📊 共发现 {len(grouped)} 个关键词。")
    print(f"📊 Found {len(grouped)} keywords in total.")

    # ==========================
    # Phase 1: 微观分析 (Map)
    # Phase 1: Micro-analysis (Map)
    # ==========================
    print("\n--- Step 1: 微观事实提取 ---")
    print("\n--- Step 1: Micro-fact Extraction ---")
    
    for index, row in grouped.iterrows():
        kw = row['关键词']
        raw_text = row['微博正文']
        
        # 长度截断
        # Length truncation
        if settings.MAX_TEXT_LENGTH > 0 and len(raw_text) > settings.MAX_TEXT_LENGTH:
            raw_text = raw_text[:settings.MAX_TEXT_LENGTH] + "\n...(内容过长已截断)... / ...(content truncated due to length)..."
            
        print(f"   -> 正在分析 [{kw}] (长度: {len(raw_text)})")
        print(f"   -> Analyzing [{kw}] (length: {len(raw_text)})")
        
        messages = [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': stage1_prompt.format(keyword=kw, content=raw_text)}
        ]
        
        result = get_completion(messages, temperature=settings.TEMP_STAGE_1)
        
        if result:
            stage1_results_dict[kw] = result
        else:
            stage1_results_dict[kw] = "（分析失败）/ (Analysis failed)"

    # 保存阶段一结果
    # Save Stage 1 results
    pd.DataFrame(list(stage1_results_dict.items()), columns=['关键词', '微观分析结果'])\
      .to_csv(settings.OUTPUT_STAGE1_CSV, index=False, encoding='utf-8-sig')
    print(f"✅ 阶段一完成，已保存至 {settings.OUTPUT_STAGE1_CSV}")
    print(f"✅ Stage 1 completed, saved to {settings.OUTPUT_STAGE1_CSV}")

    # ==========================
    # Phase 2: 宏观分析 (Reduce)
    # Phase 2: Macro-analysis (Reduce)
    # ==========================
    print("\n--- Step 2: 全局关联分析 ---")
    print("\n--- Step 2: Global Correlation Analysis ---")
    
    # 拼接所有的微观摘要
    # Concatenate all micro-summaries
    combined_summaries = ""
    for kw, summary in stage1_results_dict.items():
        combined_summaries += f"=== 关于关键词【{kw}】的事实摘要 === / === Fact Summary for Keyword 【{kw}】 ===\n{summary}\n\n"
        
    messages_s2 = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': stage2_prompt.format(all_summaries=combined_summaries)}
    ]
    
    final_analysis = get_completion(messages_s2, temperature=settings.TEMP_STAGE_2)
    
    if final_analysis:
        # 保存分析报告
        # Save analysis report
        with open(settings.OUTPUT_STAGE2_MD, "w", encoding="utf-8") as f:
            f.write(final_analysis)
        print(f"✅ 阶段二完成，已保存至 {settings.OUTPUT_STAGE2_MD}")
        print(f"✅ Stage 2 completed, saved to {settings.OUTPUT_STAGE2_MD}")
        
        # ==========================
        # Phase 3: 生成知识库文件
        # Phase 3: Generate Knowledge Base File
        # ==========================
        with open(settings.OUTPUT_FINAL_CONTEXT, "w", encoding="utf-8") as f:
            f.write("【自动生成的背景知识库】 / 【Auto-generated Background Knowledge Base】\n\n")
            f.write("#####################################################\n")
            f.write("PART 1: 全局事件背景与关联网络 (Global Context)\n")
            f.write("PART 1: Global Event Background and Correlation Network\n")
            f.write("#####################################################\n\n")
            f.write(final_analysis)
            f.write("\n\n")
            f.write("#####################################################\n")
            f.write("PART 2: 关键词微观事实库 (Fact Dictionary)\n")
            f.write("PART 2: Keyword Micro-fact Dictionary\n")
            f.write("#####################################################\n\n")
            f.write(combined_summaries)
            
        print(f"🎉 最终知识库已生成: {settings.OUTPUT_FINAL_CONTEXT}")
        print(f"🎉 Final knowledge base generated: {settings.OUTPUT_FINAL_CONTEXT}")
        print("💡 提示: 下一步打标时，请直接读取这个文件的内容作为 Context。")
        print("💡 Tip: For the next labeling step, please read this file's content as Context.")
    else:
        print("❌ 阶段二分析失败，无法生成最终报告。")
        print("❌ Stage 2 analysis failed, unable to generate final report.")

if __name__ == "__main__":
    main()