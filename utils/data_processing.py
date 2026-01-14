"""
数据处理入口函数
将processing模块的函数进行封装，参数由settings.py统一管理
"""

from processing.post_analysis import (
    load_posts_from_folder,
    extract_top_topics,
    dedupe_posts,
    clean_text,
    tokenize_and_count_words,
    create_word_frequency_dataframe,
    create_wordclouds,
)
from . import settings


def data_processing(
    load_files_from_folder=False,
    extract_topics=False,
    dedupe=False,
    tokenize=False,
    word_frequency=False,
    create_wordcloud=False,
):
    """
    数据处理的统一入口函数
    
    参数：
    -----------
    load_files_from_folder : bool
        是否执行从文件夹加载CSV文件的操作
    extract_topics : bool
        是否执行提取话题的操作
    dedupe : bool
        是否执行去重操作
    tokenize : bool
        是否执行分词和词频统计的操作
    word_frequency : bool
        是否生成词频DataFrame
    create_wordcloud : bool
        是否生成词云图片
        
    返回值：
    -----------
    dict
        包含各处理步骤输出的字典
        
    示例：
    -----------
    >>> result = data_processing(load_files_from_folder=True, tokenize=True)
    >>> df = result['df']
    >>> word_freq = result['word_freq_by_keyword']
    """
    result = {}
    df = None
    
    # 1. 加载文件
    if load_files_from_folder:
        if settings.LOAD_POSTS_FOLDER_PATH is None:
            raise ValueError("LOAD_POSTS_FOLDER_PATH未设置，请在settings.py中配置")
        
        df = load_posts_from_folder(
            folder_path=settings.LOAD_POSTS_FOLDER_PATH,
            keyword_column=settings.LOAD_POSTS_KEYWORD_COLUMN,
        )
        result['df'] = df
        print(f"✓ 已加载数据，共{len(df)}行")
    
    # 2. 提取话题
    if extract_topics:
        if df is None:
            raise ValueError("需要先执行load_files_from_folder")
        
        df = extract_top_topics(
            df=df,
            topics_column=settings.EXTRACT_TOPICS_TOPICS_COLUMN,
            id_column=settings.EXTRACT_TOPICS_ID_COLUMN,
        )
        result['df'] = df
        print(f"✓ 已提取话题")
    
    # 3. 去重
    if dedupe:
        if df is None:
            raise ValueError("需要先执行load_files_from_folder")
        
        df = dedupe_posts(
            df=df,
            keyword_col=settings.DEDUPE_KEYWORD_COL,
            text_col=settings.DEDUPE_TEXT_COL,
            time_col=settings.DEDUPE_TIME_COL,
            sum_cols=settings.DEDUPE_SUM_COLS,
            similarity_threshold=settings.DEDUPE_SIMILARITY_THRESHOLD,
            min_len_for_similarity=settings.DEDUPE_MIN_LEN_FOR_SIMILARITY,
            debug=settings.DEDUPE_DEBUG,
            debug_pairs_path=settings.DEDUPE_DEBUG_PAIRS_PATH,
            auto_clean=settings.DEDUPE_AUTO_CLEAN,
        )
        result['df'] = df
        print(f"✓ 已去重，剩余{len(df)}行")
    
    # 4. 分词和词频统计
    if tokenize:
        if df is None:
            raise ValueError("需要先执行load_files_from_folder")
        
        word_freq_by_keyword = tokenize_and_count_words(
            df=df,
            text_column=settings.TOKENIZE_TEXT_COLUMN,
            keyword_column=settings.TOKENIZE_KEYWORD_COLUMN,
            word_length_range=settings.TOKENIZE_WORD_LENGTH_RANGE,
        )
        result['word_freq_by_keyword'] = word_freq_by_keyword
        print(f"✓ 已完成分词统计")
    
    # 5. 生成词频DataFrame
    if word_frequency:
        if 'word_freq_by_keyword' not in result:
            raise ValueError("需要先执行tokenize")
        
        word_freq_df = create_word_frequency_dataframe(
            word_freq_by_keyword=result['word_freq_by_keyword'],
            top_n=settings.WORD_FREQ_TOP_N,
        )
        result['word_freq_df'] = word_freq_df
        print(f"✓ 已生成词频DataFrame")
    
    # 6. 生成词云
    if create_wordcloud:
        if 'word_freq_df' not in result:
            raise ValueError("需要先执行word_frequency")
        
        create_wordclouds(
            df=result['word_freq_df'],
            keyword_column=settings.WORDCLOUD_KEYWORD_COLUMN,
            word_column=settings.WORDCLOUD_WORD_COLUMN,
            freq_column=settings.WORDCLOUD_FREQ_COLUMN,
            top_n=settings.WORDCLOUD_TOP_N,
            font_path=settings.WORDCLOUD_FONT_PATH,
            colors_list=settings.WORDCLOUD_COLORS_LIST,
            cols=settings.WORDCLOUD_COLS,
            figsize=settings.WORDCLOUD_FIGSIZE,
            prefer_horizontal=settings.WORDCLOUD_PREFER_HORIZONTAL,
            relative_scaling=settings.WORDCLOUD_RELATIVE_SCALING,
            min_font_size=settings.WORDCLOUD_MIN_FONT_SIZE,
            show=settings.WORDCLOUD_SHOW,
        )
        result['wordcloud_generated'] = True
        print(f"✓ 已生成词云")
    
    return result
