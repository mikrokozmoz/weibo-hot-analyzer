"""
处理模块配置文件
所有processing模块使用的参数都在这里配置
"""

# load_posts_from_folder 参数
LOAD_POSTS_FOLDER_PATH = 'files/结果文件'  # 输入文件夹路径，必须指定
LOAD_POSTS_KEYWORD_COLUMN = '关键词'  # 新增列的名称

# extract_top_topics 参数
EXTRACT_TOPICS_TOPICS_COLUMN = '话题'  # 包含话题的列名
EXTRACT_TOPICS_ID_COLUMN = 'id'  # 微博ID列名

# dedupe_posts 参数
DEDUPE_KEYWORD_COL = '关键词'  # 用于分组的关键词列名
DEDUPE_TEXT_COL = '微博正文_cleaned'  # 用于判断重复的清洗后正文列名
DEDUPE_TIME_COL = '发布时间'  # 用于选择保留哪一条记录的时间列
DEDUPE_SUM_COLS = None  # 需要求和的数值列列表，None则为 ['点赞数','评论数','转发数','互动总数']
DEDUPE_SIMILARITY_THRESHOLD = 0.88  # 相似度阈值
DEDUPE_MIN_LEN_FOR_SIMILARITY = 6  # 最小长度以计算相似度
DEDUPE_DEBUG = False  # 是否输出debug信息
DEDUPE_DEBUG_PAIRS_PATH = None  # debug对的保存路径
DEDUPE_AUTO_CLEAN = False  # 当指定的text_col不存在时，是否自动基于'微博正文'进行清洗

# clean_text 参数
# clean_text是单文本清洗，无需全局参数

# tokenize_and_count_words 参数
TOKENIZE_TEXT_COLUMN = '微博正文'  # 包含文本的列名
TOKENIZE_KEYWORD_COLUMN = '关键词'  # 包含关键词的列名
TOKENIZE_WORD_LENGTH_RANGE = (2, 4)  # 保留词的长度范围 (最小, 最大)

# create_word_frequency_dataframe 参数
WORD_FREQ_TOP_N = 50  # 保留词频前N的词

# create_wordclouds 参数
WORDCLOUD_KEYWORD_COLUMN = '关键词'  # 关键词列名
WORDCLOUD_WORD_COLUMN = '词'  # 词频表中表示词的列名
WORDCLOUD_FREQ_COLUMN = '词频'  # 词频表中表示频率的列名
WORDCLOUD_TOP_N = 30  # 每个关键词取前N个词用于绘图
WORDCLOUD_FONT_PATH = r"D:\code\fonts\NotoSansSC-Bold.ttf"  # 字体路径（None则使用系统默认字体）
WORDCLOUD_COLORS_LIST = None  # 颜色列表，None则使用蓝绿黄渐变
WORDCLOUD_COLS = 3  # 子图列数
WORDCLOUD_FIGSIZE = (20, 10)  # 图像大小 (宽, 高)
WORDCLOUD_PREFER_HORIZONTAL = 1.0  # 词云中词的水平倾向
WORDCLOUD_RELATIVE_SCALING = 0.5  # 词云中词大小的相对缩放
WORDCLOUD_MIN_FONT_SIZE = 10  # 词云中词的最小字体大小
WORDCLOUD_SHOW = True  # 是否立即显示词云图
