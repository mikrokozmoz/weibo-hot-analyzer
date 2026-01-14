"""
命令行入口
支持: python -m utils --load_files_from_folder 等参数
"""

import argparse
import sys
from .data_processing import data_processing


def main():
    parser = argparse.ArgumentParser(
        description='微博数据处理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m utils --load_files_from_folder --tokenize
  python -m utils --load_files_from_folder --dedupe --tokenize --create_wordcloud
        """
    )
    
    parser.add_argument(
        '--load_files_from_folder',
        action='store_true',
        help='从指定文件夹加载CSV文件'
    )
    parser.add_argument(
        '--extract_topics',
        action='store_true',
        help='提取话题信息'
    )
    parser.add_argument(
        '--dedupe',
        action='store_true',
        help='进行去重操作'
    )
    parser.add_argument(
        '--tokenize',
        action='store_true',
        help='执行分词和词频统计'
    )
    parser.add_argument(
        '--word_frequency',
        action='store_true',
        help='生成词频DataFrame'
    )
    parser.add_argument(
        '--create_wordcloud',
        action='store_true',
        help='生成词云图片'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，显示帮助信息
    if not any([
        args.load_files_from_folder,
        args.extract_topics,
        args.dedupe,
        args.tokenize,
        args.word_frequency,
        args.create_wordcloud,
    ]):
        parser.print_help()
        sys.exit(0)
    
    # 执行数据处理
    try:
        result = data_processing(
            load_files_from_folder=args.load_files_from_folder,
            extract_topics=args.extract_topics,
            dedupe=args.dedupe,
            tokenize=args.tokenize,
            word_frequency=args.word_frequency,
            create_wordcloud=args.create_wordcloud,
        )
        print("\n✓ 处理完成！")
        return result
    except Exception as e:
        print(f"\n✗ 错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
