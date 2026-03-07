#!/usr/bin/env python3
"""
从大型 Excel 文件中筛选训练歌曲
- 50000 首中文歌
- 50000 首英文歌
- 去重
- 验证音频文件存在
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def is_chinese(text):
    """判断文本是否包含中文字符"""
    if pd.isna(text) or text is None:
        return False
    text = str(text)
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def contains_other_language(text):
    """
    检测文本是否包含中文和英文之外的其他语言字符
    如日语、韩语、俄语、阿拉伯语等
    """
    if pd.isna(text) or text is None:
        return False
    text = str(text)
    
    # 检查是否包含其他语言字符
    # 日文平假名: \u3040-\u309f
    # 日文片假名: \u30a0-\u30ff
    # 韩文: \uac00-\ud7af
    # 俄语西里尔字母: \u0400-\u04ff
    # 阿拉伯语: \u0600-\u06ff
    # 泰语: \u0e00-\u0e7f
    # 希伯来语: \u0590-\u05ff
    other_language_pattern = r'[\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff\u0e00-\u0e7f\u0590-\u05ff]'
    
    return bool(re.search(other_language_pattern, text))

def is_pure_english(text):
    """
    判断文本是否为纯英文（只包含英文字母、数字、空格和常见标点）
    排除日语、韩语、阿拉伯语等其他语言字符
    """
    if pd.isna(text) or text is None:
        return False
    text = str(text).strip()
    if not text:
        return False
    
    # 允许的字符：英文字母（大小写）、数字、空格、常见英文标点
    allowed_pattern = r'^[a-zA-Z0-9\s\.,\'\"\-\(\)&!?\/:]+$'
    
    # 检查是否包含其他语言字符
    if contains_other_language(text):
        return False
    
    return bool(re.match(allowed_pattern, text))

def classify_song_language(song_name, artist_name):
    """
    判断歌曲语言
    
    规则：
    - 歌名或歌手有中文，且歌名和歌手都只包含中文/英文（无其他语言） -> 中文歌
    - 歌名和歌手都是纯英文（只含英文字母、数字、标点）-> 英文歌
    - 其他 -> None（日语、韩语等，或中文但混有其他语言）
    
    Returns:
        'chinese', 'english', or None
    """
    song_has_chinese = is_chinese(song_name)
    artist_has_chinese = is_chinese(artist_name)
    
    # 检查是否包含其他语言字符（日语、韩语等）
    song_has_other = contains_other_language(song_name)
    artist_has_other = contains_other_language(artist_name)
    
    # 判断中文歌：至少一个有中文，且歌名和歌手都不能包含其他语言
    if (song_has_chinese or artist_has_chinese) and not song_has_other and not artist_has_other:
        return 'chinese'
    
    # 检查是否为纯英文
    song_is_english = is_pure_english(song_name)
    artist_is_english = is_pure_english(artist_name)
    
    # 歌名和歌手都是纯英文 -> 英文歌
    if song_is_english and artist_is_english:
        return 'english'
    
    # 其他情况（包含日语、韩语等，或中文但混有其他语言）
    return None

def main():
    print("=" * 70)
    print("从 Excel 中筛选训练歌曲")
    print("=" * 70)
    
    # 文件路径
    excel_path = Path("/home/zjw524/projects/LIVI-Lyrics-Informed-Version-Identification/240523-全部入库样本-out.xlsx")
    output_dir = Path("/home/zjw524/projects/LIVI-Lyrics-Informed-Version-Identification/data/train_data/audio_links")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[1/5] 读取 Excel 文件...")
    print(f"文件大小: {excel_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("这可能需要几分钟...")
    
    # 只读取需要的列，提高效率
    try:
        df = pd.read_excel(
            excel_path,
            sheet_name='样本情况',
            usecols=['歌名', '歌手', '服务器样本路径'],
            engine='openpyxl'
        )
        print(f"✓ 读取完成，共 {len(df)} 行数据")
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        print("\n尝试查看可用的 sheet 名称...")
        xl_file = pd.ExcelFile(excel_path)
        print(f"可用的 sheets: {xl_file.sheet_names}")
        return
    
    print(f"\n列名: {df.columns.tolist()}")
    print(f"\n前 3 行示例:")
    print(df.head(3).to_string())
    
    # 数据清洗
    print(f"\n[2/5] 数据清洗...")
    print(f"原始行数: {len(df)}")
    
    # 删除缺失音频路径的行
    df = df.dropna(subset=['服务器样本路径'])
    print(f"删除缺失路径后: {len(df)} 行")
    
    # 删除重复的歌曲（基于歌名和歌手）
    df['歌曲标识'] = df['歌名'].astype(str) + '_' + df['歌手'].astype(str)
    df_dedup = df.drop_duplicates(subset='歌曲标识', keep='first')
    print(f"去重后: {len(df_dedup)} 行")
    
    # 分类歌曲语言
    print(f"\n[3/5] 分类歌曲语言...")
    df_dedup['语言'] = df_dedup.apply(
        lambda row: classify_song_language(row['歌名'], row['歌手']),
        axis=1
    )
    
    # 统计
    lang_counts = df_dedup['语言'].value_counts()
    print(f"\n语言分布:")
    for lang, count in lang_counts.items():
        print(f"  {lang}: {count}")
    
    # 筛选中文歌和英文歌
    chinese_songs = df_dedup[df_dedup['语言'] == 'chinese'].copy()
    english_songs = df_dedup[df_dedup['语言'] == 'english'].copy()
    
    print(f"\n可用歌曲:")
    print(f"  中文歌: {len(chinese_songs)} 首")
    print(f"  英文歌: {len(english_songs)} 首")
    
    # 检查音频文件是否存在
    print(f"\n[4/5] 验证音频文件存在性...")
    
    def check_audio_file(row_data):
        """检查单个音频文件是否存在（用于多线程）"""
        idx, row = row_data
        audio_path = Path(row['服务器样本路径'])
        
        if audio_path.exists() and audio_path.is_file():
            return {
                'song_name': row['歌名'],
                'artist': row['歌手'],
                'audio_path': str(audio_path),
                'file_size_mb': audio_path.stat().st_size / 1024 / 1024
            }
        return None
    
    def verify_audio_exists(df_subset, language_name, target_count=50000, max_workers=48):
        """验证并筛选指定数量的有效歌曲（使用多线程）"""
        valid_songs = []
        lock = threading.Lock()
        
        print(f"\n正在验证 {language_name} 歌曲 (使用 {max_workers} 个线程)...")
        
        # 将 DataFrame 转换为可迭代的行数据
        rows_to_check = list(df_subset.iterrows())
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_row = {executor.submit(check_audio_file, row_data): row_data for row_data in rows_to_check}
            
            # 使用 tqdm 显示进度
            with tqdm(total=len(rows_to_check), desc=f"验证{language_name}") as pbar:
                for future in as_completed(future_to_row):
                    pbar.update(1)
                    
                    result = future.result()
                    if result is not None:
                        with lock:
                            if len(valid_songs) < target_count:
                                result['song_id'] = f"{language_name}_{len(valid_songs)+1:06d}"
                                result['language'] = language_name
                                valid_songs.append(result)
                                
                                # 如果已经找到足够的歌曲，可以提前结束
                                if len(valid_songs) >= target_count:
                                    # 取消剩余的任务
                                    for f in future_to_row:
                                        f.cancel()
                                    break
        
        print(f"✓ 找到 {len(valid_songs)} 首有效的{language_name}歌曲")
        return valid_songs
    
    # 验证中文歌（目标：50000 首）
    selected_chinese = verify_audio_exists(chinese_songs, 'chinese', target_count=50000)
    
    # 验证英文歌（目标：50000 首）
    selected_english = verify_audio_exists(english_songs, 'english', target_count=50000)
    
    # 合并结果
    all_selected = selected_chinese + selected_english
    
    print(f"\n[5/5] 生成元数据文件...")
    print(f"总共筛选: {len(all_selected)} 首歌曲")
    print(f"  中文歌: {len(selected_chinese)} 首")
    print(f"  英文歌: {len(selected_english)} 首")
    
    # 保存为 JSON 格式
    metadata_json = output_dir / "train_songs_metadata.json"
    with open(metadata_json, 'w', encoding='utf-8') as f:
        json.dump(all_selected, f, ensure_ascii=False, indent=2)
    print(f"\n✓ JSON 元数据: {metadata_json}")
    
    # 保存为 CSV 格式（便于查看）
    metadata_csv = output_dir / "train_songs_metadata.csv"
    pd.DataFrame(all_selected).to_csv(metadata_csv, index=False, encoding='utf-8-sig')
    print(f"✓ CSV 元数据: {metadata_csv}")
    
    # 生成音频路径列表文件（用于后续处理）
    audio_list_file = output_dir / "train_songs_audio_paths.txt"
    with open(audio_list_file, 'w', encoding='utf-8') as f:
        for song in all_selected:
            f.write(f"{song['audio_path']}\n")
    print(f"✓ 音频路径列表: {audio_list_file}")
    
    # 分别保存中文歌和英文歌的路径列表
    chinese_list_file = output_dir / "train_songs_chinese_paths.txt"
    with open(chinese_list_file, 'w', encoding='utf-8') as f:
        for song in selected_chinese:
            f.write(f"{song['audio_path']}\n")
    print(f"✓ 中文歌路径列表: {chinese_list_file}")
    
    english_list_file = output_dir / "train_songs_english_paths.txt"
    with open(english_list_file, 'w', encoding='utf-8') as f:
        for song in selected_english:
            f.write(f"{song['audio_path']}\n")
    print(f"✓ 英文歌路径列表: {english_list_file}")
    
    # 统计信息
    total_size = sum(s['file_size_mb'] for s in all_selected)
    print(f"\n统计信息:")
    print(f"  歌曲总数: {len(all_selected)}")
    print(f"  中文歌: {len(selected_chinese)} 首")
    print(f"  英文歌: {len(selected_english)} 首")
    print(f"  总文件大小: {total_size:.1f} MB ({total_size/1024:.1f} GB)")
    print(f"  平均文件大小: {total_size/len(all_selected):.1f} MB")
    
    # 显示示例
    print(f"\n示例数据（前 5 首中文歌）:")
    for song in selected_chinese[:5]:
        print(f"  - [{song['song_id']}] {song['song_name']} - {song['artist']}")
        print(f"    路径: {song['audio_path']}")
        print(f"    大小: {song['file_size_mb']:.1f} MB")
    
    print(f"\n示例数据（前 5 首英文歌）:")
    for song in selected_english[:5]:
        print(f"  - [{song['song_id']}] {song['song_name']} - {song['artist']}")
        print(f"    路径: {song['audio_path']}")
        print(f"    大小: {song['file_size_mb']:.1f} MB")
    
    print("\n" + "=" * 70)
    print("✓ 完成！")
    print("=" * 70)
    print(f"\n元数据文件已保存到:")
    print(f"  {metadata_json}")
    print(f"  {metadata_csv}")
    print(f"\n音频路径列表:")
    print(f"  全部: {audio_list_file}")
    print(f"  中文: {chinese_list_file}")
    print(f"  英文: {english_list_file}")
    print(f"\n下一步: 使用这些元数据文件进行模型训练")

if __name__ == "__main__":
    main()
