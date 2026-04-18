import os
import re

def identify_footer_ad(lines):
    # 查找最后一个二级标题
    last_h2_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('## '):
            last_h2_idx = i
            
    if last_h2_idx == -1:
        # 如果没有二级标题，尝试查找最后一个一级标题
        for i, line in enumerate(lines):
            if line.startswith('# '):
                last_h2_idx = i
    
    if last_h2_idx == -1:
        return None # 没找到任何标题，不处理
    
    # 关键词：公众号, 扫码, 加群, 关注哦, 最新图解文章
    ad_keywords = ["公众号", "扫码", "加群", "关注哦", "最新图解文章"]
    
    # 从最后一个标题开始向后寻找所有内容块（由分割线分隔）
    # 我们要找的是最后一个包含广告关键词的块及其开始的分割线
    footer_start_idx = -1
    
    # 查找所有分割线的索引
    separator_indices = []
    for i in range(last_h2_idx + 1, len(lines)):
        if re.match(r'^---+$', lines[i].strip()):
            separator_indices.append(i)
            
    if not separator_indices:
        # 如果没有分割线，检查是否有关键词
        for i in range(last_h2_idx + 1, len(lines)):
            if any(keyword in lines[i] for keyword in ad_keywords):
                footer_start_idx = i
                break
    else:
        # 检查每个分割线后的内容
        for i in range(len(separator_indices)):
            start = separator_indices[i]
            end = separator_indices[i+1] if i + 1 < len(separator_indices) else len(lines)
            block_content = "\n".join(lines[start+1:end])
            if any(keyword in block_content for keyword in ad_keywords):
                footer_start_idx = start
                break

    if footer_start_idx != -1:
        return footer_start_idx
    return None

def clean_footers(directory, dry_run=True):
    modified_count = 0
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.md'):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                
                idx = identify_footer_ad(lines)
                if idx is not None:
                    print(f"File: {filepath}")
                    print(f"  [X] Found footer ad starting at line {idx+1}:")
                    for i in range(idx, min(idx + 3, len(lines))):
                        print(f"    {i+1}: {lines[i]}")
                    print(f"    ...")
                    
                    if not dry_run:
                        new_lines = lines[:idx]
                        # 清理末尾空行
                        while new_lines and not new_lines[-1].strip():
                            new_lines.pop()
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(new_lines) + '\n')
                    modified_count += 1
                else:
                    # print(f"File: {filepath} - No footer ad found after last heading.")
                    pass
    
    if dry_run:
        print(f"\n[Dry Run] Total files with detectable footer ads: {modified_count}")
        print("Run with --apply to actually modify the files.")
    else:
        print(f"\n[Apply] Cleaned footer ads in {modified_count} files.")

if __name__ == "__main__":
    import sys
    is_dry_run = "--apply" not in sys.argv
    clean_footers('data/md/os', dry_run=is_dry_run)
