#!/bin/bash
# 删除 validation_folder.txt 中列出的所有路径下的 .smi 文件和 scrubbed_ligands_nocond 文件夹

# 默认参数
FOLDER_LIST="${1:-test_structureSAFE/validation_set/validation_folder.txt}"

# 检查文件是否存在
if [ ! -f "$FOLDER_LIST" ]; then
    echo "Error: Folder list file not found: $FOLDER_LIST"
    exit 1
fi

echo "=========================================="
echo "Cleaning up .smi files and scrubbed_ligands_nocond folders"
echo "=========================================="
echo "Folder list: $FOLDER_LIST"
echo "=========================================="
echo ""

# 确认操作
read -p "⚠️  This will delete all .smi files and 'scrubbed_ligands_nocond' folders. Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Operation cancelled."
    exit 0
fi

# 统计
total_folders=0
deleted_smi_count=0
deleted_scrubbed_count=0
total_smi_files=0
total_scrubbed_dirs=0

# 读取文件夹列表并处理
while IFS= read -r folder_path || [[ -n "$folder_path" ]]; do
    # 跳过空行
    [[ -z "$folder_path" ]] && continue
    
    # 去除前后空格
    folder_path=$(echo "$folder_path" | xargs)
    
    # 检查文件夹是否存在
    if [ ! -d "$folder_path" ]; then
        echo "⚠️  Folder not found: $folder_path (skipping)"
        continue
    fi
    
    ((total_folders++))
    
    # 获取文件夹名称（CHEMBL ID）
    chembl_id=$(basename "$folder_path")
    
    echo "[$total_folders] Processing: $chembl_id"
    
    # 1. 删除所有 .smi 文件
    smi_files=$(find "$folder_path" -maxdepth 1 -name "*.smi" -type f)
    if [ -n "$smi_files" ]; then
        smi_count=0
        while IFS= read -r smi_file; do
            if [ -f "$smi_file" ]; then
                rm -f "$smi_file"
                ((smi_count++))
                echo "  🗑️  Deleted: $(basename "$smi_file")"
            fi
        done <<< "$smi_files"
        total_smi_files=$((total_smi_files + smi_count))
        deleted_smi_count=$((deleted_smi_count + smi_count))
    else
        echo "  ℹ️  No .smi files found"
    fi
    
    # 2. 删除 scrubbed_ligands_nocond 文件夹
    scrubbed_dir="${folder_path}/scrubbed_ligands_nocond"
    if [ -d "$scrubbed_dir" ]; then
        total_scrubbed_dirs=$((total_scrubbed_dirs + 1))
        # 统计文件夹中的文件数量
        file_count=$(find "$scrubbed_dir" -type f | wc -l)
        rm -rf "$scrubbed_dir"
        echo "  🗑️  Deleted: scrubbed_ligands_nocond/ ($file_count files)"
        ((deleted_scrubbed_count++))
    else
        echo "  ℹ️  No scrubbed_ligands_nocond folder found"
    fi
    
    echo ""
    
done < "$FOLDER_LIST"

echo "=========================================="
echo "Summary:"
echo "  Total folders processed: $total_folders"
echo "  .smi files deleted: $deleted_smi_count"
echo "  scrubbed_ligands_nocond folders deleted: $deleted_scrubbed_count"
echo "=========================================="
echo "✅ Cleanup completed!"

