#!/bin/bash
# Organize generated molecules into MolGenBench folder structure

# 设置默认参数
JSON_FILE="outputs/generated_molecules/molgenbench/lead_optimization_h2l_scaffold/point/12_23_25_300M_xdock_lr6e-6_KLnormclamp5_ufa_gate1_info0.10.2_xattn_new_b128a1/ckpt34350_t1.5k80p.98_round1_uncond/lead_optimization_results.json"
BASE_DIR="/home/yang2531/Documents/Project/StructureSAFE_benchmarking/MolGenBench/MolGenBench_Version1"
OUTPUT_FOLDER_NAME="StructureSAFE_xdock_point_122325_t1.5k80p.98_Hit_to_Lead"  # 自定义文件夹名称
FORMAT="smi"  # 输出格式：smi, csv, 或 txt
# DEDUPLICATE=true  # 默认去重，如果不想去重，添加 --no-deduplicate 参数


python scripts/data_processing/molgenbench/organize_generated_molecules.py \
    --json_file "$JSON_FILE" \
    --base_dir "$BASE_DIR" \
    --output_folder_name "$OUTPUT_FOLDER_NAME" \
    --format "$FORMAT"
    # --no-deduplicate  # 如果不想去重，取消此行注释

echo ""
echo "Done! Generated molecules organized in:"
echo "  $BASE_DIR/{UniProt_ID}/Round1/Hit_to_Lead_Results/{SeriseID}/$OUTPUT_FOLDER_NAME/"

