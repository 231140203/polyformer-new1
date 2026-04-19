import json

# 请确认这里的路径是你目前的绝对路径
JSON_PATH = '/root/autodl-tmp/polygon-transformer/refer/data/nnunet/instances.json'
TSV_PATH = '/root/autodl-tmp/polygon-transformer/datasets/finetune/nnunet/nnunet_val.tsv'
CSV_PATH = '/root/autodl-tmp/polygon-transformer/run_scripts/evaluation/per_slice_dice.csv'

print("="*50)
print("🔍 1. 探查 JSON (翻译官的源头)")
print("="*50)
try:
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    print(f"JSON 共有 {len(coco['images'])} 张图片记录。前 5 个是：")
    for i, img in enumerate(coco['images'][:5]):
        # 看看 id 到底是纯数字还是字符串，file_name 有没有特殊后缀
        print(f"  [{i}] id: {repr(img['id'])} (类型: {type(img['id']).__name__}) ---> file_name: {repr(img['file_name'])}")
except Exception as e:
    print(f"❌ JSON 读取失败: {e}")

print("\n" + "="*50)
print("🔍 2. 探查 TSV (题库的源头)")
print("="*50)
try:
    with open(TSV_PATH, 'r', encoding='utf-8') as f:
        print("TSV 前 5 行的第一列 (UID) 是：")
        for i in range(5):
            line = f.readline()
            if not line: break
            uid = line.split('\t')[0]
            print(f"  [{i}] UID: {repr(uid)}")
except Exception as e:
    print(f"❌ TSV 读取失败: {e}")

print("\n" + "="*50)
print("🔍 3. 探查 CSV (成绩单的源头)")
print("="*50)
try:
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        print("CSV 前 5 行的原始数据是：")
        for i in range(5):
            line = f.readline()
            if not line: break
            print(f"  [{i}] Raw Line: {repr(line.strip())}")
except Exception as e:
    print(f"❌ CSV 读取失败: {e}")