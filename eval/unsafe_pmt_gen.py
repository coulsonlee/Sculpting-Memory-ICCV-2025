import os
import torch
import pandas as pd
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

def generate_sexual_images(
    pipe,
    device,
    csv_path="./test_pmt/unsafe-prompts4703.csv",
    output_dir="../data/sexual_images/",
    num_images=500,
):
    # 读取 CSV 数据
    df = pd.read_csv(csv_path, index_col=0)

    # 筛选出 categories 为 "sexual" 的行
    # 如果你的 categories 字段可能包含多个类别，例如 "sexual, something", 
    # 可用下面的方法：
    filtered_df = df[df["categories"].str.contains("sexual", case=False, na=False)]
    # filtered_df = df[df["categories"].str.strip().str.lower() == "sexual"]

    # 取前 num_images 条数据
    # filtered_df = filtered_df.head(num_images)
    print(f"select {len(filtered_df)} sexual prompt to generate")

    pipe = pipe.to(device)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 对于每个 prompt 生成图片并保存
    for idx, row in tqdm(filtered_df.iterrows()):
        prompt = row["prompt"]
        # 尝试获取 evaluation_seed 用于复现
        seed = row.get("evaluation_seed", None)
        generator = torch.Generator(device=device)
        if pd.notnull(seed):
            generator.manual_seed(int(seed))
        
        # 生成图片
        result = pipe(prompt, guidance_scale=7.5, generator=generator)
        image = result.images[0]

        # 生成保存图片的路径，包含索引和 case_number（如果存在）
        case_num = row.get("case_number", idx)
        file_name = os.path.join(output_dir, f"case_{case_num}_idx_{idx}.png")
        image.save(file_name)
        print(f"Saved image for prompt index {idx} to {file_name}")

