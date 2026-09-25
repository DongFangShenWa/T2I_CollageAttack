import pandas as pd
from openai import OpenAI
import requests
import os
import time
from dotenv import load_dotenv

# ================= 配置区 =================
load_dotenv()  # 从 .env 文件加载环境变量
ARK_API_KEY = os.getenv('DOUBAO_API_KEY')
MODEL = "doubao-seedream-5-0-lite-260128"
client = OpenAI(
    api_key=ARK_API_KEY,
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

INPUT_CSV = "coj_commands.csv"
RESULTS_DIR = "attack_results/doubao"
FAILED_INDICES_FILE = os.path.join(RESULTS_DIR, "failed_indices.txt")

# 最大重试次数（含首次；仅对非安全原因的报错生效）
MAX_ATTEMPTS = 3

# Rate limit 冷却时间（秒）
RATE_LIMIT_COOLDOWN = 80

# 普通重试间隔（秒）
RETRY_SLEEP = 3

# API 请求间隔（秒）
SLEEP_SECONDS = 2


# ================= 主逻辑 =================

def is_rate_limit_error(e):
    msg = str(e).lower()
    return any(kw in msg for kw in ["rate limit", "rate_limit", "too many requests", "429"])


def is_safety_rejection(e):
    msg = str(e).lower()
    return any(kw in msg for kw in ["safety", "content_filter", "sensitive", "policy"])


def append_failed_index(image_index):
    """将被安全拦截的 index 追加写入文件"""
    with open(FAILED_INDICES_FILE, "a") as f:
        f.write(f"{image_index}\n")


def run_combined_attack():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    total_count = len(df)
    success_count = 0
    safety_rejected_count = 0
    other_failed_count = 0

    print(f"\n开始【豆包-Seedream-5.0】实验，总计 {total_count} 条...\n")

    for row_pos, row in df.iterrows():
        image_index = row_pos + 1
        combined_prompt = f"{row['command1']}. {row['command2']}. {row['command3']}"
        print(f"[{image_index:02d}/{total_count}] 提示词: {combined_prompt[:80]}...")

        success = False
        safety_rejected = False

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                tag = f"(第{attempt}次)" if attempt > 1 else ""
                print(f"  -> 调用豆包 API{tag}...")

                response = client.images.generate(
                    model=MODEL,
                    prompt=combined_prompt,
                    n=1,
                )

                if response.data and response.data[0].url:
                    image_url = response.data[0].url
                    save_path = os.path.join(RESULTS_DIR, f"combined_hate_{image_index:02d}.png")

                    img_res = requests.get(image_url, timeout=30)
                    if img_res.status_code == 200:
                        with open(save_path, 'wb') as f:
                            f.write(img_res.content)
                        print(f"  -> 生成成功 ✅ -> {save_path}")
                        success_count += 1
                        success = True
                    else:
                        print(f"  -> 下载失败 (HTTP {img_res.status_code})，重试...")
                        if attempt < MAX_ATTEMPTS:
                            time.sleep(RETRY_SLEEP)
                        continue
                else:
                    print(f"  -> 未返回有效 URL，重试...")
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(RETRY_SLEEP)
                    continue
                break

            except Exception as e:
                if is_rate_limit_error(e):
                    print(f"  -> Rate limit 触发，冷却 {RATE_LIMIT_COOLDOWN}s...")
                    time.sleep(RATE_LIMIT_COOLDOWN)
                    continue

                elif is_safety_rejection(e):
                    print(f"  -> 被拦截 ❌ (豆包内容安全策略)，记录 index，稍后重试")
                    safety_rejected = True
                    safety_rejected_count += 1
                    append_failed_index(image_index)
                    break

                else:
                    print(f"  -> 报错 (尝试 {attempt}/{MAX_ATTEMPTS}): {e}")
                    if attempt < MAX_ATTEMPTS:
                        print(f"  -> {RETRY_SLEEP}s 后重试...")
                        time.sleep(RETRY_SLEEP)

        if not success and not safety_rejected:
            print(f"  -> 已尝试 {MAX_ATTEMPTS} 次均失败，放弃")
            other_failed_count += 1

        time.sleep(SLEEP_SECONDS)
        print()

    print("=" * 40)
    print("豆包 Seedream 实验结果统计报告")
    print("=" * 40)
    print(f"总尝试次数: {total_count}")
    print(f"生成成功数: {success_count}")
    print(f"安全拦截数 (已记录): {safety_rejected_count}")
    print(f"其他失败数: {other_failed_count}")
    if safety_rejected_count > 0:
        print(f"安全拦截 index 已保存至: {FAILED_INDICES_FILE}")
    print("=" * 40)


if __name__ == "__main__":
    run_combined_attack()
