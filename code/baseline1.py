import os
import json
from transformers import AutoTokenizer, AutoModel

# 设置模型和分词器路径
MODEL_PATH = os.environ.get('MODEL_PATH', '/root/autodl-tmp/glm-4-9b-chat')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

# 文件路径
weibo_file = os.path.join(os.path.dirname(__file__), "/root/LX/Generation/weibo.json")
baseline_output_file = os.path.join(os.path.dirname(__file__), "/root/LX/Generation/baseline_results.json")

# Prompt 模板
baseline_prompt_template = '''
Based on the following statement:
"{claim}"
Is the statement true (1) or false (0)? Please respond with only 1 or 0.
'''

def generate_response(prompt):
    """调用 LLM 生成响应"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 去掉原始 prompt 内容
    response_without_prompt = response.replace(prompt.strip(), "").strip()

    # 检查去掉 prompt 后的响应中是否包含 '1' 或 '0'
    if '1' in response_without_prompt:
        return 1
    elif '0' in response_without_prompt:
        return 0
    else:
        raise ValueError(f"Invalid response: {response_without_prompt}")

def baseline_classification(weibo_file, baseline_output_file):
    """
    使用 baseline 方法对 weibo.json 中的声明进行预测，并计算准确率。
    """
    if not os.path.exists(weibo_file):
        print(f"{weibo_file} does not exist.")
        return

    # 读取 weibo.json 文件
    with open(weibo_file, 'r', encoding='utf-8') as f:
        weibo_data = json.load(f)

    baseline_results = []
    total = 0
    correct = 0
    true_correct = 0
    false_correct = 0
    true_total = 0
    false_total = 0

    # 遍历 weibo 数据
    for entry in weibo_data:
        claim_id = entry["id"]
        claim = entry["Claim"]
        true_label = int(entry["Label"])  # 转为整数

        # 构造 Prompt
        prompt = baseline_prompt_template.replace("{claim}", claim)
        print(f"Processing ID {claim_id}")

        # 调用模型生成预测标签
        try:
            prediction = generate_response(prompt)
            prediction = int(prediction)  # 转为整数
        except ValueError:
            print(f"Invalid response for ID {claim_id}: {prediction}")
            prediction = -1  # 无效响应

        # 更新统计信息
        total += 1
        if prediction == true_label:
            correct += 1
            if true_label == 1:
                true_correct += 1
            elif true_label == 0:
                false_correct += 1

        if true_label == 1:
            true_total += 1
        elif true_label == 0:
            false_total += 1

        # 保存每条记录的结果
        baseline_results.append({
            "id": claim_id,
            "claim": claim,
            "true_label": true_label,
            "predicted_label": prediction,
            "correct": prediction == true_label
        })

    # 计算准确率
    total_accuracy = correct / total if total > 0 else 0
    true_accuracy = true_correct / true_total if true_total > 0 else 0
    false_accuracy = false_correct / false_total if false_total > 0 else 0

    # 保存结果到文件
    output_data = {
        "baseline_results": baseline_results,
        "summary": {
            "total": total,
            "correct": correct,
            "total_accuracy": total_accuracy,
            "true_total": true_total,
            "true_correct": true_correct,
            "true_accuracy": true_accuracy,
            "false_total": false_total,
            "false_correct": false_correct,
            "false_accuracy": false_accuracy
        }
    }

    with open(baseline_output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Baseline results saved to {baseline_output_file}")
    print(f"Total Accuracy: {total_accuracy:.2f}")
    print(f"True Accuracy: {true_accuracy:.2f}")
    print(f"False Accuracy: {false_accuracy:.2f}")

# 主函数调用
if __name__ == "__main__":
    baseline_classification(weibo_file, baseline_output_file)
