import os
import re
import json
from transformers import AutoTokenizer, AutoModel

# 设置模型和分词器路径
MODEL_PATH = os.environ.get('MODEL_PATH', '/root/autodl-tmp/glm-4-9b-chat')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

# 文件路径
input_program_file = os.path.join(os.path.dirname(__file__), "/root/LX/Generation/execute_program.json")
output_analysis_file = os.path.join(os.path.dirname(__file__), "/root/LX/Generation/emotion_narrative_analysis.json")

# 定义预定义的情感类别和叙述技巧
VALID_EMOTIONS = ["anger", "fear", "surprise", "neutral","sadness","joy","disgust", "anticipation"]
VALID_NARRATIVE_TECHNIQUES = ["Exaggeration", "Inflammatory", "Fabrication", "Objectivity", "Fairness", "Evidence-based"]

# Prompt 模板
emotion_prompt_template = '''
Analyze the emotional tone of the following statement:
"{claim}"
Choose one emotion from the following options: anger, fear, surprise, neutral, sadness, joy, disgust, anticipation.
Only respond with one word in these options
'''
narrative_prompt_template = '''
Analyze the narrative techniques used in the following statement:
"{claim}"
Choose one  techniques from the following options: Exaggeration, Inflammatory, Fabrication, Objectivity, Fairness, Evidence-based.
Only respond with one word in these options
'''

def get_last_processed_id(output_file):
    """
    尝试读取 output_file 中最后一个合法的 ID。

    参数：
        output_file (str): 文件路径。

    返回：
        int: 最大的 ID，若未找到合法数据返回 None。
    """
    try:
        with open(output_file, 'r', encoding='utf-8') as file:
            max_id = 0
            id_pattern = re.compile(r'"id"\s*:\s*(\d+)')

            for line in file:
                matches = id_pattern.findall(line)
                for match in matches:
                    num = int(match)
                    if max_id is None or num > max_id:
                        max_id = num
            print(max_id)
            return max_id
    except FileNotFoundError:
        print(f"文件 {output_file} 未找到。")
    except Exception as e:
        print(f"发生错误: {e}")

    return 0

def generate_response(prompt):
    """调用模型生成响应"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    #print(response)
    if "one word in these options" in response:
        response = response.split("one word in these options", 1)[1].strip()
    return response

def analyze_emotion(claim):
    """分析情感"""
    prompt = emotion_prompt_template.replace("{claim}", claim)
    response = generate_response(prompt)
        # 返回有效的情感类别
    valid_emotion = next((emotion for emotion in VALID_EMOTIONS if emotion.lower() in response.lower()), "unknown")
    
    return valid_emotion

def analyze_narrative(claim):
    """分析叙述技巧"""
    prompt = narrative_prompt_template.replace("{claim}", claim)
    response = generate_response(prompt)
    
    # 返回有效的叙述技巧
    valid_technique = next((tech for tech in VALID_NARRATIVE_TECHNIQUES if tech.lower() in response.lower()), "unknown")

    return valid_technique

def analyze_emotion_and_narrative(input_program_file, output_analysis_file):
    # 获取上次处理的最后一个 ID
    last_processed_id = get_last_processed_id(output_analysis_file)
    print(f"Last processed ID: {last_processed_id}")

    # 创建一个字典用于存储 emotion 和 narrative_techniques 的结果
    analysis_results = {}

    # 读取输入文件
    with open(input_program_file, 'r', encoding='utf-8') as f:
        programs = json.load(f)

    

    # 打开输出文件以追加模式写入
    with open(output_analysis_file, 'a', encoding='utf-8') as output_file:
        if last_processed_id == 0:output_file.write("[\n")
        for idx, news in enumerate(programs):
            # 跳过已处理的新闻
            if idx + 1 <= last_processed_id:
                continue

            try:
                # 分别分析情感和叙述技巧
                emotion = analyze_emotion(news['claim'])
                narrative_techniques = analyze_narrative(news['claim'])
                print(emotion)
                print(narrative_techniques)
                analysis_results[news['id']] = {
                        "emotion": emotion,
                        "narrative_techniques": narrative_techniques
                    }
                #print(analysis_results[news['id']]['emotion'])

                # 构造结果
                result = {
                    'id': news['id'],
                    'claim': news['claim'],
                    'emotion': analysis_results[news['id']]['emotion'],
                    'narrative_techniques': analysis_results[news['id']]['narrative_techniques']
                }
                #print(result)
                # 写入结果到输出文件
                json.dump(result, output_file, indent=2, ensure_ascii=False)
                output_file.write(",\n")  # 每条记录以逗号分隔
                #output_file.write(json.dumps(result, ensure_ascii=False) + ',\n')
                print(f"Processed analysis for claim ID: {news['id']}")

            except Exception as e:
                print(f"Error processing claim ID {news['id']}: {e}")
                continue  # 忽略当前声明，继续处理下一个
        output_file.seek(output_file.tell() - 2, os.SEEK_SET)  # 移动文件指针到最后一个逗号的位置
        output_file.truncate()
        output_file.write("]")  # JSON 数组结束符号
    print(f"All analyses processed and appended to {output_analysis_file}.")

# 主函数调用
if __name__ == "__main__":
    analyze_emotion_and_narrative(input_program_file, output_analysis_file)
