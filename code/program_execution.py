import os
import json
import re
from transformers import AutoTokenizer, AutoModel

# 设置模型和分词器路径
MODEL_PATH = os.environ.get('MODEL_PATH', '/root/autodl-tmp/glm-4-9b-chat')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

# 文件路径
execute_program_file = os.path.join(os.path.dirname(__file__), "/root/LX/Generation/execute_program.json")
emotion_narrative_analysis_file = os.path.join(os.path.dirname(__file__), "/root/LX/Generation/emotion_narrative_analysis.json")
result_file = os.path.join(os.path.dirname(__file__), "result.json")


def get_last_processed_id(output_file):
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


def load_emotion_narrative_analysis(file_path):

    if not os.path.exists(file_path):
       raise FileNotFoundError(f"{file_path} does not exist.")
   
    with open(file_path, 'r', encoding='utf-8') as f:
       try:
           analysis_data = json.load(f)  # 读取整个 JSON 数组
       except json.JSONDecodeError as e:
           print(f"JSONDecodeError: {e}")
           return {}
    return {item["id"]: {"emotion": item.get("emotion", "neutral"), "narrative_techniques": item.get("narrative_techniques", [])} for item in analysis_data}


def extract_commands(program):
    """从 def program() 中提取所有命令"""
    commands = []
    lines = program.split("\n")
    for line in lines:
        line = line.strip()
        if line and ("= Question" in line or "= Verify" in line or "= Predict" in line):
            commands.append(line)
    return commands

def answer_question(question,claim):
   """
   调用 LLM 生成回答，并返回第一个句子，去掉原始的 prompt。
   根据问题长度动态调整 max_new_tokens。
   """
   prompt = f"I read the following information: {claim}.Answer the following question with few words as briefly as possible, not necessarily in a complete sentence:\n{question}\nThe answer is:"
   inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
   
   # 动态调整 max_new_tokens，假设每个单词平均需要 1.5 个 token
   question_length = len(question.split())
   max_new_tokens = int(question_length * 1.5) + 10  # 加 10 以确保有足够的空间生成完整句子
   outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
   response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # 去掉原始的 prompt 部分
   if response.startswith(prompt):
       response = response[len(prompt):].strip()
    # 只返回第一个句子
   first_sentence = response.split('.')[0] + '.'
   return first_sentence


def verify_command(claim, message):
   prompt = f"I read the following message {message}.Is the following statement true or false?\n\"{claim}\"Answer only with True or False:"
   inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
   
   # 动态调整 max_new_tokens，假设每个单词平均需要 1.5 个 token
   prompt_length = len(prompt.split())
   max_new_tokens = int(prompt_length * 0.1) + 4  # 加 4 以确保有足够的空间生成完整句子
   
   outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
   result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
   #print("result:", result)
   if result.startswith(prompt):
       result = result[len(prompt):].strip()
   #print("result:", result)
   if result.find("True") != -1:
      return "True"
   elif result.find("False") != -1:
      return "False"
   return None  # 如果都不包含，返回 None 或其他适当的值

def verify_with_information_from_file(claim, emotion, narrative_techniques, message):
   information = f"The information contains {emotion} emotions and employs {narrative_techniques} narrative techniques."
   prompt = f"I read the following information {message}.{information}\nBased on the above, is the following statement true or false?\n\"{claim}\"\nAnswer only with True or False:"
   inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
   
   # 动态调整 max_new_tokens
   prompt_length = len(prompt.split())
   max_new_tokens = int(prompt_length * 0.1) + 5  # 加 5 以确保有足够的空间生成完整句子
   
   outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
   result = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
   #print("result:", result)
   if result.startswith(prompt):
       result = result[len(prompt):].strip()
   #print("result:", result)
   if result.find("True") != -1:
      return "True"
   elif result.find("False") != -1:
      return "False"
   return None  # 如果都不包含，返回 None 或其他适当的值



def execute_programs(execute_program_file, emotion_narrative_analysis_file, result_file):
    # 加载情感和叙述分析文件
    emotion_narrative_map = load_emotion_narrative_analysis(emotion_narrative_analysis_file)

    # 获取最后处理的 ID
    last_processed_id = get_last_processed_id(result_file)
    print(f"Last processed ID: {last_processed_id}")

    # 读取 execute_program.json 文件
    with open(execute_program_file, 'r', encoding='utf-8') as f:
        programs = json.load(f)

    # 打开 result_file 以追加模式写入
    with open(result_file, 'a', encoding='utf-8') as output_file:
        if last_processed_id == 0:
            output_file.write("[\n")  # 添加数组起始符号
        
        for program_data in programs:
            # 跳过已处理的 ID
            if program_data["id"] <= last_processed_id:
                continue

            try:
                # 提取命令并处理
                program = program_data["predicted_programs"][0]
                commands = extract_commands(program)
                fact_results = {}  # 基本验证结果
                fact_with_results = {}  # 带情绪和叙述技巧的验证结果
                questions = {}  # 存储问题变量
                answers = {}  # 用于存储 Question 的答案
                qclaim = program_data.get("claim", "")
                # 第一轮验证：提取问题和执行基本 Verify
                for command in commands:
                    if "= Question" in command:
                        return_var, question = command.split("=")[0].strip(), command.split("Question")[1].strip().strip('(").')
                        answer = answer_question(question,qclaim)
                        #print(answer)
                        answers[return_var] = answer  # 存储答案
                        questions[return_var] = question

                    elif "= Verify" in command:
                        fact_idx = command.split("=")[0].strip()
                        claim = command.split("Verify")[1].strip().strip('(").')

                        # 替换嵌套变量
                        for key, value in answers.items():
                            #print("claimbe",claim)
                            claim = claim.replace(f"{{{key}}}", value)
                            #print("claimaf",claim)

                        # 调用 Verify 执行基本认证
                        fact_results[fact_idx] = verify_command(claim,qclaim)
                        #print(fact_results)

                # 第二轮验证：带情绪和叙述技巧的 Verify
                for command in commands:
                    if "= Verify" in command:
                        fact_idx = command.split("=")[0].strip()
                        claim = command.split("Verify")[1].strip().strip('(").')

                        # 替换嵌套变量
                        for key, value in answers.items():
                            claim = claim.replace(f"{{{key}}}", value)

                        # 读取情绪和叙述技巧数据
                        analysis = emotion_narrative_map.get(program_data["id"], {})
                        emotion = analysis.get("emotion", "neutral")
                        narrative_techniques = analysis.get("narrative_techniques", [])

                        # 调用 Verify 执行带情绪和叙述技巧的认证
                        fact_with_results[fact_idx+"with"] = verify_with_information_from_file(claim, emotion, narrative_techniques,qclaim)
                        #print(fact_with_results)

                # 构造结果并写入文件
                result = {
                    "id": program_data["id"],
                    "claim": program_data["claim"],
                    "Question": questions,
                    "answers": answers,
                    "basic_verification": fact_results,  # 第一轮验证结果
                    "emotion_narrative_verification": fact_with_results  # 第二轮验证结果
                }

                json.dump(result, output_file, indent=2, ensure_ascii=False)
                output_file.write(",\n")  # 每条记录以逗号分隔

                print(f"Processed ID: {program_data['id']}")

            except Exception as e:
                print(f"Error processing ID {program_data['id']}: {e}")
                continue  # 遇到错误时跳过当前声明

        output_file.seek(output_file.tell() - 2, os.SEEK_SET)  # 移动文件指针到最后一个逗号的位置
        output_file.truncate()
        output_file.write("]")  # JSON 数组结束符号


if __name__ == "__main__":
    execute_programs(execute_program_file, emotion_narrative_analysis_file, result_file)
