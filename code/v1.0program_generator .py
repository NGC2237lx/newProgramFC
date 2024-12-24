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
input_file = os.path.join(os.path.dirname(__file__), "/root/LX/Generation/weibo.json")
output_file = os.path.join(os.path.dirname(__file__), "/root/LX/Generation/execute_program.json")

# Prompt 模板
prompt_template = '''
Generate a python-like program that describes the reasoning steps required to verify the claim step-by-step. 
You can call three functions in the program:
1. Question() to answer a question; 
2. Verify() to verify a simple claim; 
3. Predict() to predict the veracity label. 
You don't need to implement them given functions. You only need to complete the program() function.Please use no more than 5 functions in the program.
Several examples are given as follows.

# The claim is that The song recorded by Fergie that was produced by Polow da Don and was followed by Life Goes On was M.I.L.F.$.
def program():
    fact_1 = Verify("M.I.L.F.$ was recorded by Fergie that was produced by Polow da Don.")
    fact_2 = Verify("M.I.L.F.$ was was followed by Life Goes On.")
    label = Predict(fact_1 and fact_2)
#end

# The claim is that Gregg Rolie and Rob Tyner, are not a keyboardist.
def program():
    fact_1 = Verify("Gregg Rolie is not a keyboardist.")
    fact_2 = Verify("Rob Tyner is not a keyboardist.")
    label = Predict(fact_1 and fact_2)
#end

# The claim is that The model of car Trevor Bayne drives was introduced for model year 2006. The Rookie of The Year in the 1997 CART season drives it in the NASCAR Sprint Cup.
def program():
    answer_1 = Question("Which model of car is drived by Trevor Bayne?")
    fact_1 = Verify(f"{answer_1} was introduced for model year 2006.")
    answer_2 = Question("Who is the Rookie of The Year in the 1997 CART season?")
    fact_2 = Verify(f"{answer_2} drives the model of car Trevor Bayne drives in the NASCAR Sprint Cup.")
    label = predict(fact_1 and fact_2)
#end

# The claim is that Sumo wrestler Toyozakura Toshiaki committed match-fixing, ending his career in 2011 that started in 1989.
def program():
    fact_1 = Verify("Toyozakura Toshiaki ended his career in 2011 that started in 1989.")
    fact_2 = Verify("Toyozakura Toshiaki is a Sumo wrestler.")
    fact_3 = Verify("Toyozakura Toshiaki committed match-fixing.")
    label = Predict(fact_1 and fact_2 and fact_3)
#end

The claim is that [[CLAIM]] and you needs to generate like the above examples with the three functions and end the program() with #end.
def program():
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


def create_result(news_id, claim, program):
    """格式化结果"""
    return {
        'id': news_id,
        'claim': claim,
        'predicted_programs': [program]
    }

def generate_programs(input_file, output_file):
    """逐条生成程序并写入文件"""
    last_processed_id = get_last_processed_id(output_file)
    print(f"Last processed ID: {last_processed_id}")

    # 打开输出文件，以追加模式写入
    with open(output_file, 'a', encoding='utf-8') as f_out:
        if last_processed_id == 0:
            f_out.write("[\n")  # 如果文件为空，写入 JSON 数组的起始符号

        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f_in:
            data = json.load(f_in)

        news_id = last_processed_id + 1  # 从上次的 ID + 1 开始
        for index, item in enumerate(data):
            if index + 1 <= last_processed_id:  # 跳过已处理的记录
                continue

            claim = item.get("Claim")
            if not claim:
                print(f"Skipping item {news_id}: Missing claim.")
                continue

            # 替换 Prompt 中的 [[CLAIM]]
            prompt = prompt_template.replace('[CLAIM]', claim)

            # 模型生成
            device = model.device
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.5)
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 提取程序部分
            start_marker = "and end the program() with #end."
            program_start = full_output.find(start_marker)
            if program_start != -1:
                program_start += len(start_marker)
                program = full_output[program_start:].strip()
                end_marker = "#end"
                program_end = program.find(end_marker)
                if program_end != -1:
                    program = program[:program_end + len(end_marker)].strip()
            else:
                program = "Invalid output: Start marker not found."

            # 写入结果
            result = create_result(news_id, claim, program)
            json.dump(result, f_out, indent=2, ensure_ascii=False)
            f_out.write(",\n")  # 每条记录以逗号分隔

            print(f"Processed claim {news_id}: {claim}")
            news_id += 1

        f_out.write("]")  # JSON 数组结束符号



# 调用函数生成程序
if __name__ == "__main__":
    generate_programs(input_file, output_file)
