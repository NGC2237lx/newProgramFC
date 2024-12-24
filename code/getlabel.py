import os
import json

# 文件路径
predicted_file = os.path.join(os.path.dirname(__file__), "result_count.json")  # 预测结果文件
ground_truth_file = os.path.join(os.path.dirname(__file__), "weibo.json")  # 正确答案文件
emotion_file = os.path.join(os.path.dirname(__file__), "emotion_narrative_analysis.json")  # 情感文件
potential_propagators_file = os.path.join(os.path.dirname(__file__), "potential_propagators.json")  # 谣言传播源文件
output_comparison_file = os.path.join(os.path.dirname(__file__), "accuracy_comparison.json")  # 对比输出文件

# 定义坏标签
BAD_EMOTIONS = ["anger", "fear", "disgust"]
BAD_NARRATIVE_TECHNIQUES = ["Exaggeration", "Inflammatory", "Fabrication"]


def calculate_metrics(predicted_file, ground_truth_file, output_comparison_file):
    """
    计算预测结果与正确答案的准确率、精确率、召回率和 F1 分数，并输出每条记录的对比结果。
    """
    if not os.path.exists(predicted_file) or not os.path.exists(ground_truth_file):
        print("One or both input files do not exist.")
        return

    # 读取文件内容
    with open(predicted_file, 'r', encoding='utf-8') as f:
        predicted_results = json.load(f)

    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)

    # 读取情感文件内容
    with open(emotion_file, 'r', encoding='utf-8') as f:
        emotion_data = json.load(f)

    # 构建 ground_truth 字典，便于查找
    ground_truth_dict = {entry["id"]: entry["Label"] for entry in ground_truth_data}

    # 初始化变量
    correct_predictions = 0
    total_predictions = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    correct_real_news = 0  # 真实新闻预测正确数量
    total_real_news = 0    # 真实新闻总数量
    correct_fake_news = 0  # 假新闻预测正确数量
    total_fake_news = 0    # 假新闻总数量
    fact_total_score = 0
    fact_with_total_score = 0
    total_num = 0
    potential_propagators_count=0
    fact_win=0
    factwith_win=0


    alpha=0.5
    beta=0.5

    c=0
    comparison_results = []

    potential_propagators = []  # 用于存储谣言传播源的列表

    # 遍历预测结果
    for entry in predicted_results:
        entry_id = entry["id"]
        fact_score = entry["FactScore"]
        fact_with_score = entry["Fact_withScore"]
        fact_total_score += fact_score
        fact_with_total_score += fact_with_score
        all_num = entry["all_num"]
        total_num += all_num

        

        id = ground_truth_dict[entry_id]
        

        if(fact_score!=fact_with_score):
            if(fact_score / all_num ==1):
                fact_win+=1
            elif(fact_with_score / all_num ==1):
                if int(ground_truth_dict[entry_id]) == 1:
                    #print("factwith win")
                    factwith_win+=1
        # 检查 all_num 是否为零
        if all_num == 0:
            print(f"Warning: all_num is zero for entry id {entry_id}. Skipping this entry.")
            continue

        
        # if(fact_score!=fact_with_score): 
        #                 print(f"id: {entry_id}, fact_score: {fact_score}, fact_with_score: {fact_with_score}, "
        #           f"all_num: {all_num}, fact_score/all_num: {fact_score/all_num:.2f}, "
        #           f"fact_with_score/all_num: {fact_with_score/all_num:.2f}")
        #                 print(alpha * (fact_score/all_num) + beta * (fact_with_score/all_num))

        #                 print()
        

        # 计算最终预测标签
        if(alpha * (fact_score/all_num) + beta * (fact_with_score/all_num) ==1):
            label_prediction = 1
        else:
            label_prediction = 0
        
        if label_prediction == 0:
            # 查找与 entry_id 匹配的情感条目
            emotion_entry = next((item for item in emotion_data if item["id"] == entry_id), None)
            
            if emotion_entry:
                emotions = emotion_entry.get("emotion", "")
                narrative_techniques = emotion_entry.get("narrative_techniques", "")
                # print(emotions)
                # print(narrative_techniques)
                has_bad_emotion = emotions in BAD_EMOTIONS
                has_bad_narrative = narrative_techniques in BAD_NARRATIVE_TECHNIQUES

                if has_bad_emotion is True and has_bad_narrative is True:
                    potential_propagators_count+=1
                    potential_propagators.append({
                        "id": entry_id,
                        "Claim": ground_truth_data[entry_id - 1]["Claim"],
                        "emotions": emotions,
                        "narrative_techniques": narrative_techniques
                    })
                    

            

        # 获取正确答案
        ground_truth_label = ground_truth_dict.get(entry_id, None)

        # 比较预测与正确答案
        if ground_truth_label is not None:
            ground_truth_label = int(ground_truth_label)
            is_correct = label_prediction == ground_truth_label
            if is_correct:
                correct_predictions += 1

            # 分类统计真实新闻和假新闻
            if ground_truth_label == 1:
                total_real_news += 1
                if is_correct:
                    correct_real_news += 1
            elif ground_truth_label == 0:
                total_fake_news += 1
                if is_correct:
                    correct_fake_news += 1

            # 统计分类指标
            if label_prediction == 1 and ground_truth_label == 1:
                true_positives += 1
            elif label_prediction == 1 and ground_truth_label == 0:
                false_positives += 1
            elif label_prediction == 0 and ground_truth_label == 1:
                false_negatives += 1

            total_predictions += 1

            # 保存对比结果
            comparison_results.append({
                "id": entry_id,
                "Claim": ground_truth_data[entry_id - 1]["Claim"],
                "PredictedLabel": label_prediction,
                "GroundTruthLabel": ground_truth_label,
                "Correct": is_correct
            })

    # 保存谣言传播源到 potential_propagators.json 文件
    with open(potential_propagators_file, 'w', encoding='utf-8') as f:
        json.dump(potential_propagators, f, indent=2, ensure_ascii=False)

    # 计算评价指标
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 计算真实新闻和假新闻的准确率
    real_news_accuracy = correct_real_news / total_real_news if total_real_news > 0 else 0
    fake_news_accuracy = correct_fake_news / total_fake_news if total_fake_news > 0 else 0

    # 保存对比结果到文件
    with open(output_comparison_file, 'w', encoding='utf-8') as f:
        json.dump({
            "comparison_results": comparison_results,
            "metrics": {
                "OverallAccuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1_score,
                "RealNewsAccuracy": real_news_accuracy,
                "FakeNewsAccuracy": fake_news_accuracy
            }
        }, f, indent=2, ensure_ascii=False)

    # 打印指标
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1_score:.2%}")
    print(f"Real News Accuracy: {real_news_accuracy:.2%}")
    print(f"Fake News Accuracy: {fake_news_accuracy:.2%}")
    print(f"Comparison results saved to {output_comparison_file}")
    print(f"Potential Propagators Count: {potential_propagators_count}")

# 调用函数
if __name__ == "__main__":
    calculate_metrics(predicted_file, ground_truth_file, output_comparison_file)
