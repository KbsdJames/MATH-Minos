import json
import pdb
# from sklearn.metrics import f1_score, precision_recall_fscore_support
import re

import json
import argparse
# from sklearn.metrics import f1_score, precision_recall_fscore_support

def f1_micro(y_true, y_pred):
    # 将y_true和y_pred转换为列表以防传入的是NDArray
    y_true = list(y_true)
    y_pred = list(y_pred)

    # 确保y_true和y_pred有相同数量的数据点
    assert len(y_true) == len(y_pred), "The number of predictions must match the number of labels"
    
    tp = 0
    fp = 0
    fn = 0
    
    # 遍历所有的类别
    unique_classes = set(y_true).union(set(y_pred))
    for cls in unique_classes:
        for t, p in zip(y_true, y_pred):
            if t == cls and p == cls:
                tp += 1
            elif t != cls and p == cls:
                fp += 1
            elif t == cls and p != cls:
                fn += 1
    
    precision_micro = tp / (tp + fp) if tp + fp != 0 else 0
    recall_micro = tp / (tp + fn) if tp + fn != 0 else 0
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) != 0 else 0
    
    return f1_micro

def evaluate_text_general(text):
    # 用于存储评估结果
    evaluation_results = []
    
    # 使用正则表达式找到所有的StepX: 和其后的描述
    steps = re.findall(r'(Step\d+): (.+?)\.', text, re.DOTALL)
    
    # 检查每一个步骤是否包含表明错误的关键字，如"incorrect" "not correct" "wrong"
    error_keywords = ["incorrect", "not correct", "wrong", "false"]
    
    for step, description in steps:
        # 默认假设每个步骤是正确的
        correctness = True
        # 检查描述中是否有错误的关键字
        if any(keyword in description.lower() for keyword in error_keywords):
            correctness = False
        
        evaluation_results.append((step, correctness))
    
    # 检查最终答案是否正确
    if "final answer is False" in text:
        final_answer = False
    elif "final answer is True" in text:
        final_answer = True
    else:
        final_answer = None  # 如果文本中没有明确指出最终答案是True还是False，我们不能确定

    if final_answer is not None:
        evaluation_results.append(('final answer', final_answer))
        
    return evaluation_results

def process_model_output(js_c):
    y_pred = js_c[0][1]
    y_pred = evaluate_text_general(y_pred)
    # pdb.set_trace()
    return y_pred

def main(test_path, result_path):
    output = []
    with open(test_path, 'r') as f:
        for line in f.readlines():
            output.append(json.loads(line))
    with open(result_path, 'r') as f:
        result = json.load(f)
    output = output[:len(result)]

    acc = []
    ans_lst = []
    y_pred_lst = []
    for js_c, js_a in zip(output, result):
        
        ans = js_a['label']['acc']
        if js_a['label']['y_pred'] == '[invalid]':
            continue
        ans_lst.append(ans)
        
        y_pred = process_model_output(js_c)
        cor = 'True'
        for y in y_pred:
            if y[1] == False:
                cor = 'False'


        y_pred_lst.append(cor)
        acc.append(ans in y_pred)

    f1_micro_score = f1_micro(ans_lst, y_pred_lst)
    print(f"Accuracy: {sum(acc) / len(acc)}")
    print(f"F1 Micro Score: {f1_micro_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_path", help="Path to the test file containing predictions.")
    parser.add_argument("result_path", help="Path to the result file containing true labels.")
    args = parser.parse_args()
    main(args.test_path, args.result_path)