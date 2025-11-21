from tools.llm import ChatGpt, BaseModel
import jsonlines
from data_utils.dataset import *
import os
import re
from tqdm import tqdm
#from tools.change2excel import change2excel, change2excel_base

import logging
import chardet
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

logging.getLogger('backoff').addHandler(logging.StreamHandler())


def encode_detect(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
        result = chardet.detect(data)
    return result['encoding']


def evaluation(predictions, labels, final_dataset=None, output_dir=None,datatype="gsm8k"):
    """
    Args:
        predictions (_type_): _description_
        labels (_type_): _description_
    """
    dataset_num = len(labels)
    acc_num = 0
    error_idx = []
    output_file = f"{output_dir}/DUR_final_responses.txt"
    error_file = f"{output_dir}/DUR_error_questions.jsonl"
    answer_file = f"{output_dir}/answers.txt"
    question_file = f"{output_dir}/dataset.txt"
    for i in range(dataset_num):

        label = labels[i].replace(',', '')

        print("predictions--",len(predictions))
        if predictions[i] != '':
            pred = str(predictions[i]).strip().replace(',', '')
        else:
            print("Line {} error".format(i + 1))
            pred = 1e9

        if datatype == "aqua" or datatype == "commonsenseqa":
            if pred == label:
                acc_num += 1
            else:
                error_idx.append(i + 1)
        elif datatype == "math":
            if 'incorrect' not in pred.lower():
                acc_num += 1
            else:
                error_idx.append(i + 1)
        elif datatype == "strategyqa" or datatype == "coin_flip":
            if 'yes' in pred.lower() and 'yes' in label.lower() or 'no' in pred.lower() and 'no' in label.lower():
                acc_num += 1
            else:
                error_idx.append(i + 1)
        elif datatype == "last_letters":
            pred = pred.replace(",", "").replace("'", "").replace('"', "").replace(" ", "")
            if pred.lower() == label.lower():
                acc_num += 1
            else:
                error_idx.append(i + 1)
        else:
            from fractions import Fraction
            def parse_value_to_float(value):
                """
                从字符串中提取数字或分数，优先匹配分数，最终以 float 类型返回。
                """
                # 优先匹配分数，再匹配小数
                match = re.search(r"[-+]?\d+/\d+|[-+]?\d*\.?\d+", value)
                if match:
                    num_str = match.group()  # 提取匹配的字符串
                    # 如果是分数形式，使用 Fraction 转换；否则直接用 float
                    if '/' in num_str:
                        return float(Fraction(num_str))
                    else:
                        return float(num_str)
                # 如果未匹配到数字，返回 None
                return None

            # 解析 label 和 pred
            label = parse_value_to_float(label)
            pred = parse_value_to_float(str(pred))
            if pred - label == 0.0:
                acc_num += 1
            elif abs(pred - label) <= 1e-9:
                acc_num += 1
                print(f"exist float error. pred is {pred} label is {label}")
            else:
                error_idx.append(i + 1)


    # extract error question
    if final_dataset is not None:
        with open(output_file, encoding='utf-8') as f:
            final_responses = f.readlines()
            final_responses = [eval(i.strip()) for i in final_responses]
        with open(answer_file, encoding='utf-8') as f:
            answer_file = f.readlines()
            answer_file = [(i.strip()) for i in answer_file]
        with open(question_file, encoding='utf-8') as f:
            question_file = f.readlines()
            question_file = [(i.strip()) for i in question_file]
        for i in range(len(error_idx)):
            idx = error_idx[i] - 1
            gold_answer = labels[idx]
            origin_answer = answer_file[idx]
            origin_question = question_file[idx]
            error_response = final_responses[idx]
            error_question = final_dataset[idx]
            error_answer = str(predictions[idx])
            information = {"idx": idx + 1, "error_question": error_question, "error_response": error_response,
                           "error_answer": error_answer, "gold_answer": gold_answer}

            with jsonlines.open(error_file, 'a') as f:
                f.write(information)
        #change2excel(error_file, output_dir)

    return acc_num / dataset_num


def evaluation_base(predictions, labels, final_dataset=None, output_dir=None,datatype="gsm8k"):
    """
    Args:
        predictions (_type_): _description_
        labels (_type_): _description_
    """
    # dataset_num  = len(labels)
    dataset_num = len(labels)
    acc_num = 0
    error_idx = []
    output_file = f"{output_dir}/baseline_raw_responses.txt"

    base_error_file = f"{output_dir}/baseline_error_questions.jsonl"
    answer_file = f"{output_dir}/answers.txt"
    question_file = f"{output_dir}/dataset.txt"
    for i in range(dataset_num):

        label = labels[i].replace(',', '')

        if predictions[i] != '':
            try:
                #pred = eval(predictions[i].strip().replace(',', ''))
                pred = predictions[i].strip().replace(',', '')
            except:
                print(f"Wrong line {i} content: {predictions[i]} ")
                pred = -1

        else:
            print("Line {} error".format(i + 1))
            pred = 1e9


        if datatype == "aqua" or datatype == "commonsenseqa":
            if pred == label:
                acc_num += 1
            else:
                error_idx.append(i + 1)
        elif datatype == "math":
            if 'incorrect' not in pred.lower():
                acc_num += 1
            else:
                error_idx.append(i + 1)
        elif datatype == "strategyqa" or datatype == "coin_flip":
            if 'yes' in pred.lower() and 'yes' in label.lower() or 'no' in pred.lower() and 'no' in label.lower():
                acc_num += 1
            else:
                error_idx.append(i + 1)
        elif datatype == "last_letters":
            pred = pred.replace(",", "").replace("'", "").replace('"', "").replace(" ", "")
            if pred.lower() == label.lower():
                acc_num += 1
            else:
                error_idx.append(i + 1)
        # elif datatype == "math":
        #
        #     # def remove_is_and_colon(text):
        #     #     # 匹配 'is' 或 ':' 及其之前的所有内容
        #     #     result = re.sub(r'.*(is|:)', '', text)
        #     #     result = result.replace("\\\\", "\\")
        #     #     return result.strip()  # 去掉多余的空格
        #     from tools.mathAnswerExtract import extract_math_answer, normalize_final_answer
        #     pred = extract_math_answer(pred)
        #     pred = normalize_final_answer(pred)
        #
        #     #pred = remove_is_and_colon(pred)
        #
        #     if pred == label:
        #         acc_num += 1
        #     else:
        #         error_idx.append(i + 1)
        #         print("pred",pred,"label",label)
        else:
            from fractions import Fraction
            def parse_value_to_float(value):
                """
                从字符串中提取数字或分数，优先匹配分数，最终以 float 类型返回。
                """
                # 优先匹配分数，再匹配小数
                match = re.search(r"[-+]?\d+/\d+|[-+]?\d*\.?\d+", value)
                if match:
                    num_str = match.group()  # 提取匹配的字符串
                    # 如果是分数形式，使用 Fraction 转换；否则直接用 float
                    if '/' in num_str:
                        return float(Fraction(num_str))
                    else:
                        return float(num_str)
                # 如果未匹配到数字，返回 None
                return None

            # 解析 label 和 pred
            label = parse_value_to_float(label)
            pred = parse_value_to_float(pred)
            if pred - label == 0.0:
                acc_num += 1
            elif abs(pred - label) <= 1e-9:
                acc_num += 1
                print(f"exist float error. pred is {pred} label is {label}")
            else:
                error_idx.append(i + 1)

    # extract error question
    if final_dataset != None:
        with open(output_file, encoding=encode_detect(output_file)) as f:
            final_responses = f.readlines()
            final_responses = [eval(i.strip()) for i in final_responses]
        with open(answer_file, encoding=encode_detect(answer_file)) as f:
            answer_file = f.readlines()
            answer_file = [(i.strip()) for i in answer_file]
        with open(question_file, encoding=encode_detect(question_file)) as f:
            question_file = f.readlines()
            question_file = [(i.strip()) for i in question_file]
        for i in range(len(error_idx)):
            idx = error_idx[i] - 1
            origin_answer = answer_file[idx]
            origin_question = question_file[idx]
            error_response = final_responses[idx]
            error_question = final_dataset[idx]
            error_answer = str(predictions[idx])
            information = {"idx": str, "error_question": str, "error_response": str, "error_answer": str, "answer": str}
            information["idx"] = idx + 1
            information["error_answer"] = error_answer
            information["error_question"] = error_question
            information["error_response"] = error_response
            information["answer"] = origin_answer

            with jsonlines.open(base_error_file, 'a') as f:
                f.write(information)
        #change2excel_base(base_error_file, output_dir)

    return acc_num / dataset_num


def evaluation_DUE_simplified(predictions, labels, final_dataset=None, output_dir=None):
    """

    Args:
        predictions (_type_): _description_
        labels (_type_): _description_
    """
    dataset_num = len(labels)
    acc_num = 0
    error_idx = []
    output_file = f"{output_dir}/DUP_simplified_extracted_answer.txt"
    error_file = f"{output_dir}/DUP_simplified_error_questions.jsonl"
    base_error_file = f"{output_dir}/DUP_simplified_error_questions.jsonl"
    answer_file = f"{output_dir}/answers.txt"
    question_file = f"{output_dir}/dataset.txt"
    for i in range(dataset_num):

        label = labels[i].replace(',', '')

        if predictions[i] != '':
            try:
                #pred = eval(predictions[i].strip().replace(',', ''))
                pred = predictions[i].strip().replace(',', '')
            except:
                print(f"Wrong line {i} content: {predictions[i]} ")
                pred = -1

        else:
            print("Line {} error".format(i + 1))
            pred = 1e9

        #label = eval(label)
        # if pred - label == 0.0:
        #     acc_num += 1
        # elif abs(pred - label) <= 1e-9:
        #     acc_num += 1
        #     print(f"exist float error. pred is {pred} label is {label}")
        # else:
        #     error_idx.append(i + 1)
        if pred == label:
            acc_num += 1
        else:
            error_idx.append(i + 1)

    # extract error question
    if final_dataset is not None:
        with open(output_file, encoding=encode_detect(output_file)) as f:
            final_responses = f.readlines()
            final_responses = [i.strip() for i in final_responses]
        with open(answer_file, encoding=encode_detect(answer_file)) as f:
            answer_file = f.readlines()
            answer_file = [(i.strip()) for i in answer_file]
        with open(question_file, encoding=encode_detect(question_file)) as f:
            question_file = f.readlines()
            question_file = [(i.strip()) for i in question_file]
        for i in range(len(error_idx)):
            idx = error_idx[i] - 1
            origin_answer = answer_file[idx]
            origin_question = question_file[idx]
            error_response = final_responses[idx]
            error_question = final_dataset[idx]
            error_answer = str(predictions[idx])
            information = {"idx": idx + 1, "error_question": error_question, "error_response": error_response,
                           "error_answer": error_answer}

            with jsonlines.open(error_file, 'a') as f:
                f.write(information)
        #change2excel(error_file, output_dir)

    return acc_num / dataset_num


def extract_answer_by_chatgpt(question, response,label="", dataset="gsm8k"):
    """find answer from question's response """
    if dataset == "aqua":
        find_answer = """Here is a math multiple-choice question and a model's answer about this question. Please extract the choice from the answer txt as the final answer for question.
        QUESTION: {}

        ANSWER: {}
                        
        Final format should be a simple choice among A,B,C,D,E. If you know, simply answer A.

        The final answer is:
        """
        model_name = "gpt-4o-mini"
        api_key = "sk-**"
        extract_answer_model = ChatGpt(model=model_name, api_key=api_key)
        extract_answer_model.rateLimit = {"RPM": 1000}
        out = extract_answer_model.generate(find_answer.format(question, response))
        match = re.search(r"[A-E]", out)
        #return out
        if match:
            return match.group()
        else:
            return "E"
    elif dataset == "commonsenseqa":
        find_answer = """Here is a commonsense multiple-choice question and a model's answer about this question. Please extract the choice from the answer txt as the final answer for question.
        QUESTION: {}

        ANSWER: {}

        Final format should be a simple choice among A,B,C,D,E. If you know, simply answer A.

        The final answer is:
        """
        model_name = "gpt-4o-mini"
        api_key = "sk-**"
        extract_answer_model = ChatGpt(model=model_name, api_key=api_key)
        extract_answer_model.rateLimit = {"RPM": 1000}
        out = extract_answer_model.generate(find_answer.format(question, response))
        match = re.search(r"[A-E]", out)
        # return out
        if match:
            return match.group()
        else:
            return "E"
    elif dataset == "strategyqa":
        find_answer = """Here is a strategy judgment question and a model's answer about this question. Please extract the judgment from the answer txt as the final answer for question.
        QUESTION: {}

        ANSWER: {}

        Final format should be an word of either Yes or No. If you know, simply answer Yes.

        The final answer is:
        """
        model_name = "gpt-4o-mini"
        api_key = "sk-**"
        extract_answer_model = ChatGpt(model=model_name, api_key=api_key)
        extract_answer_model.rateLimit = {"RPM": 1000}
        out = extract_answer_model.generate(find_answer.format(question, response))
        return out
    elif dataset == "last_letters":
        find_answer = """Here is a question related to concatenating last letters, along with a model’s answer to this question. Please extract the EXACT string from the answer text to determine the final answer to the question.
        QUESTION: {}

        ANSWER: {}

        Note: Please extract the string after (the answer is:), or the string that appears at the end of the sentence.

        The final answer is:
        """
        model_name = "gpt-4o-mini"
        api_key = "sk-**"
        extract_answer_model = ChatGpt(model=model_name, api_key=api_key)
        extract_answer_model.rateLimit = {"RPM": 1000}
        out = extract_answer_model.generate(find_answer.format(question, response))
        return out
    elif dataset == "coin_flip":
        find_answer = """Here is a Coin flip related question and a model’s answer about this question. Please extract the EXACT string from the answer text as the final answer for question.
        QUESTION: {}

        ANSWER: {}

        Note: If you find that the coin is still heads up, the answer is "yes". If you find that the coin is not still heads up or tails up. the answer is "no".Your answer should be either "yes" or "no".

        The final answer is:
        """
        model_name = "gpt-4o-mini"
        api_key = "sk-**"
        extract_answer_model = ChatGpt(model=model_name, api_key=api_key)
        extract_answer_model.rateLimit = {"RPM": 1000}
        out = extract_answer_model.generate(find_answer.format(question, response))
        return out
    elif dataset == "math":
        find_answer = """You are the wise mathematics answer verifier:You identify as math word problem answer verifier, not an assistant.
You will be provided a math word problem, the real answer for this math word problem, and the predicted output from a generation model. You should compare predicted answer form the predicted output with the real answer and determine if the predicted output is correct or incorrect.
You should not solve the problem by yourself, you only job is to act as a verifier.
        QUESTION: {}

        PREDICTED OUTPUT: {}

        REAL ANSWER: {}

        Your answer between and are limited to only one word:correct or incorrect. If you know, simply answer is correct.

        Your answer is:
        """
        #You will be provided a math word problem, the real answer for this math word problem, and the predicted answer from a generation model. You should understand the problem and validate the correctness of the generated answer in the context of the provided math word problem and the real answer.
        model_name = "gpt-4o-mini"
        api_key = "sk-**"
        extract_answer_model = ChatGpt(model=model_name, api_key=api_key)
        extract_answer_model.rateLimit = {"RPM": 1000}
        out = extract_answer_model.generate(find_answer.format(question,response,label))
        return out
    else:
        find_answer = """Here is a math question and a model's answer about this question. Please extract the EXACT number from the answer txt as the final answer for question.
        QUESTION: {}

        ANSWER: {}

        Final format should be a legal 'number' without any suffix such as '$'. If you know, simply answer 0.

        The final answer is:
        """
        model_name = "gpt-4o-mini"
        api_key = "sk-**"
        extract_answer_model = ChatGpt(model=model_name, api_key=api_key)
        extract_answer_model.rateLimit = {"RPM": 1000}
        out = extract_answer_model.generate(find_answer.format(question, response))
        return out

def extract_answer_by_Rex(response):
    import re

    #text = "The answer is (D), rs. 630."
    match = re.search(r"The answer is \((A|B|C|D|E)\)", response)
    if match:
        return match.group(1)
    else:
        return "E"


def extract_answer_by_rule(questions, predictions: [str],dataset_labels, dataset="gsm8k", output_dir=None):
    answer_file = f"{output_dir}/baseline_extracted_responses.txt"
    if os.path.exists(answer_file):
        with open(answer_file, "r", encoding=encode_detect(answer_file)) as f:
            a = f.readlines()
            final_answers = [i.strip() for i in a]
    else:
        final_answers = []

    for i in tqdm(range(len(final_answers), len(predictions))):
        question = questions[i]
        response = predictions[i]
        out = ""
        # if dataset == "math":
        #     from tools.mathAnswerExtract import extract_math_answer, normalize_final_answer
        #     out = extract_math_answer(response)
        #     out = normalize_final_answer(out)
        # else:
        #     out = extract_answer_by_chatgpt(question, response, dataset=dataset)
        #     out = out.replace("\n", "").replace("\r", "")
        out = extract_answer_by_chatgpt(question, response, dataset_labels[i],dataset=dataset)
        out = out.replace("\n", "").replace("\r", "")
        final_answers.append(out)

        with open(answer_file, 'a') as an:
            an.write(str(out) + "\n")
    return final_answers


def reasoning_base(model: BaseModel, dataset, output_dir, batch_size,datatype="gsm8k"):
    """

    Args:
        model (str): llm model name
        dataset (list,Dataset): a dataset object contain inputs and labels
        :param batch_size:
        :param dataset:
        :param model:
        :param output_dir:
    """
    os.makedirs(output_dir, exist_ok=True)
    # run
    # current_file_path = os.path.abspath(__file__)
    dataset_inputs, dataset_labels, dataset_answers = dataset
    output_file = f"{output_dir}/baseline_raw_responses.txt"
    model.dataset_generate(dataset_inputs, output_file, batch_size=batch_size)
    # evaluation
    with open(output_file, encoding=encode_detect(output_file)) as f:
        predictions = f.readlines()
        predictions = [i.strip() for i in predictions]

    predictions_save_file = f"{output_dir}/baseline_extracted_responses.txt"
    if not os.path.exists(predictions_save_file) or len(open(predictions_save_file).readlines()) != len(dataset_inputs):
        extracted_predictions = extract_answer_by_rule(dataset_inputs, predictions=predictions, dataset_labels=dataset_labels,dataset=datatype,
                                                       output_dir=output_dir)
    else:
        with open(predictions_save_file, encoding=encode_detect(predictions_save_file)) as f:
            extracted_predictions = f.readlines()
            extracted_predictions = [i.strip() for i in extracted_predictions]
    acc = evaluation_base(extracted_predictions, dataset_labels, predictions, output_dir,datatype)
    return acc


def DUE_prompting(model: ChatGpt, dataset, output_dir, Batch_size,datatype="gsm8k"):
    dataset_inputs, dataset_labels, dataset_answers = dataset

    # step1. extract core question
    print("Reveal core question")
    core_question_prompt = " Please extract core question, only the most comprehensive and detailed one!"

    core_question_datasets = [i + core_question_prompt for i in dataset_inputs]
    print("core question stage")
    output_file = f"{output_dir}/core_question_responses.txt"
    model.dataset_generate(core_question_datasets, output_file, batch_size=Batch_size)
    with open(output_file, encoding='utf-8') as f:
        core_question_responses = f.readlines()
        core_question_responses = [eval(i.strip()) for i in core_question_responses]

    # step2. extract information
    print("Extract problem-solving information")
    hints_datasets = []
    for i in range(len(dataset_inputs)):
        hint_prompt = dataset_inputs[
                          i] + " \nNote: Please extract the most useful information related to the core question (" + \
                      core_question_responses[
                          i] + "), only extract the most useful information, and list them one by one!"
        hints_datasets.append(hint_prompt)
    output_file = f"{output_dir}/useful_infomation_responses.txt"
    model.dataset_generate(hints_datasets, output_file, batch_size=Batch_size)
    with open(output_file,encoding='utf-8') as f:
        useful_responses = f.readlines()
        useful_responses = [eval(i.strip()) for i in useful_responses]

    #return 0

    # step3. get the final answer
    print("generate final answer")
    final_datasets = []
    example8=""
    # 文件路径
    file_path = '\data_utils\prompt\svamp\svamp.txt'  # 替换为您的文件路径
    # 读取整个文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        example8 = file.read()
        print(example8)

    if datatype == "aqua":
        for i in range(len(dataset_inputs)):
            if "The core question is: " in core_question_responses[i]:
                core_question_responses[i] = core_question_responses[i].replace("The core question is: ", "")
            pattern = r"Answer Choices: .*"
            match = re.search(pattern, dataset_inputs[i])
            final_prompt = example8 + "\n\nquestion:" + dataset_inputs[i] + "\nHint:" + useful_responses[
                i] + "\n" + match.group() + "\ncore_question:" + core_question_responses[
                               i] + "\nPlease fully understand the Hint and question information and integrated comprehensively, and then give the answer carefully and give details !" + "\nresponse:"
            final_datasets.append(final_prompt)
    else:
        for i in range(len(dataset_inputs)):
            if "The core question is: " in core_question_responses[i]:
                core_question_responses[i] = core_question_responses[i].replace("The core question is: ", "")
            final_prompt = example8 + "\n\nquestion:" + dataset_inputs[i] + "\nHint:" + useful_responses[
                i] + "\ncore_question:" + core_question_responses[
                               i] + "\nPlease fully understand the Hint and question information and integrated comprehensively, and then give the answer carefully and give details !" + "\nresponse:"
            final_datasets.append(final_prompt)


    output_file = f"{output_dir}/DUR_final_responses.txt"
    model.dataset_generate(final_datasets, output_file, batch_size=Batch_size)

    with open(output_file, encoding='utf-8') as f:
        final_responses = f.readlines()
        final_responses = [eval(i.strip()) for i in final_responses]

    # step4. extract answer
    answer_file = f"{output_dir}/DUR_answer.txt"
    if os.path.exists(answer_file):
        with open(answer_file, "r", encoding='utf-8') as f:
            a = f.readlines()
            final_answers = [i.strip() for i in a]
    else:
        final_answers = []

    for i in tqdm(range(len(final_answers), len(final_responses))):
        question = dataset_inputs[i]
        response = final_responses[i]
        out = ""
        if datatype == "math":
            out = extract_answer_by_chatgpt(question, response,dataset_labels[i],dataset=datatype)
        else:
            out = extract_answer_by_chatgpt(question, response, dataset=datatype)
        # replace"\n" and "\r"
        out = out.replace("\n", "").replace("\r", "")
        with open(answer_file, 'a') as an:
            an.write(str(out) + "\n")

    with open(answer_file, encoding='utf-8') as f:
        final_answers = f.readlines()
        final_answers = [i.strip() for i in final_answers]

    return evaluation(final_answers, dataset_labels, final_datasets, output_dir,datatype)

import os
from collections import Counter
from statistics import mode

from fractions import Fraction


def parse_value_to_float(value):
    """
    从字符串中提取数字或分数，优先匹配分数，最终以 float 类型返回。
    """
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return None
    # 优先匹配分数，再匹配小数
    match = re.search(r"[-+]?\d+/\d+|[-+]?\d*\.?\d+", value)
    if match:
        num_str = match.group()  # 提取匹配的字符串
        # 如果是分数形式，使用 Fraction 转换；否则直接用 float
        if '/' in num_str:
            return float(Fraction(num_str))
        else:
            return float(num_str)
    # 如果未匹配到数字，返回 None
    return None


def DUE_prompting_SC(model: ChatGpt, dataset, output_dir, Batch_size,datatype="gsm8k",vote = 10):
    dataset_inputs, dataset_labels, dataset_answers = dataset

    # step1. extract core question
    print("Reveal core question")
    core_question_prompt = " Please extract core question, only the most comprehensive and detailed one!"

    core_question_datasets = [i + core_question_prompt for i in dataset_inputs]
    print("core question stage")
    output_file = f"{output_dir}/core_question_responses.txt"
    model.dataset_generate(core_question_datasets, output_file, batch_size=Batch_size)
    with open(output_file, encoding='utf-8') as f:
        core_question_responses = f.readlines()
        core_question_responses = [eval(i.strip()) for i in core_question_responses]

    # step2. extract information
    print("Extract problem-solving information")
    hints_datasets = []
    for i in range(len(dataset_inputs)):
        hint_prompt = dataset_inputs[
                          i] + " \nNote: Please extract the most useful information related to the core question (" + \
                      core_question_responses[
                          i] + "), only extract the most useful information, and list them one by one!"
        hints_datasets.append(hint_prompt)
    output_file = f"{output_dir}/useful_infomation_responses.txt"
    model.dataset_generate(hints_datasets, output_file, batch_size=Batch_size)
    with open(output_file,encoding='utf-8') as f:
        useful_responses = f.readlines()
        useful_responses = [eval(i.strip()) for i in useful_responses]

    #return 0
    expanded_inputs = [inp for inp in dataset_inputs for _ in range(vote)]
    expanded_labels = [label for label in dataset_labels for _ in range(vote)]
    expanded_core_question_responses = [qes for qes in core_question_responses for _ in range(vote)]
    expanded_useful_responses = [use for use in useful_responses for _ in range(vote)]

    # step3. get the final answer
    print("generate final answer")
    final_datasets = []
    example8=""
    # 文件路径
    file_path = '.\data_utils\prompt\svamp\svamp.txt'  # 替换为您的文件路径
    # 读取整个文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        example8 = file.read()
        print(example8)

    if datatype == "aqua":
        for i in range(len(expanded_inputs)):
            if "The core question is: " in expanded_core_question_responses[i]:
                expanded_core_question_responses[i] = expanded_core_question_responses[i].replace("The core question is: ", "")
            pattern = r"Answer Choices: .*"
            match = re.search(pattern, expanded_inputs[i])
            final_prompt = example8 + "\n\nquestion:" + expanded_inputs[i] + "\nHint:" + expanded_useful_responses[
                i] + "\n" + match.group() + "\ncore_question:" + expanded_core_question_responses[
                               i] + "\nPlease fully understand the Hint and question information and integrated comprehensively, and then give the answer carefully and give details !" + "\nresponse:"
            final_datasets.append(final_prompt)
    else:
        for i in range(len(expanded_inputs)):
            if "The core question is: " in expanded_core_question_responses[i]:
                expanded_core_question_responses[i] = expanded_core_question_responses[i].replace("The core question is: ", "")
            final_prompt = example8 + "\n\nquestion:" + expanded_inputs[i] + "\nHint:" + expanded_useful_responses[
                i] + "\ncore_question:" + expanded_core_question_responses[
                               i] + "\nPlease fully understand the Hint and question information and integrated comprehensively, and then give the answer carefully and give details !" + "\nresponse:"
            final_datasets.append(final_prompt)


    output_file = f"{output_dir}/DUR_final_responses.txt"
    model.dataset_generate(final_datasets, output_file, batch_size=Batch_size)

    with open(output_file, encoding='utf-8') as f:
        predictions = f.readlines()
        predictions = [eval(i.strip()) for i in predictions]

    # step4. extract answer
    answer_file = f"{output_dir}/DUR_answer.txt"
    if os.path.exists(answer_file):
        with open(answer_file, "r", encoding='utf-8') as f:
            a = f.readlines()
            final_answers = [i.strip() for i in a]
    else:
        final_answers = []

    for i in tqdm(range(len(final_answers), len(predictions))):
        question = expanded_inputs[i]
        response = predictions[i]
        out = ""
        if datatype == "math":
            out = extract_answer_by_chatgpt(question, response,expanded_labels[i],dataset=datatype)
        else:
            out = extract_answer_by_chatgpt(question, response, dataset=datatype)
        # replace"\n" and "\r"
        out = out.replace("\n", "").replace("\r", "")
        with open(answer_file, 'a') as an:
            an.write(str(out) + "\n")

    with open(answer_file, encoding='utf-8') as f:
        extracted_predictions = f.readlines()
        extracted_predictions = [i.strip() for i in extracted_predictions]

    # 转换预测值为数值类型
    extracted_predictions = [parse_value_to_float(pred) for pred in extracted_predictions]

    # 确保提取后的预测与扩展的输入数量一致
    assert len(extracted_predictions) == len(dataset_inputs) * vote, "预测结果数量与扩展输入数量不一致！"

    # 分组预测结果，每组对应一个问题的 vote 次回答
    grouped_predictions = [extracted_predictions[i:i + vote] for i in range(0, len(extracted_predictions), vote)]

    # 通过投票（取众数）生成最终预测
    final_predictions = []
    for group in grouped_predictions:
        if group:
            try:
                prediction = mode(group)
                # 使用 mode 获取众数
                final_predictions.append(prediction)
            except:
                # 如果无众数（所有值出现次数相同），使用 Counter 选取
                final_predictions.append(Counter(group).most_common(1)[0][0])

    return evaluation(final_predictions, dataset_labels, final_datasets, output_dir,datatype)

def DUE_prompting_commonsenseqa(model: ChatGpt, dataset, output_dir, Batch_size,datatype="commonsenseqa"):
    dataset_inputs, dataset_labels, dataset_answers = dataset

    # step1. extract core question
    print("Reveal core question")

    core_question_responses = []
    useful_responses = []
    for i in range(len(dataset_inputs)):
        # 将 'Answer Choices:' 之前的部分赋值给 core_question_datasets
        core_question_responses.append(dataset_inputs[i].split("Answer Choices:")[0].strip())

        # 将 'Answer Choices:' 及之后的部分赋值给 useful_responses
        useful_responses.append("Answer Choices:" + dataset_inputs[i].split("Answer Choices:")[1].strip())

    # step3. get the final answer
    print("generate final answer")
    final_datasets = []
    example8=""
    # 文件路径
    file_path = '.\data_utils\prompt\CommonsenseQA\CommonsenseQA.txt'  # 替换为您的文件路径
    # 读取整个文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        example8 = file.read()
        print(example8)

    for i in range(len(dataset_inputs)):
        final_prompt = example8 + "\n\nquestion:" + dataset_inputs[i] + "\nHint:" + useful_responses[
            i] + "\ncore_question:" + core_question_responses[
                        i] + "\nPlease fully understand the Hint and question information and integrated comprehensively, and then give the answer carefully and give details !" + "\nresponse:"
        final_datasets.append(final_prompt)



    output_file = f"{output_dir}/DUR_final_responses.txt"
    model.dataset_generate(final_datasets, output_file, batch_size=Batch_size)

    with open(output_file, encoding='utf-8') as f:
        final_responses = f.readlines()
        final_responses = [eval(i.strip()) for i in final_responses]

    # step4. extract answer
    answer_file = f"{output_dir}/DUR_answer.txt"
    if os.path.exists(answer_file):
        with open(answer_file, "r", encoding='utf-8') as f:
            a = f.readlines()
            final_answers = [i.strip() for i in a]
    else:
        final_answers = []

    for i in tqdm(range(len(final_answers), len(final_responses))):
        question = dataset_inputs[i]
        response = final_responses[i]

        out = extract_answer_by_chatgpt(question, response, dataset=datatype)
        # replace"\n" and "\r"
        out = out.replace("\n", "").replace("\r", "")
        with open(answer_file, 'a') as an:
            an.write(str(out) + "\n")

    with open(answer_file, encoding='utf-8') as f:
        final_answers = f.readlines()
        final_answers = [i.strip() for i in final_answers]

    return evaluation(final_answers, dataset_labels, final_datasets, output_dir,datatype)

def DUE_prompting_last_letter(model: ChatGpt, dataset, output_dir, Batch_size,datatype="last_letters"):
    dataset_inputs, dataset_labels, dataset_answers = dataset

    # step1. extract core question
    print("Reveal core question")

    core_question_responses = []
    useful_responses = []
    for i in range(len(dataset_inputs)):
        # 将 'Answer Choices:' 之前的部分赋值给 core_question_datasets
        core_question_responses.append("Take the last letters of each words and concatenate them sequentially.")
        match = re.search(r'"(.*?)"', dataset_inputs[i])
        extracted_string = match.group(1)
        words = extracted_string.split()

        # 获取每个单词的最后一个字母并连接
        last_letters = "".join([word[-1] for word in words])
        # 格式化输出
        output = "\n".join([f"{i + 1}.{word}" for i, word in enumerate(words)])
        # 将 'Answer Choices:' 及之后的部分赋值给 useful_responses
        useful_responses.append(output)


    # step3. get the final answer
    print("generate final answer")
    final_datasets = []
    example8=""
    # 文件路径
    file_path = '.\data_utils\prompt\last_letter\last_letter.txt'  # 替换为您的文件路径
    # 读取整个文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        example8 = file.read()
        print(example8)

    for i in range(len(dataset_inputs)):
        final_prompt = example8 + "\n\nquestion:" + dataset_inputs[i] + "\nHint:" + useful_responses[
            i] + "\ncore_question:" + core_question_responses[
                        i] + "\nPlease fully understand the Hint and question information and integrated comprehensively, and then give the answer carefully and give details !" + "\nresponse:"
        final_datasets.append(final_prompt)



    output_file = f"{output_dir}/DUR_final_responses.txt"
    model.dataset_generate(final_datasets, output_file, batch_size=Batch_size)

    with open(output_file, encoding='utf-8') as f:
        final_responses = f.readlines()
        final_responses = [eval(i.strip()) for i in final_responses]

    # step4. extract answer
    answer_file = f"{output_dir}/DUR_answer.txt"
    if os.path.exists(answer_file):
        with open(answer_file, "r", encoding='utf-8') as f:
            a = f.readlines()
            final_answers = [i.strip() for i in a]
    else:
        final_answers = []

    for i in tqdm(range(len(final_answers), len(final_responses))):
        question = dataset_inputs[i]
        response = final_responses[i]

        out = extract_answer_by_chatgpt(question, response, dataset=datatype)
        # replace"\n" and "\r"
        out = out.replace("\n", "").replace("\r", "")
        with open(answer_file, 'a') as an:
            an.write(str(out) + "\n")

    with open(answer_file, encoding='utf-8') as f:
        final_answers = f.readlines()
        final_answers = [i.strip() for i in final_answers]

    return evaluation(final_answers, dataset_labels, final_datasets, output_dir,datatype)



def main():
    # model_name = "gpt-4"
    # model_name = "gpt-3.5-turbo-0613"
    model_name = "gpt-3.5-turbo"
    #model_name = "gpt-4o-mini"
    api_key = "sk-*"
    model = ChatGpt(model=model_name, api_key=api_key)
    model.rateLimit = {"RPM": 200}
    model.temperature = 0.7
    # prepare datasetss
    #取样个数
    sample_num = None
    Bathch_size = 10
    dataset_inputs, dataset_labels, dataset_answers = gsm8k(sample_num=sample_num, seed=2023, split="test")#test_fs-our-200
    #dataset_inputs, dataset_labels, dataset_answers = AQuA(sample_num=sample_num, seed=2023, split="test")
    #dataset_inputs, dataset_labels, dataset_answers = MATH(sample_num=sample_num, seed=2023, split="precalculus-300")
    #dataset_inputs, dataset_labels, dataset_answers = CommonsenseQA(sample_num=sample_num, seed=2023, split="test")last_letters
    #dataset_inputs, dataset_labels, dataset_answers = last_letters(sample_num=sample_num, seed=2023, split="test")
    #dataset_inputs, dataset_labels, dataset_answers = coin_flip(sample_num=sample_num, seed=2023, split="test")
    #dataset_inputs, dataset_labels, dataset_answers = math_3(sample_num=sample_num, seed=2023, split="counting_and_probability-50")
    #dataset_inputs, dataset_labels, dataset_answers = MultiArith(sample_num=sample_num, seed=2023,split="test1")
    dataset_inputs, dataset_labels, dataset_answers = SVAMP(sample_num=sample_num, seed=2023,split="test")
    #dataset_inputs, dataset_labels, dataset_answers = AddSub(sample_num=sample_num, seed=2023, split="test")
    #dataset_inputs, dataset_labels, dataset_answers = SingleEq(sample_num=sample_num, seed=2023, split="test")
    dataset = (dataset_inputs, dataset_labels, dataset_answers)
    # if sample_num:
    #     output_dir = "outputs/gsm8k/test_split_{}_samples_gpt3.5".format(sample_num)
    # else:
    #     output_dir = "outputs/gsm8k/all_test_dataset_gpt3.5"
    # if sample_num:
    #     output_dir = "outputs/math1/counting_and_probability/test_split_{}_samples_gpt3.5".format(sample_num)
    # else:
    #     output_dir = "outputs/gsm8k/test_dataset_gpt3.5-test-200-our-api"
    #output_dir = "outputs/svamp/test_dataset_gpt3.5-test-1000-aug-ministeps-8-shot-auto"
    #output_dir = "outputs/gsm8k/test_dataset_gpt3.5-test-1319-aug-ministeps-8-shot-not-aug"
    #output_dir = "outputs/aqua/test_dataset_gpt3.5-test-254-aug-ministeps-8-shot-api2"
    #output_dir = "outputs/svamp/test_dataset_gpt3.5-test-395-aug-ministeps-8-shot-gsm8k-api"
    #output_dir = "outputs/math/test_dataset_gpt4o-test-300-precalculus-4-shot-DUR"
    output_dir = "outputs/svamp/vote/test_dataset_gpt-3.5-1000-test-DUR-4omini-extrator"
    os.makedirs(output_dir, exist_ok=True)
    # save sampled dataset
    # 读取文件加入编码格式UTF-8
    if not os.path.exists(os.path.join(output_dir, "dataset.txt")):
        with open(os.path.join(output_dir, "dataset.txt"), "w", encoding='utf-8') as f:
            for data in dataset_inputs:
                f.write(str(data) + "\n")
    if not os.path.exists(os.path.join(output_dir, "labels.txt")):
        with open(os.path.join(output_dir, "labels.txt"), "w", encoding='utf-8') as f:
            for data in dataset_labels:
                f.write(str(data) + "\n")
    if not os.path.exists(os.path.join(output_dir, "answers.txt")):
        with open(os.path.join(output_dir, "answers.txt"), "w", encoding='utf-8') as f:
            for data in dataset_answers:
                f.write(str(data) + "\n")




    # reasoning_base: baseline
    base_acc = 0
    our_acc = 0
    #base_acc = reasoning_base(model=model, dataset=dataset, output_dir=output_dir, batch_size=Bathch_size)
    our_acc = DUE_prompting(model=model, dataset=dataset, output_dir=output_dir, Batch_size=Bathch_size, datatype = "svamp")
    our_acc = DUE_prompting_SC(model=model, dataset=dataset, output_dir=output_dir, Batch_size=Bathch_size, datatype="svamp",vote=10)
    #our_acc = DUE_prompting_commonsenseqa(model=model, dataset=dataset, output_dir=output_dir, Batch_size=Bathch_size,datatype="commonsenseqa")
    #our_acc = DUE_prompting_last_letter(model=model, dataset=dataset, output_dir=output_dir, Batch_size=Bathch_size,datatype="last_letters")

    #print(f"Baseline acc is {base_acc}\nour acc is {our_acc}\nour simplified acc is: {our_simplified_acc}.")
    print(f"Baseline acc is {base_acc}\nour acc is {our_acc}\n.")

if __name__ == '__main__':
    main()
