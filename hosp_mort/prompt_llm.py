import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import pandas as pd
import os
import json
import argparse
from tqdm import tqdm

def load_data(file_path, col='text_by_day1'):
    df = pd.read_csv(file_path)
    reports = df[col].tolist()
    return reports

def format(paragraph, tokenizer, model_id, SYSTEM_PROMPT, PROMPT):
    if 'Mistral' in model_id or 'gemma' in model_id or 'Mixtral' in model_id:
        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n{PROMPT}\n{paragraph}\n\nRisk score: "},
        ]
    elif 'OpenBioLLM' in model_id or 'medalpaca' in model_id or 'meditron' in model_id:
        return f"{SYSTEM_PROMPT}\n{PROMPT}\n{paragraph}\n\nRisk score: "
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{PROMPT}\n{paragraph}\n\nRisk score: "},
        ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    return prompt

def generate(llm, sampling_params, prompts):
    outputs = llm.generate(prompts, sampling_params)
    return [i.outputs[0].text for i in outputs]


def main():
    parser = argparse.ArgumentParser(description="Medical AI Assistant for Risk Assessment")
    parser.add_argument('--model_id', type=str, default="EleutherAI/gpt-neo-2.7B", help='Model ID for the language model')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for processing reports')
    parser.add_argument('--max_tokens', type=int, default=50, help='Maximum number of tokens to generate')

    # do sampling flag
    parser.add_argument('--sampling', action='store_true', default=False, help='Use sampling for generation')

    # quantization flag
    parser.add_argument('--quantization', action='store_true', default=False, help='Quantization flag')

    args = parser.parse_args()

    batch_size = args.batch_size
    model_id = args.model_id
    max_tokens = args.max_tokens
    sampling = args.sampling
    quantization = args.quantization

    reports = load_data('/path/to/data-mort-test.csv')
    output_dir = '/path/to/outputs/hosp_mort_prediction'
    cache_dir = f'/path/to/llms/{model_id}'

    os.makedirs(output_dir, exist_ok=True)

    if sampling:
        sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=max_tokens)
    else:
        sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)


    tokenizer = AutoTokenizer.from_pretrained(cache_dir)

    if not quantization:
        llm = LLM(model=cache_dir, gpu_memory_utilization=0.9, trust_remote_code=True)
    else:
        llm = LLM(model=cache_dir, gpu_memory_utilization=0.9, trust_remote_code=True, dtype=torch.bfloat16, quantization="bitsandbytes", load_format="bitsandbytes")


    SYSTEM_PROMPT = "You are a medical artificial intelligence assistant designed to provide accurate, evidence-based, and professional assessments of patient health concerns. In addition to offering clear and empathetic explanations, you are tasked with calculating and delivering an assessment score that reflects the patient’s risk level based on the provided information. Ensure that all risk scores are grounded in current medical guidelines and best practices. Emphasize that these scores are informational and should not replace personalized medical advice from a healthcare professional. Always prioritize patient safety, confidentiality, and encourage follow-up with a qualified provider."
    # PROMPT = "Given the following clinical notes charted by the healthcare providers, please provide an assessment score for the patient's risk of in-hospital mortality. The score should be a numerical value between 0 and 100, with higher values indicating a higher risk of mortality. Your answer should be a single score only without anything else. The patient's clinical notes are as follows:"

    prompt_1 = "Based on the clinical notes provided by the healthcare team, calculate a numerical risk score for the patient’s in-hospital mortality. The score should range from 0 to 100, where higher numbers indicate greater mortality risk. Provide only the numerical score, without additional commentary. The patient’s clinical notes are as follows:"
    prompt_2 = "Using the clinical notes recorded by healthcare providers, determine the patient’s in-hospital mortality risk as a numerical score between 0 and 100. A higher score reflects a greater risk. Please respond with only the score. The clinical notes for the patient are provided below:"
    prompt_3 = "Review the following clinical notes from healthcare professionals and assign a risk score for the patient’s likelihood of in-hospital mortality. The score should be a number between 0 and 100, with higher values signifying higher risk. Provide only the numerical score. The patient’s notes are as follows:"
    prompt_4 = "Please evaluate the patient’s in-hospital mortality risk based on the clinical notes below, providing a numerical score between 0 and 100. Higher scores indicate increased risk. Respond with the score only. The patient’s clinical notes are as follows:"
    prompt_5 = "Given the healthcare providers’ clinical notes, assign a single numerical score to represent the patient’s in-hospital mortality risk, on a scale from 0 to 100. Higher numbers suggest a higher risk. Provide only the score. The following are the patient’s clinical notes:"

    prompt_dict = {
        "v1": prompt_1,
        "v2": prompt_2,
        "v3": prompt_3,
        "v4": prompt_4,
        "v5": prompt_5
    }

    def get_output_path(prompt_version):
        return os.path.join(output_dir, f'output_maxtoken{max_tokens}_{prompt_version}.jsonl')

    def generate_for_prompt_version(prompt_version):
        PROMPT = prompt_dict[prompt_version]
        outputs = []
        output_path = get_output_path(prompt_version)

        for i in tqdm(range(0, len(reports), batch_size)):
            paragraphs = reports[i:i+batch_size]
            prompts = [format(paragraph, tokenizer, model_id, SYSTEM_PROMPT, PROMPT) for paragraph in paragraphs]
            predictions = generate(llm, sampling_params, prompts)

            outputs.extend(predictions)

            with open(output_path, 'a') as f:
                for pred in predictions:
                    json.dump(pred, f)
                    f.write('\n')
        return outputs
    

    for prompt_version in prompt_dict.keys():
        _ = generate_for_prompt_version(prompt_version)


if __name__ == "__main__":
    main()
        

