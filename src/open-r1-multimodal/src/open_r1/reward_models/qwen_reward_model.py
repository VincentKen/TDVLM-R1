import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import time

class QwenReasoningRewardModel(nn.Module):
    def __init__(self, base_model_name="Qwen/Qwen2-7B-Instruct"):
        super().__init__()
        self.base_model_name = base_model_name
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.processor = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            use_fast=True,
        )

        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _prepare_inputs(self, question, student_response, reference_answer):
        def ensure_str(x):
            return "\n".join(x) if isinstance(x, list) else str(x)
        question = ensure_str(question)
        student_response = ensure_str(student_response)
        reference_answer = ensure_str(reference_answer)
        chat = [
            {"role": "system", "content": "You are a highly skilled and impartial evaluator tasked with scoring how well a student's response matches the provided answer. The provided answer is known to be true. Start with a thorough, side-by-side comparative analysis enclosed within <think> and </think> tags. Give a single numeric score indicating their similarity between 0-1 within <answer> and </answer> tags."},
            {"role": "user", "content": 
                f"Question: {question}\n"
                f"Student Response: {student_response}\n"
                f"Reference Answer (The true answer): {reference_answer}\n"
                f"Think about how the student's response compares to the reference answer and provide this reasoning in <think>...</think> tags. Then provide a score between 0 and 1 in <answer>...</answer> tags."
            }
        ]
        # t = time.time()
        prompt = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # print(f"Prompt generation took {time.time() - t:.4f} seconds")
        # t = time.time()
        inputs = self.processor(text=prompt, return_tensors="pt", padding=True, truncation=True)
        # print(f"Tokenization took {time.time() - t:.4f} seconds")
        return {k: v.to(self.base_model.device) for k, v in inputs.items()}

    def forward(self, question, student_response, reference_answer):
        with torch.no_grad():
            inputs = self._prepare_inputs(question, student_response, reference_answer)
            # t = time.time()
            generated_ids = self.base_model.generate(**inputs, do_sample=False,
                max_new_tokens=128,
                num_beams=1,
                top_p=1.0,
                temperature=None,
                top_k=None
            )
            # print(f"Generation took {time.time() - t:.4f} seconds")
            # t = time.time()
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # print(f"Decoding took {time.time() - t:.4f} seconds")
            # t = time.time()
            reasoning, score_from_text = self.extract_reasoning_and_score(output_text)
            # print(f"Extracting reasoning and score took {time.time() - t:.4f} seconds")

        try:
            score_from_text = float(score_from_text)
        except ValueError:
            # print(f"Failed to convert score from text: {score_from_text}. Try getting numeric value from answer")
            # Find all numbers (float or integer)
            matches = re.findall(r"\b\d+(?:\.\d+)?\b", score_from_text)
            for match in matches:
                try:
                    score = float(match)
                    # If clearly between 0 and 1, assume it's already normalized
                    if 0.0 <= score <= 1.0:
                        score_from_text = score
                    # If it's a whole number up to 100, assume it needs normalization
                    elif 1 < score <= 100:
                        score_from_text = round(score / 100.0, 4)
                except ValueError:
                    continue
            score_from_text = None
        return score_from_text, reasoning

    @staticmethod
    def extract_reasoning_and_score(text_output):
        """
        Extracts <think>...</think> and <answer>...</answer> segments from the model output.
        Returns reasoning (str) and answer_score (float or None).
        """
        # Extract answer from content if it has think/answer tags
        reasoning_match = re.findall(r'<think>(.*?)</think>', text_output, re.DOTALL)
        reasoning = reasoning_match[-1].strip() if reasoning_match else text_output.strip()

        # Extract answer from content if it has think/answer tags
        answer_match = re.findall(r'<answer>(.*?)</answer>', text_output, re.DOTALL)
        answer = answer_match[-1].strip() if answer_match else 0.0

        return reasoning, answer