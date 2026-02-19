import argparse
import json
import os

from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, get_template
from swift.utils import get_model_parameter_info, get_logger, seed_everything

logger = get_logger()
seed_everything(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone inference script for echoflux model.")
    parser.add_argument("--model_id_or_path", type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--json_file", type=str, default=None,
                        help="Path to a JSON file with one or more inference items. "
                             "Each item should have 'messages' and optionally 'videos'/'images'.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional path to save results as JSON.")
    parser.add_argument("--system", type=str, default=None,
                        help="System prompt override.")
    parser.add_argument("--max_new_tokens", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def build_engine(model_path, system=None):
    engine = PtEngine(model_path, torch_dtype='bfloat16', model_type="qwen2_5_vl",
                      attn_impl='flash_attn', use_hf=1)
    template = get_template(engine.model_meta.template, engine.tokenizer, default_system=system)
    info = get_model_parameter_info(engine.model)
    logger.info(f'model_parameter_info: {info}')
    return engine


def run_inference(engine, infer_request, max_new_tokens, temperature):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature, stream=False)
    gen = engine.infer([infer_request], request_config)
    return gen[0].choices[0].message.content




def infer_from_json(engine, json_path, *, max_new_tokens, temperature):
    """Run inference on every item in a JSON file and return a list of results."""
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if isinstance(dataset, dict):
        dataset = [dataset]

    results = []
    for idx, data in enumerate(dataset):
        videos_path = data.get("videos", None)
        images_path = data.get("images", None)
        messages = data["messages"]


        kwargs = {"messages": messages}
        if videos_path:
            kwargs["videos"] = videos_path
        if images_path:
            kwargs["images"] = images_path

        infer_request = InferRequest(**kwargs)
        response = run_inference(engine, infer_request, max_new_tokens, temperature)

        result = {
            "video_id": data.get("video_id"),
            "question": messages[1]["content"],
            "ground_truth": messages[-1]["content"] if len(messages) > 2 else None,
            "generated_text": response,
            "target_disease": data.get("target_disease"),
            "disease": data.get("disease"),
        }
        results.append(result)
        print(f"[{idx + 1}/{len(dataset)}] video_id={result['video_id']}")
        print(f"  Q: {result['question']}")
        print(f"  A: {response}\n")

    return results


def main():
    args = parse_args()
    engine = build_engine(args.model_id_or_path, system=args.system)

    if args.json_file:
        results = infer_from_json(engine, args.json_file,
                                  max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature)
    else:
        raise ValueError("Provide --json_file.")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
