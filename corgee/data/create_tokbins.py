import argparse
import json
import os
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np


def process_file(input_file, output_file, tokenizer):
    def get_tokbin_row(_str):
        tokens = tokenizer.encode(_str)
        return np.array([len(tokens)]+tokens)
    
    tokenized_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tokenized_data.append(get_tokbin_row(data['query']))
            tokenized_data.append(get_tokbin_row(data['positive_doc']))
            if 'negative_docs' in data:
                for doc in data['negative_docs']:
                    tokenized_data.append(get_tokbin_row(doc))
    np.concatenate(tokenized_data).astype(np.uint32).tofile(output_file)


def main():
    parser = argparse.ArgumentParser(description="Tokenize JSONL files and convert to tokbin format.")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer model name or path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing JSONL files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tokenized files")
    
    args = parser.parse_args()
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each file in the input directory
    for input_file in Path(args.input_dir).glob('*.jsonl'):
        output_file = Path(args.output_dir) / f"{input_file.stem}.tokbin"
        print(f"Processing {input_file} -> {output_file}")
        process_file(input_file, output_file, tokenizer)


if __name__ == "__main__":
    main()