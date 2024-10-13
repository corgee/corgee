# CORGEE: Contrastive Generalizable Embeddings

<div align="center">
  <img src="resources/logo.png" alt="CORGEE Logo" width="480" height="300">
</div>

CORGEE is a highly optimized implementation for training state-of-the-art embedding models using Contrastive Generalizable Embeddings.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Important Parameters](#important-parameters)

## Environment Setup

1. Create and activate a fresh conda environment
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Dataset Preparation

### Data Format
Prepare your datasets as jsonl files with the following columns:
- `query`: str
- `positive_doc`: str
- `negative_docs`: List[str] (not needed for pretraining)

Sample datasets:
- Pretraining: `resources/pretraining_data/*.jsonl`
- Fine-tuning: `resources/finetuning_data/*.jsonl`

### Tokenization
Training requires pretokenized datasets stored as binary files. To tokenize your data:

```bash
# For pretraining data
python corgee/data/create_tokbins.py \
  --tokenizer intfloat/multilingual-e5-base \
  --input_dir resources/pretraining_data/ \
  --output_dir resources/pretraining_data_tokenized/

# For fine-tuning data
python corgee/data/create_tokbins.py \
  --tokenizer intfloat/multilingual-e5-base \
  --input_dir resources/finetuning_data/ \
  --output_dir resources/finetuning_data_tokenized/
```

## Training

1. Create a `config.yaml` file with relevant parameters.
   - Sample pretraining and finetuning configs are provided in the `configs/` directory.

2. Start training:

   ### Single Node
   For running on a single node:
   ```bash
   source run.sh config.yaml
   ```

   ### Multiple Nodes
   For running on multiple nodes (e.g., 4 nodes):
   ```bash
   DIST_NUM_NODES=4 source run.sh config.yaml
   ```

   Adjust the `DIST_NUM_NODES` value according to your setup.

3. Parameter Configuration:
   - Set parameters in `config.yaml`
   - Override important parameters via command line as needed

Sample configs are provided in `configs/`

## Important Parameters

| Parameter | Description |
|-----------|-------------|
| `output_dir` | Directory for logs and saved models |
| `batch_size` | Training batch size |
| `max_forward_batch_size` | Maximum batch size for GPU forwarding |
| `files` | Dictionary of dataset configurations |

### Dataset Configuration

Each dataset in the `files` dictionary requires:
- `num_steps`: Number of training batches to sample
- `maxlen1`: Maximum tokens in query
- `maxlen2`: Maximum tokens in positive/negative documents
- `file_pattern`: Regex pattern for tokbin files

**Note**: Batches are sampled from one dataset at a time. For language-wise sampling, make each language a separate dataset.