## Table of Content
- [Architecture](#architecture)
  * [Embeddings](#embeddings)
  * [Attention](#attention)
  * [Point-wise Feed forward layer](#point-wise-feed-forward-layer)
- [Pretraining](#pretraining)
  * [Pre-training objectives](#pre-training-objectives)
    + [Denoising objective](#denoising-objective)
    + [Language model objective](#language-model-objective)
- [Quantization](#quantization)
- [Stanford CS244N](#stanford-cs244n)

## Architecture
Up to 2023, most of the trending LMs follow the transformer architecture. 

### Embeddings 
Compared to traditional RNNs, the sequential feature of the input is not taken care of by the recurrent process. On the contrary, positional embedding handles this. 

The embedding matrix is shared between the input embedding and the output classification process.

### Attention
> [!NOTE]  
> It Might be quite obvious, but I still want to note here: shared matrix is used for each token in the sequence.

- **Encoder**: Fully visiable structure. No Causal mask.
- **Decoder**: Tokens appearing at subsequent times are masked.
- **Predix LM** (T5): Tokens representing the input prefix/instruction are fully visible to each other, while tokens in the real input can only see the preceding tokens.

### Point-wise Feed forward layer
- **Expand layer**: (d_model, d_ffn)
- **Contraction layer**: (d_ffn, d_model)

## Pretraining 
### Pre-training objectives
#### Denoising objective
corrupt the input sequence and reproduce it in the output.
- **Bert** (masked language modeling, MLM): corrupts 15% of the tokens. 80% of the corrupted tokens are replaced with a special mask token, 10% are replaced with a random token, and the remaining 10% are unchanged. The task is to reproduce the entire original sequence.
- **Mass style** (T5): replace corrupted tokens with the special mask token.
- **Replace style** (T5): replace the consecutive span of corrupted tokens with a single mask token, and only predict the concatenated corrupted spans. Note that the specific mask token prefixes each span in the target span.
- **drop style** (T5): drop the corrupted tokens from the input.

> [!NOTE]  
> **Replace style** and **drop style** might speed up the training process and require lower computational cost, as the target sequence is shorter. (How to define shorter as still one token has to be produced each time?)

#### Language model objective
> ❓ Training will be done at once for all time steps. For example, given the input sequence, each time step will output logits for the next token for speed up.

- **Teacher Forcing**: Each token is predicted given the input tokens extracted from the ground truth sequence. **Likely to cause repetition in decoding.**
- **Curriculum Learning**: Randomly choose to use the ground truth output or the generated output from the previous time step as input for the current time step.

## Quantization
It means a projection from a set of indices to real domains. Typically, people save weights in 32-bit for storage and calculate gradients in 16-bit. 
FP32 and bFP16 (BrainFP16) don't have differences in their ranges.

- **8-bit quantization**: mixed-precision matrix decomposition: 8-bit quantizes the normal states & weights and leaves the outliers unchanged.
- **Normal float 4**: In layman's terms, instead of using the normal "sign-exponent-mantissa" schema to represent actual values, the normal float data type keeps an index value
pair where the kth value is the kth quantile of the source tensor. It's information-theoretically optimal, as they claim!

## Stanford CS244N

> ❓ attention quadratic complexity problem?

> ❓ mT5: exact same model but trained on multilingual corpus.

> ❓ worst case overfitting: memorize key information.

> ❓ language model perplexity?

Reading list: 
- Attention is all you need
- Exploring the Limits of Transfer Learning with a UnifiedText-to-Text Transformer
- mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer
- Scaling Instruction-Finetuned Language Models
- How Much Knowledge Can You Pack Into the Parameters of a Language Model?
- How Can We Know What Language Models Know?
- MULTITASK PROMPTED TRAINING ENABLES ZERO-SHOT TASK GENERALIZATION
- Emergent Abilities of Large Language Models
- Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks

## Experiments log
| Model | Quantization |Input | Input Tokens | Output Max Tokens | Time per request | RAM usage |
| -------- |-------- | --------- | -------- | --------- | -------- | --------- |
| LLM-2-7b-chat-GGUF | Q3_K_S |        | 36 | 16 | 1.7s | 0.33G |
| LLM-2-7b-chat-GGUF | Q3_K_S |        | 36 | 32 | 3.5s | 0.4G |
| LLM-2-7b-chat-GGUF | Q3_K_S |        | 36 | 128 | 13.7s | 0.4G |
| LLM-2-7b-chat-GGUF | Q6_K |       | 36 | 32 | 4.6s | 0.3G |
| LLM-2-7b-chat-GGUF | Q6_K |       | 36 | 128 | 22.0s | 0.4G |
| LLM-2-7b-chat-GGUF | Q8_0 |       | 169 | 16 | 3.0s | 0.4G |
| LLM-2-7b-chat-GGUF | Q8_0 |       | 36 | 16 | 2.9s | 0.3G |
| LLM-2-7b-chat-GGUF | Q8_0 |       | 169 | 32 | 5.9s | 0.4G |
| LLM-2-7b-chat-GGUF | Q8_0 |       | 36 | 32 | 5.8s | 1.7G |
| LLM-2-7b-chat-GGUF | Q8_0 |       | 169 | 128 | 24.0s | 2G |
| LLM-2-7b-chat-GGUF | Q8_0 |       | 36 | 128 | 24.0s | 2G |
