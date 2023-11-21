## Quantization
It means a projection from a set of indices to real domains. Normally people save weights in 32-bit for storage and calculate gradients in 16-bit. 
FP32 and bFP16 (BrainFP16) don't have difference in their ranges.

- 8-bit quantization: mixed-precision matrix decomposition: 8-bit quantize the normal states & weights and leave the outliers unchanged.
- Normal float 4: In layman's terms, instead of using the normal "sign-exponent-mantissa" schema to represent actual values, the normal float data type keeps an index-value
pair where the kth value is the kth quantile of the source tensor.


## Stanford CS244N

T5-the second two method only target the masked out tokens, much lower cost? which two?
repeat of training data (repeats of templates, repeat of patterns) will only get the model overfitted (lower training loss).
attention quadratic complexity problem?
pre-training doesnt benefit translation tasks?
mT5: exact same model but trained on multillingual corpus.
worst case overfitting: memorize key information.
language model perplexity?

Reading list: 
- Exploring the Limits of Transfer Learning with a UnifiedText-to-Text Transformer
- mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer
- Scaling Instruction-Finetuned Language Models
- How Much Knowledge Can You Pack Into the Parameters of a Language Model?
- How CanWe Know What Language Models Know?
- MULTITASK PROMPTED TRAINING ENABLES ZERO-SHOT TASK GENERALIZATION
- Emergent Abilities of Large Language Models
- Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks
