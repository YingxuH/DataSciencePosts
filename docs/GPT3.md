# GPT 3 advantages

1. Meta-learning 

Meta-learning is firstly proposed in Chelsea's work [^1]. It is proposed for training the model with generally applicable knowledge such that limited steps of fine-tuning can achieve out-standing performance. During each training step, the possible future parameters of the model will be projected towards different tasks in specified steps of gradient updates, then the loss made by the projected parameters on each of the tasks will be summed together as the loss function of the meta-learning process.  

GPT-3 used the same term 'meta-learning' （or 'zero-shot transfer'） to describe its training process, which is a little bit different from the term defined in Chelsea's paper. GPT-3's meta-learning process is based on un-supervised learning, e.g. purely the next token prediction on the large set of corpus. However, they are developed bearing the same idea, which is to let the model learn a generalized and widly applicable pattern for possible down-stream tasks.

The meta-learning used in GPT-3 can be configured with a outer-loop and inner-loop, where the inner-loop is also referred to as 'in-context learning'. During the outer-loop, the model is trained via un-supervised learning to improve its general capability. For the inner-loop, the model will be fed with some instances of the tasks without any parameter updating, then the model is required to complete the following instances following the same pattern. 

- Few-Shot learning: The model is given a few demonstrations at inference time, together with the task explainations.
- One-Shot learning: The model is given one example at inference time, together with the task explainations. 
- Zero-Shot learning: Only natural language instruction is given at inference time. 

2. Sparse Transformer

The idea is similar to the sparse transformer introduced in Child's paper [^2]. 

3. Large Capacity

4. Large scale training data set. 

It started from the CommonCrawl dataset and filtered out the low-quality and duplicated cotent. The authors removed duplications in the CommonCrawl dataset to keep the integrity of the validation data set. In addition, other high-quality while smaller-scale corpura has been added to improve the diversity. The augmented dataset includes: 
- expanded WebText
- Internet books corpora (Books1 and Books2)
- English-language Wikipedia


[^1]: https://arxiv.org/pdf/1703.03400.pdf
[^2]: https://arxiv.org/pdf/1904.10509.pdf
