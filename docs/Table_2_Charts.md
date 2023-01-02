## Table2Charts

Table2Charts is a recent work with effort in automating the data visualization process with sequence generation architecture and reinforcement learning techniques. It models the chart recommendation process as generating sequences of pre-defined action tokens, done by a deep-Q network combined with a set of human-defined rules. This work breaks down the chart recommendation work into chart type suggesting and chart auto-completion and claims to solve the following problems:

1. It models the chart recommendation task as a sequence generation problem, which enables the usage of reinforcement learning models. 
2. It alleviates the exposure bias problem with the search sampling method. 
3. Most prior arts couldn't handle chart-type recommendations and chart auto-completion simultaneously. On the other hand, Training these two tasks separately will consume a large amount of time and memory.

### Deep-Q network with copy mechanism 
The sequence generation model leverages the encoder-decoder framework, where the encoder takes the field header, data type, and data statistics as input. The decoder includes attention mechanisms to select the most appropriate fields for the next step and determine the type of the next token: whether it refers to a specific data field or represents an action. 

### Search Sampling 
As the model will be thoroughly trained by the teacher-forcing method, it is very likely that the inference process deviates hugely from the ground truth. During inference, this work adopts the next-token generation Q network as the heuristic function for beam searching. The expanded states will then be stored in a replay memory for periodic updates of the Q-network. 

### Mixed Training and Transfer Learning
The chart-type recommendation module and chart auto-completion module share the same encoder. More specifically, the encoder will be trained on chart-type recommendations first and fixed for the subsequent auto-completion training. 

### Large Training corpus
- Excel corpus: 266252 charts created from 165214 tables. 
- Plotly corpus: 67617 charts from 36888 tables



