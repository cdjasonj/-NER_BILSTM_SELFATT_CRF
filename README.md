# -
跑了一下多个粒度嵌入对中文命名实体识别效果的影响
===



嵌入层：分别对各个粒度进行嵌入，记录下对中文命名实体识别任务带来的影响<br>
网络结构: 记录下self-attention对序列标注任务带来的提升<br>

baseline: char_bilstm-crf , char_bilstm-cnn-crf <br>
实验模型: char_bilstm_self_attention + bichar_embedding + trichar_embedding + word_embedding  

