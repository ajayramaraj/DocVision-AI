---
dataset_info:
- config_name: default
  features:
  - name: id
    dtype: string
  - name: words
    sequence: string
  - name: bboxes
    sequence:
      sequence: int64
  - name: ner_tags
    sequence:
      class_label:
        names:
          '0': O
          '1': B-HEADER
          '2': I-HEADER
          '3': B-QUESTION
          '4': I-QUESTION
          '5': B-ANSWER
          '6': I-ANSWER
  - name: image
    dtype: image
  splits:
  - name: train
    num_bytes: 13202959.0
    num_examples: 149
  - name: test
    num_bytes: 4753604.0
    num_examples: 50
  download_size: 16632013
  dataset_size: 17956563.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
