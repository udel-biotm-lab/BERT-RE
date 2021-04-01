### 1. Evaluation datasets for the PPI, DDI and ChemProt tasks: 
[PPI](https://drive.google.com/file/d/1dn2yDKj7-3SsyKQ5Zm_5sTlLxTCfqQpy/view?usp=sharing)\
[DDI](https://drive.google.com/file/d/1EEtN1LMI-W4iqtsXVfc64v5PsoAEmJad/view?usp=sharing)\
[ChemProt](https://drive.google.com/file/d/1XSieVU673Ey52xSV16pZ7a_8fqBJFd6k/view?usp=sharing)

### 2. Sub-domain pre-training data: 
[PPI](https://drive.google.com/file/d/1dn2yDKj7-3SsyKQ5Zm_5sTlLxTCfqQpy/view?usp=sharing)\
[DDI](https://drive.google.com/file/d/1f03yS_hTY5-lGR4N9siDYjalAeZrler8/view?usp=sharing)\
[ChemProt](https://drive.google.com/file/d/1KitpphP5B9wKN01NoiKg65z11vckeoka/view?usp=sharing)

```
#!/usr/bin/env bash

STORAGE_BUCKET=gs://subbert_file
#mlm+nsp
BERT_BASE_DIR=$STORAGE_BUCKET/pubmedbert

output_dir=$STORAGE_BUCKET/pubmedbert_gene
pretraining_file=$STORAGE_BUCKET/data/gene_protein_sentence_nltk_wwm


#python3 run_pretraining.py  --input_file=${pretraining_file}1.tfrecord  --output_dir=${output_dir}  --do_train=True --do_eval=True --bert_config_file=${BERT_BASE_DIR}/bert_config.json --init_checkpoint=$BERT_BASE_DIR/model.ckpt --train_batch_size=192  --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=10000 --num_warmup_steps=10  --learning_rate=2e-5 --use_tpu=True --tpu_name=subbert

for i in {7..9}
do
	ls ${output_dir}
	#python create_pretraining_data.py --input_file=/usa/psu/Documents/BERT/biobert/data/drug_sentence_nltk${i}.txt  --output_file=./data/drug_sentence_nltk_mlm${i}.tfrecord  --vocab_file='/usa/psu/Documents/BERT/biobert/biobert_v1.1_pubmed/vocab.txt'  --do_lower_case=true  --max_seq_length=128  --max_predictions_per_seq=20   --masked_lm_prob=0.15  --random_seed=12345  --dupe_factor=5
	python3 run_pretraining.py  --input_file=${pretraining_file}$((${i}+1)).tfrecord  --output_dir=$output_dir  --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=${output_dir}/model.ckpt-${i}0000 --train_batch_size=192  --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=$((${i}+1))0000 --num_warmup_steps=10  --learning_rate=2e-5 --use_tpu=True --tpu_name=subbert
	
	

done

```

```
python run_re.py --task_name=$TASK_NAME --do_train=true --do_eval=false --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-10000 --max_seq_length=128 --train_batch_size=16 --learning_rate=2e-5 --num_train_epochs=${s} --do_lower_case=false --data_dir=${RE_DIR}${i} --output_dir=${OUTPUT_DIR}${i} --model_name="attention_last_layer"

```
