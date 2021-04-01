### 1. Evaluation datasets for the PPI, DDI and ChemProt tasks: 
[PPI](https://drive.google.com/file/d/1dn2yDKj7-3SsyKQ5Zm_5sTlLxTCfqQpy/view?usp=sharing)\
[DDI](https://drive.google.com/file/d/1EEtN1LMI-W4iqtsXVfc64v5PsoAEmJad/view?usp=sharing)\
[ChemProt](https://drive.google.com/file/d/1XSieVU673Ey52xSV16pZ7a_8fqBJFd6k/view?usp=sharing)

### 2. Sub-domain pre-training data: 
[PPI](https://drive.google.com/file/d/1wto7L-SD7yzLvmKpWD2RcPwevFXr3IZ7/view?usp=sharing)\
[DDI](https://drive.google.com/file/d/1f03yS_hTY5-lGR4N9siDYjalAeZrler8/view?usp=sharing)\
[ChemProt](https://drive.google.com/file/d/1KitpphP5B9wKN01NoiKg65z11vckeoka/view?usp=sharing)

Due to the memory limit, we split the pre-training data into 10 files during sub-domain pre-training. If enough memory is available for you, just combine all the files into one file. Also, before pre-training, we need to covert the text data into tfrecord file for the BERT model using the following code, here we use PPI as an example. For pre-training of BioBERT model, we have to set "do_lower_case" to "false" and "do_whole_word_mask" to "false", and for PubMedBERT,  we have to set "do_lower_case" and "do_whole_word_mask" to "true".

```
for i in {1..10}
do
	python create_pretraining_data.py --input_file=./data/gene_protein_sentence_nltk${i}.txt  --output_file=./data/gene_protein_sentence_nltk_wwm${i}.tfrecord  --vocab_file='./pubmedbert/vocab.txt'  --do_lower_case=true --do_whole_word_mask=true  --max_seq_length=128  --max_predictions_per_seq=20   --masked_lm_prob=0.15  --random_seed=12345  --dupe_factor=5

done

```
After the creation of tfrecord files for BERT model, we then can run pre-training (here we use Google Cloud TPU): 
```
BERT_BASE_DIR="./pubmedbert"
output_dir="./pubmedbert_gene"
pretraining_file="./data/gene_protein_sentence_nltk_wwm"


python3 run_pretraining.py  --input_file=${pretraining_file}1.tfrecord  --output_dir=${output_dir}  --do_train=True --do_eval=True --bert_config_file=${BERT_BASE_DIR}/bert_config.json --init_checkpoint=$BERT_BASE_DIR/model.ckpt --train_batch_size=192  --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=10000 --num_warmup_steps=10  --learning_rate=2e-5 --use_tpu=True --tpu_name=subbert

for i in {1..9}
do
	python3 run_pretraining.py  --input_file=${pretraining_file}$((${i}+1)).tfrecord  --output_dir=$output_dir  --do_train=True --do_eval=True --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=${output_dir}/model.ckpt-${i}0000 --train_batch_size=192  --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=$((${i}+1))0000 --num_warmup_steps=10  --learning_rate=2e-5 --use_tpu=True --tpu_name=subbert

done
```

After the pre-training, we then can fine-tuning the BERT model on the evaluation sets of PPI, DDI and ChemProt:
$TASK_NAME='aimed' or 'ddi13' or 'chemprot';
$BERT_DIR is the path where we store the pre-trained BERT model using sub-domain data;
$OUTPUT_DIR is the path where we can store the fine-tuned BERT model;
```
python run_re.py --task_name=$TASK_NAME --do_train=true --do_eval=false --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BERT_DIR/bert_config.json --init_checkpoint=$BERT_DIR/model.ckpt-10000 --max_seq_length=128 --train_batch_size=16 --learning_rate=2e-5 --num_train_epochs=${s} --do_lower_case=false --data_dir=${RE_DIR}${i} --output_dir=${OUTPUT_DIR}${i} 

```
