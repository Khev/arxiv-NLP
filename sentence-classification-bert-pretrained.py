import numpy as np
import pickle as pkl
from tqdm import tqdm, trange
from collections import defaultdict
import torch, io, gzip, json, random, argparse, os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import (BertTokenizer, BertConfig, AdamW, BertForSequenceClassification,
        WarmupLinearSchedule)
from ftfy import fix_text
from arxiv_public_data.config import DIR_BASE, DIR_OUTPUT, DIR_FULLTEXT

f_metadata = os.path.join(DIR_BASE, 'arxiv-metadata-oai-2019-03-01.json.gz')

#Got these from Matt. Some are redundant, oddly.
cat_map = {
  "astro-ph": "astro-ph",
  "cond-mat": "cond-mat",
  "cs": "cs",
  "gr-qc": "gr-qc",
  "hep-ex": "hep-ex",
  "hep-lat": "hep-lat",
  "hep-ph": "hep-ph",
  "hep-th": "hep-th",
  "math-ph": "math-ph",
  "nlin": "nlin",
  "nucl-ex": "nucl-ex",
  "nucl-th": "nucl-th",
  "physics": "physics",
  "quant-ph": "quant-ph",
  "math": "math",
  "q-bio": "q-bio",
  "q-fin": "q-fin",
  "stat": "stat",
  "eess": "eess",
  "econ": "econ",
  "acc-phys": "physics.acc-ph",
  "adap-org": "nlin.AO",
  "alg-geom": "math.AG",
  "ao-sci": "physics.ao-ph",
  "atom-ph": "physics.atom-ph",
  "bayes-an": "physics.data-an",
  "chao-dyn": "nlin.CD",
  "chem-ph": "physics.chem-ph",
  "cmp-lg": "cs.CL",
  "comp-gas": "nlin.CG",
  "dg-ga": "math.DG",
  "funct-an": "math.FA",
  "mtrl-th": "cond-mat.mtrl-sci",
  "patt-sol": "nlin.PS",
  "plasm-ph": "physics.plasm-ph",
  "q-alg": "math.QA",
  "solv-int": "nlin.SI",
  "supr-con": "cond-mat.supr-con"
}


def load_ith_fulltext(i):
     """ 
     Loads in the i-th fulltext as a string

     FILL THIS IN COLIN
     """

        
def clean_doc(x):
    return fix_text(x)


def load_data(N, fname, data_type):
    
    
    #MAX_LENS = [50, 250, 500]  #truncate all titles, abstracts, fulltext to this level
    #N, data_type = args.N, args.data_type
    #if data_type == 'title':
    #     MAX_LEN = MAX_LENS[0]
    #elif data_type == 'abstract':
    #     MAX_LEN = MAX_LENS[1]
    #elif data_type == 'fulltext':
    #     MAX_LEN = MAX_LENS[2]
    
    MAX_LEN = 512  #BERT default
    input_ids = []
    labels, label_dict, ctr = [], {}, 0
    attention_masks = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    with gzip.open(fname, 'rt', encoding='utf-8') as fin:
        for row in fin.readlines():

            #Load metadata
            m = json.loads(row)

            #Build label list
            if data_type != 'fulltext':
                sentence = clean_doc(m[data_type])
            else:
                sentence = load_ith_fulltext(i)  ###needs to be filled in
                sentence = clean_doc(sentence)

            # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
            sentence = "[CLS] " + sentence + " [SEP]" 

            #category
            category = m['categories'][0].split(' ')[0]

            #update cateogies -- apply matt's map
            if category in cat_map: category = cat_map[category]

            #Then add to the dics
            if category not in label_dict:
                index = len(label_dict)
                label_dict[category] = index  # ex: {'hep-ph':0, 'math.CO:1',,}
            else:
                index = label_dict[category]
            labels.append(index)


            #Tokenize
            tokenized_sentence = tokenizer.tokenize(sentence)  #Ex: ['the', 'cat', 'ate']

            #Convert to IDs + pad
            input_id = tokenizer.convert_tokens_to_ids(tokenized_sentence)  #Ex: [1,10,3]
            input_id = pad_sequences([input_id], maxlen=MAX_LEN, dtype="long",truncating="post",padding="post")
            input_ids.append(input_id[0])
            
            #Attention mask
            seq_mask = [float(i>0) for i in input_id[0]]
            attention_masks.append(seq_mask)
            
            #Ctr
            ctr += 1
            if ctr >= N: break
                
    return np.array(input_ids), attention_masks, labels, label_dict  


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def evaluate(preds,labels):

    #Compute stats
    logliklihood = 0
    top1, top3, top5 = 0,0,0
    indicies_top5 = np.flip(np.argsort(preds,axis=1),axis=1)[:,:5]
    for i,row in enumerate(indicies_top5):
        label = labels[i]
        index_max = row[0]
        logit = logits[i][index_max]
        prob = 1.0 / (1.0 + np.exp(-logit))
        logliklihood += np.log(prob)
        if row[0] == label:
            top1 += 1
        if label in row[:3]:
            top3 += 1
        if label in row[:5]:
            top5 += 1
    acc1, acc3, acc5 = top1 / len(labels), top3 / len(labels), top5 / len(labels)
    return acc1, acc3, acc5, logliklihood



#########################################################################################################################



if __name__ == '__main__':
    
    #Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='number of documents')
    parser.add_argument('data_type', type=str, help='options = [title,abstract]')
    parser.add_argument('--gpu', type=bool, default=True,  help='use GPU or not')
    parser.add_argument('--batch_size',type=int,default=4, help='number of samples per batch to GPU')
    parser.add_argument('--epochs', type=int, default = 2, help='number of epochs')

    args = parser.parse_args()
    gpu = args.gpu
    if gpu == True: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = torch.device("cpu")

    #Load and process data
    print('Loading & prepping data')
    N, data_type = args.N, args.data_type
    input_ids, attention_masks, labels, label_dict = load_data(N,f_metadata,data_type)
        
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                                random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=2018, test_size=0.1)

    
    
    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    
    
    # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
    batch_size = args.batch_size
    
    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for
    # loop with an iterator the entire dataset does not need to be loaded into memory
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    print('Loading & prepping data')
    
    #Model
    print("Loading model")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_dict))
    if gpu:
        model.cuda()
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    
    
    #Test and train
    # Store our loss and accuracy for plotting
    train_loss_set = []
    epochs = args.epochs 

    num_training_steps = epochs * len(train_dataloader)

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=0.1 * num_training_steps,
        t_total=num_training_steps
    )

    print("Beginning training")
    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        #Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            train_loss_set.append(loss)    
            #Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            scheduler.step()

        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        model.eval()

        #Tracking variables 
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

      #Evaluate data for one epoch
        acc1, acc3, acc5, logliklihood = 0,0,0,0
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            #evaluate
            acc1_temp, acc3_temp, acc5_temp, logliklihood_temp = evaluate(logits,label_ids)

            acc1 += acc1_temp
            acc3 += acc3_temp
            acc5 += acc5_temp
            logliklihood += logliklihood_temp
            nb_eval_steps += 1
        
    acc1 /= nb_eval_steps
    acc3 /= nb_eval_steps
    acc5 /= nb_eval_steps
    
    num_examples = len(batch)*nb_eval_steps
    perplexity = 2** (-logliklihood / num_examples / np.log(2))
    line = '{}: top1, top3, top5, perplexity = {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(data_type, acc1,acc3,acc5,perplexity)
    print(line)
   
    #Save data
    DIR_NAME = './output'
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    fname = os.path.join(DIR_NAME[2:],'classification-results-N-{}-{}.txt'.format(N,data_type))
    np.savetxt(fname,[line],fmt='%s')

