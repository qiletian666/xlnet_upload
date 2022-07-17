"""
Copyright 2020 ICES, University of Manchester, Evenset Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#Code by Shuhao Qi
import os
from typing import final

import sklearn_crfsuite
import pickle
from pytorch_transformers import XLNetTokenizer

import utils
from ner_plugins.NER_abstract import NER_abstract
from utils.spec_tokenizers import tokenize_fa
from tqdm import tqdm_notebook as tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_transformers import XLNetModel
from torch.optim import Adam
from torch.autograd import Variable

class NERModel(nn.Module):

    def __init__(self, num_class=11):
        super(NERModel, self).__init__()
        self.model = XLNetModel.from_pretrained("./")
        self.dropout = nn.Dropout(0.1)
        self.l1 = nn.Linear(768,num_class)

    def forward(self, x, attention_mask=None):
        outputs = self.model(x, attention_mask=attention_mask)
        x = outputs[0]
        x = self.dropout(x)
        x = self.l1(x)
        return x


class NER_XLnet(NER_abstract):
    """
    The class for executing XLnet labelling based on i2b2 dataset (2014).

    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if  os.path.exists("Models/NER_XLnet1.pth"):
            self.model=torch.load("Models/NER_XLnet1.pth")
        else:
            self.model = NERModel(num_class=9)
        self._tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.label_dict={'O':0,'DATE':1,'NAME':2,'AGE':3,'LOCATION':4, 'CONTACT':5, 'PROFESSION':6,'X':7,'[PAD]':8}
        self.labels_map = ['O', 'DATE', 'NAME', 'AGE', 'LOCATION', 'CONTACT', 'PROFESSION', 'X', '[PAD]']
        pass

    def loss_function(self,logits, target, masks, num_class=9):
        criterion = nn.CrossEntropyLoss(reduction='none')
        logits = logits.view(-1, num_class)
        target = target.view(-1)
        masks = masks.view(-1)
        cross_entropy = criterion(logits, target)
        loss = cross_entropy * masks
        loss = loss.sum() / (masks.sum() + 1e-12)
        loss = loss.to(self.device)
        return loss



    def save_model(self,path):
        pickle.dump(self.model, open(path, 'wb'))

    def transform_sequences(self,tokens_labels):
        """
        Transforms sequences into the X and Y sets. For X it creates features, while Y is list of labels
        :param tokens_labels: Input sequences of tuples (token,lable)
        :return:
        """
        input_ids = []
        input_labels = []
        for tokens_label in tokens_labels:
            input_id=[]
            input_label=[]
            for id,label in tokens_label:
                token = self._tokenizer.tokenize(id)
                for i,j in enumerate(token):
                    if i == 0:
                        try:
                            input_id.append(self._tokenizer.convert_tokens_to_ids(j))
                            input_label.append(self.label_dict[label])
                        except KeyError:
                            input_label.append(0)
                    else:
                        input_label.append(7)
                        input_id.append(self._tokenizer.convert_tokens_to_ids(j))
            input_ids.append(input_id)
            input_labels.append(input_label)


        for j in range(len(input_ids)):
            # Pad sample data to length 512,the length can be adjusted to 1024 or 2048 depending on the conditions of the training equipment
            i = input_ids[j]
            if len(i) <= 512:
                input_ids[j].extend([0] * (512- len(i)))
            else:
                input_ids[j] = i[0:512]

        for j in range(len(input_labels)):
            # Pad sample data to length 512,the length can be adjusted to 1024 or 2048 depending on the conditions of the training equipment
            i = input_labels[j]
            if len(i) <= 512:
                input_labels[j].extend([8] * (512 - len(i)))
            else:
                input_labels[j] = i[0:512]

        return input_ids,input_labels




    def learn(self,X,Y,epochs =1):
        """
        Function for training XLnet algorithm
        :param X: Training set input tokens and features
        :param Y: Training set expected outputs
        :param epochs: Epochs are basically used to calculate max itteration as epochs*200
        :return:
        """
        train_set = TensorDataset(torch.LongTensor(X),
                                  torch.LongTensor(Y))
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=8,
                                  shuffle=True)

        self.model.to(self.device)
        self.model.train()

        optimizer = Adam(self.model.parameters(), lr=1e-5)


        for i in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = Variable(data).to(self.device), Variable(target).to(self.device)
                optimizer.zero_grad()
                # generate mask
                mask = []
                for sample in data:
                    mask.append([1 if i != 0 else 0 for i in sample])
                mask = torch.FloatTensor(mask).to(self.device)

                output = self.model(data, attention_mask=mask)

                # get model predictions
                pred = torch.argmax(output, dim=2)
                loss = self.loss_function(output, target, mask)
                loss.backward()
                optimizer.step()






    def save(self,model_path):
        """
        Function that saves the XLnet model using pickle
        :param model_path: File name in Models/ folder
        :return:
        """
        torch.save(self.model, "Models/" + model_path + "1.pth")
        print("Saved model to disk")





    def evaluate(self,X,Y):
        """
        Function that takes testing data and evaluates them by making classification report matching predictions with Y argument of the function
        :param X: Input sequences of words with features
        :param Y: True labels
        :return: Prints the classification report
        """


        eval_set = TensorDataset(torch.LongTensor(X),
                                  torch.LongTensor(Y))
        eval_loader = DataLoader(dataset=eval_set,
                                  batch_size=1,
                                  shuffle=False)
        self.model.eval()

        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(tqdm(eval_loader)):
            data = data.to(self.device)
            target = target.float().to(self.device)

            # generate mask
            mask = []
            for sample in data:
                mask.append([1 if i != 0 else 0 for i in sample])
            mask = torch.Tensor(mask).to(self.device)

            output = self.model(data, attention_mask=mask)

            # get model predictions
            pred = torch.argmax(output, dim=2)

            # Add the mask to the prediction result for easy calculation of accuracy
            pred = pred.float()
            pred = pred * mask
            target = target * mask

            pred = pred[:, 0:mask.sum().int().item()]
            target = target[:, 0:mask.sum().int().item()]

            correct += (pred == target).sum().item()
            total += mask.sum().item()

        print('Number of correctly classified labels:{},Total number of labels{},Accuracy:{:.2f}%'.format(
            correct, total, 100. * correct / total))

    def perform_NER(self,text):
        """
          Implemented function that performs named entity recognition using XLnet. Returns a sequence of tuples (token,label).

          :param text: text over which should be performed named entity recognition
          :type language: str

          """
        documents = [text]

        sequences = tokenize_fa(documents)
        X, Y = self.transform_sequences(sequences)
        eval_set = TensorDataset(torch.LongTensor(X),
                                 torch.LongTensor(Y))
        eval_loader = DataLoader(dataset=eval_set,
                                 batch_size=1,
                                 shuffle=False)
        self.model.eval()


        final_list = []
        for batch_idx, (data, target) in enumerate(tqdm(eval_loader)):
            data = data.to(self.device)
            # generate mask
            mask = []
            for sample in data:
                mask.append([1 if i != 0 else 0 for i in sample])
            mask = torch.Tensor(mask).to(self.device)
            output = self.model(data, attention_mask=mask)
            # get the model prediction result
            int_data = []
            for i in data.numpy()[0]:
                i1 = (int)(i)
                int_data.append(i1)
            tokens = self._tokenizer.convert_ids_to_tokens(int_data)

            pred = torch.argmax(output, dim=2)
            pred = pred.numpy()[0]

            tuple_list = []
            idx = 0
            for id, _ in sequences[batch_idx]:
                tuple_list.append((id, self.labels_map[pred[idx]]))
                token_len = len(self._tokenizer.tokenize(id))
                idx += token_len
            final_list.append(tuple_list)
        return final_list

