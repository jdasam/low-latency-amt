"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn

from .lstm import BiLSTM
from .mel import melspectrogram


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class ConvStackShortA(ConvStack):
    def __init__(self, input_features, output_features):
        super().__init__(input_features, output_features)
        self.cnn[0] = nn.Conv2d(1, output_features // 16, (1, 3), padding=(0, 1))

class ConvStackShortB(ConvStack):
    def __init__(self, input_features, output_features):
        super().__init__(input_features, output_features)
        self.cnn[3] = nn.Conv2d(output_features // 16, output_features // 16, (1, 3), padding=(0, 1))

class ConvStackShortC(ConvStack):
    def __init__(self, input_features, output_features):
        super().__init__(input_features, output_features)
        self.cnn[8] = nn.Conv2d(output_features // 16, output_features // 8, (1, 3), padding=(0, 1))

class ConvStackShortD(ConvStack):
    def __init__(self, input_features, output_features):
        super().__init__(input_features, output_features)
        self.cnn[0] = nn.Conv2d(1, output_features // 16, (1, 3), padding=(0, 1))
        self.cnn[3] = nn.Conv2d(output_features // 16, output_features // 16, (1, 3), padding=(0, 1))

class ConvStackShortE(ConvStack):
    def __init__(self, input_features, output_features):
        super().__init__(input_features, output_features)
        self.cnn[3] = nn.Conv2d(output_features // 16, output_features // 16, (1, 3), padding=(0, 1))
        self.cnn[8] = nn.Conv2d(output_features // 16, output_features // 8, (1, 3), padding=(0, 1))

class ConvStackShortF(ConvStack):
    def __init__(self, input_features, output_features):
        super().__init__(input_features, output_features)
        self.cnn[0] = nn.Conv2d(1, output_features // 16, (1, 3), padding=(0, 1))
        self.cnn[8] = nn.Conv2d(output_features // 16, output_features // 8, (1, 3), padding=(0, 1))

class ConvStackShortG(ConvStack):
    def __init__(self, input_features, output_features):
        super().__init__(input_features, output_features)
        self.cnn[0] = nn.Conv2d(1, output_features // 16, (1, 3), padding=(0, 1))
        self.cnn[3] = nn.Conv2d(output_features // 16, output_features // 16, (1, 3), padding=(0, 1))
        self.cnn[8] = nn.Conv2d(output_features // 16, output_features // 8, (1, 3), padding=(0, 1))


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStackShort(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStackShort(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStackShort(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStackShort(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def  run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator


class ARTranscriber(nn.Module):
    def __init__(self, input_features, output_features,
                 model_complexity_conv=48, model_complexity_lstm=48,
                 use_dp=False, acoustic_model_name='ConvStack'):
        super().__init__()

        def _sequence_model(input_size, output_size):
            return BiLSTM(input_size, output_size // 2)

        self.input_features = input_features
        self.output_features = output_features
        self.model_complexity_conv = model_complexity_conv
        self.model_complexity_lstm = model_complexity_lstm

        model_size_conv = model_complexity_conv * 16
        model_size_lstm = model_complexity_lstm * 16
        self.language_hidden_size = model_size_lstm
        # self.acoustic_model = ConvStackShortC(input_features, model_size_conv)
        self.acoustic_model = globals()[acoustic_model_name](input_features, model_size_conv)
        self.language_model = torch.nn.LSTM(model_size_conv + 88*2, model_size_lstm, num_layers=2, batch_first=True, bidirectional=False)   # hidden size 768, num layers 2
        self.language_model.flatten_parameters()

        self.language_post = nn.Sequential(
            torch.nn.Linear(model_size_lstm, 88 * 5)
        )

        self.class_embedding = nn.Embedding(5,2)
        self.criterion = nn.CrossEntropyLoss()
        self.melspectrogram = melspectrogram

    def forward(self, mel, gt_label=False): # [gt_label: 1x640x88]
        acoustic_out = self.acoustic_model(mel) #[mel: 1x640x229, acoustic_out: 1x640x768]
        if not isinstance(gt_label, bool):
            prev_gt = torch.cat((torch.zeros((gt_label.shape[0], 1, gt_label.shape[2]), device=mel.device, dtype=torch.long), gt_label[:, :-1, :].type(torch.LongTensor).to(mel.device)), dim=1)
            # prev_gt = torch.cat((torch.zeros((gt_label.shape[0], 1, gt_label.shape[2]), device=mel.device, dtype=torch.long), gt_label[:, :-1, :]), dim=1)
            embedded_prev_gt = self.class_embedding(prev_gt) # [1, 640, 88, 2] 
            embedded_prev_gt_flattened = embedded_prev_gt.view(mel.shape[0], mel.shape[1], 88 * 2) # -> [1, 640, 88*2]
            concated_data = torch.cat((acoustic_out, embedded_prev_gt_flattened), dim=2) # [1 640 944]
            # self.language_model.flatten_parameters()
            result, _= self.language_model(concated_data) # [1, 640, 1536], [1, 1, 1536], [1, 1, 1536]
            total_result = self.language_post(result).view(mel.shape[0], -1, 88, 5) # [1, 640, 88, 5]
            # total_result = torch.log_softmax(total_result, dim=3)
            # total_result = torch.argmax(result, dim=3)
        else:
            h, c= self.init_lstm_hidden(mel.shape[0], mel.device)
            prev_out =  torch.zeros((mel.shape[0], 1, 88*2)).to(mel.device)
            total_result = torch.zeros((mel.shape[0], mel.shape[1], 88 )).to(mel.device)
            for i in range(acoustic_out.shape[1]):
                current_data = torch.cat((acoustic_out[:,i:i+1,:], prev_out), dim=2)
                current_out, (h, c) = self.language_model(current_data, (h, c))
                current_out = self.language_post(current_out)
                current_out = current_out.view((mel.shape[0], 1, 88, 5))
                # current_out = torch.softmax(current_out, dim=3)
                current_out = torch.argmax(current_out, dim=3)
                prev_out = self.class_embedding(current_out).view(mel.shape[0], 1, 88*2)
                total_result[:,i:i+1,:] = current_out
            
        return total_result 

    def run_on_batch(self, batch, evaluation=False, label_shift=0):
        audio_label = batch['audio']
        if 'mel' in batch.keys():
            mel_label = batch['mel']
        else:
            batch['mel'] = self.melspectrogram(audio_label
                                          .reshape(-1,
                                                   audio_label
                                                   .shape[-1])[:, :-1])\
                           .transpose(-1, -2)
            mel_label = batch['mel']
        state_label = batch['label']
        if evaluation:
            label_pred = self(mel_label)
            return label_pred
        else:
            label_pred = self(mel_label, state_label) 

        pred = label_pred.permute(0,3,1,2)
        target = state_label.type(torch.LongTensor).to(label_pred.device)
        if label_shift > 0:
            loss = self.criterion(pred[:,:,:-label_shift],target[:,:-label_shift])
        else:
            loss = self.criterion(pred, target)


        return label_pred, {'main': loss}

    def save(self, filename):
        acoustic_state_dict = self.acoustic_model.module.state_dict()
        language_state_dict = self.language_model.state_dict()
        post_state_dict = self.language_post.module.state_dict()
        embedding_state_dict = self.class_embedding.state_dict()

        state_dict_list = [acoustic_state_dict, language_state_dict, post_state_dict, embedding_state_dict]
        for state_dict in state_dict_list:
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()

        parameters = dict(acoustic=acoustic_state_dict, language=language_state_dict,
                          post=post_state_dict, embedding=embedding_state_dict,
                        #   velocity=velocity_state_dict,
                          input_features=self.input_features,
                          output_features=self.output_features,
                          model_complexity_conv=self.model_complexity_conv,
                          model_complexity_lstm=self.model_complexity_lstm)

        torch.save(parameters, filename)

    def init_lstm_hidden(self, batch_size, device):
        h = torch.zeros(2, batch_size, self.language_hidden_size, device=device)
        c = torch.zeros(2, batch_size, self.language_hidden_size, device=device)
        return (h, c)

