import os
from datetime import datetime

import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pathlib import Path
import argparse
import wandb
from evaluate import evaluate
from onsets_and_frames import *



def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/dasol/userdata/low-latency-transcription/onsets-and-frames/data/MAESTRO',
                        help='directory path to the dataset')

    parser.add_argument('--acoustic_model_name', type=str, default='ConvStack')
    parser.add_argument('--label_shift_frame', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.0006)

    parser.add_argument('--train_on', type=str, default='MAESTRO')
    parser.add_argument('--resume_iteration', type=None, default=None)
    parser.add_argument('--checkpoint_interval', type=int, default=10000)

    parser.add_argument('--sequence_length', type=int, default=327680)
    parser.add_argument('--model_complexity', type=int, default=48)
    parser.add_argument('--learning_rate_decay_steps', type=int, default=10000)

    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.98)
    parser.add_argument('--leave_one_out', type=None, default=None)
    parser.add_argument('--clip_gradient_norm', type=int, default=3)
# TODO validation length?
    parser.add_argument('--validation_interval', type=int, default=1000)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=Path, default=Path('experiments/'))
    parser.add_argument('--no_log', action='store_true')



    # parser.add_argument('--num_epoch_per_log', type=int, default=1)
    # parser.add_argument('--num_iter_per_valid', type=int, default=3000)

    # parser.add_argument('--model_type', type=str, default='pitch_dur')
    # parser.add_argument('--model_name', type=str, default='pitch_dur')

    # parser.add_argument('--yml_path', type=str, default='yamls/test.yaml',
    #                     help='yaml path to the config')

    # parser.add_argument('--scheduler_factor', type=float, default=0.3)
    # parser.add_argument('--scheduler_patience', type=int, default=3)   
    #  
    # parser.add_argument('--hidden_size', type=int, default=256)
    # parser.add_argument('--num_layers', type=int, default=3)
    # parser.add_argument('--dropout', type=float, default=0.1)
    # parser.add_argument('--aug_type', type=str, default='stat')
    return parser

def update_args(args): 
    args.logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # if args.device == 'cuda':
    #   args.device = f'cuda:{args.gpu_id}'
    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        args.batch_size //= 2
        args.sequence_length //= 2
        print(f'Reducing batch size to {args.batch_size} and sequence_length to {args.sequence_length} to save memory')

    args.validation_length = args.sequence_length

    return args


def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval, acoustic_model_name, label_shift):

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    if train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length, label_shift=label_shift)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length, label_shift=label_shift)
    else:
        dataset = MAPS(groups=['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2'], sequence_length=sequence_length, label_shift=label_shift)
        validation_dataset = MAPS(groups=['ENSTDkAm', 'ENSTDkCl'], sequence_length=validation_length)

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    if resume_iteration is None:
        # model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device)
        model = ARTranscriber(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity, acoustic_model_name=acoustic_model_name).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    for i, batch in zip(loop, cycle(loader)):
        predictions, losses = model.run_on_batch(batch, label_shift=label_shift)

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        wandb.log({"loss": loss.item()}, step=i)

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        # for key, value in {'loss': loss, **losses}.items():
        #     writer.add_scalar(key, value.item(), global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                valid_result = evaluate(validation_dataset, model, label_shift=label_shift)
                new_dict = {'validation/' + key.replace(' ', '_'): np.mean(value) for key, value in valid_result.items()}
                wandb.log(new_dict, step=i)
                # for key, value in evaluate(validation_dataset, model).items():
                #     writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
            # wandb.log({"validation_loss": loss})
            model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))




if __name__ == '__main__':
    args = get_argument_parser().parse_args()
    args = update_args(args)

    torch.manual_seed(args.seed)
    wandb.init(project='low-latency-transcription',
               name=f'baseline-test-{args.iterations}',
            #    tags=['just-try'])
    )
    wandb.config.update(args)

    train(args.logdir, 
          args.device, 
          args.iterations, 
          args.resume_iteration,
          args.checkpoint_interval, 
          args.train_on, 
          args.batch_size, 
          args.sequence_length, 
          args.model_complexity, 
          args.learning_rate, 
          args.learning_rate_decay_steps, 
          args.learning_rate_decay_rate, 
          args.leave_one_out, 
          args.clip_gradient_norm, 
          args.validation_length, 
          args.validation_interval,
          args.acoustic_model_name,
          args.label_shift_frame)
