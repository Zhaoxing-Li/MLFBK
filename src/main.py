import numpy as np
import datetime
import pandas as pd
import torch
from utils import get_optimizers, get_crits
from models.bertmodel import BERT
from trainers.bertmodel_trainer import BERT_Trainer
from get_modules.get_loaders import get_loaders

from define_argparser import define_argparser

def pad(l, size, padding):
    return l + [padding] * abs((len(l)-size))

def main(config):
    # 0. device setting
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_diff, num_ap, num_pd, num_sm = get_loaders(config)

    model = BERT(
        num_q=num_q,
        num_r=num_r,
        num_pid=num_pid,
        num_ap=num_ap,
        num_pd=num_pd,
        num_sm=num_sm,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        num_head=config.num_head,
        num_encoder=config.num_encoder,
        max_seq_len=config.max_seq_len,
        device=device,
        use_leakyrelu=config.use_leakyrelu,
        dropout_p=config.dropout_p
    ).to(device)

    optimizer = get_optimizers(model, config)
    crit = get_crits(config)

    trainer = BERT_Trainer(
        model=model,
        optimizer=optimizer,
        n_epochs=config.n_epochs,
        device=device,
        num_q=num_q,
        crit=crit,
        max_seq_len=config.max_seq_len,
        grad_acc=config.grad_acc,
        grad_acc_iter=config.grad_acc_iter
    )

    train_scores, valid_scores, \
        highest_valid_score, highest_test_score  = trainer.train(train_loader, valid_loader, test_loader, config)
    

    print(train_scores)
    print(valid_scores)
    print(f"The highest test score was: {highest_test_score}")

    print("saving training and validation scores")
    model_name = config.model_fn.split(".")[0]
    train_name = model_name + "_train"
    valid_name = model_name + "_valid"
    df = pd.read_csv("../data/training_process/training_process_data.csv")
    train_scores = pad(train_scores, 100, None)
    valid_scores = pad(valid_scores, 100, None)
    df[train_name] = train_scores
    df[valid_name] = valid_scores
    df.to_csv("../data/training_process/training_process_data.csv", index=False)

    print("finished saving training scores, now saving model")

    # save the model to a file
    today = datetime.datetime.today()
    record_time = str(today.month) + "_" + str(today.day) + "_" + str(today.hour) + "_" + str(today.minute)
    # model's path
    model_path = '../model_records/' + str(highest_test_score) + "_" + record_time + "_" + config.model_fn
    # model save
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config
    }, model_path)

if __name__ == "__main__":
    # get config from define_argparser
    config = define_argparser() 

    main(config)