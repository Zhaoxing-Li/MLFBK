import numpy as np
import pandas as pd
from os.path import exists
from .IKT import FeatureEngineering

from torch.utils.data import Dataset

DATASET_DIR = "../data/algebra06/preprocessed_df.csv"
IKT_DIR = "../data/algebra06/IKT/"

class FeatureEnumerator:
    def __init__(self):
        train_file = IKT_DIR + 'ikt_features_train.csv'
        test_file = IKT_DIR + 'ikt_features_test.csv'
        valid_file = IKT_DIR + 'ikt_features_valid.csv'
        df1 = pd.read_csv(train_file)
        df2 = pd.read_csv(test_file)
        df3 = pd.read_csv(valid_file)
        df = pd.concat([df1, df2, df3], sort=False)
        
        self.u_list = np.unique(df["user_id"].values)
        self.q_list = np.unique(df["skill_id"].values)
        self.r_list = np.unique(df["correctness"].values)
        self.pid_list = np.unique(df["item_id"].values)
        self.ap_list = np.unique(df["ability_profile"].values)
        self.pd_list = np.unique(df["problem_difficulty"].values)
        self.sm_list = np.arange(101)

        self.u2idx = {u: idx for idx, u in enumerate(self.u_list)}
        self.q2idx = {q: idx for idx, q in enumerate(self.q_list)}
        self.pid2idx = {pid: idx for idx, pid in enumerate(self.pid_list)}
        self.ap2idx = {ap: idx for idx, ap in enumerate(self.ap_list)}
        self.pd2idx = {pdf: idx for idx, pdf in enumerate(self.pd_list)}
        self.sm2idx = {sm: idx for idx, sm in enumerate(self.sm_list)}

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_r = self.r_list.shape[0]
        self.num_pid = self.pid_list.shape[0]
        self.num_ap = self.ap_list.shape[0]
        self.num_pd = self.pd_list.shape[0]
        self.num_sm = self.sm_list.shape[0]

class ALGEBRA06_PID(Dataset):
    def __init__(self, max_seq_len, dataset_df, save_name, fe) -> None:
        super().__init__()

        self.dataset_df = dataset_df
        self.save_name = save_name
        self.fe = fe

        file_name = IKT_DIR + f'ikt_features_{self.save_name}.csv'
        self.ikt_features = pd.read_csv(file_name)

        self.u2idx = self.fe.u2idx
        self.q2idx = self.fe.q2idx
        self.pid2idx = self.fe.pid2idx
        self.ap2idx = self.fe.ap2idx
        self.pd2idx = self.fe.pd2idx
        self.sm2idx = self.fe.sm2idx

        self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.r_list, self.pid_seqs, self.pid_list, \
            self.ap_seqs, self.ap_list, self.pd_seqs, self.pd_list, self.sm_seqs, self.sm_list = self.preprocess()

        self.num_u = self.fe.num_u
        self.num_q = self.fe.num_q
        self.num_r = self.fe.num_r
        self.num_pid = self.fe.num_pid
        self.num_ap = self.fe.num_ap
        self.num_pd = self.fe.num_pd
        self.num_sm = self.fe.num_sm

        self.q_seqs, self.r_seqs, self.pid_seqs, self.ap_seqs, self.pd_seqs, self.sm_seqs = \
            self.match_seq_len(self.q_seqs, self.r_seqs, self.pid_seqs, self.ap_seqs, self.pd_seqs, self.sm_seqs, max_seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index], self.pid_seqs[index], self.ap_seqs[index], self.pd_seqs[index], self.sm_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = self.ikt_features
        df = df[(df["correctness"] == 0) | (df["correctness"] == 1)]

        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["skill_id"].values)
        r_list = np.unique(df["correctness"].values)
        pid_list = np.unique(df["item_id"].values)
        ap_list = np.unique(df["ability_profile"].values)
        pd_list = np.unique(df["problem_difficulty"].values)
        sm_list = np.arange(101)

        q_seqs = []
        r_seqs = []
        pid_seqs = []
        ap_seqs = []
        pd_seqs = []
        sm_seqs = []

        for u in u_list:
            df_u = df[df["user_id"] == u]

            q_seq = np.array([self.q2idx[q] for q in df_u["skill_id"].values])
            r_seq = df_u["correctness"].values
            pid_seq = np.array([self.pid2idx[pid] for pid in df_u["item_id"].values])
            ap_seq = np.array([self.ap2idx[ap] for ap in df_u["ability_profile"].values])
            pd_seq = np.array([self.pd2idx[pdf] for pdf in df_u["problem_difficulty"].values])
            sm_seq = np.array([sm * 100 for sm in df_u["skill_mastery"].values])

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
            pid_seqs.append(pid_seq)
            ap_seqs.append(ap_seq)
            pd_seqs.append(pd_seq)
            sm_seqs.append(sm_seq)

        return q_seqs, r_seqs, q_list, u_list, r_list, pid_seqs, pid_list, ap_seqs, ap_list, pd_seqs, pd_list, sm_seqs, sm_list

    def match_seq_len(self, q_seqs, r_seqs, pid_seqs, ap_seqs, pd_seqs, sm_seqs, max_seq_len, pad_val=-1):

        proc_q_seqs = []
        proc_r_seqs = []
        proc_pid_seqs = []
        proc_ap_seqs = []
        proc_pd_seqs = []
        proc_sm_seqs = []

        for q_seq, r_seq, pid_seq, ap_seq, pd_seq, sm_seq in zip(q_seqs, r_seqs, pid_seqs, ap_seqs, pd_seqs, sm_seqs):

            i = 0
            while i + max_seq_len < len(q_seq):
                proc_q_seqs.append(q_seq[i:i + max_seq_len])
                proc_r_seqs.append(r_seq[i:i + max_seq_len])
                proc_pid_seqs.append(pid_seq[i:i + max_seq_len])
                proc_ap_seqs.append(ap_seq[i:i + max_seq_len])
                proc_pd_seqs.append(pd_seq[i:i + max_seq_len])
                proc_sm_seqs.append(sm_seq[i:i + max_seq_len])

                i += max_seq_len

            proc_q_seqs.append(
                np.concatenate(
                    [
                        q_seq[i:], 
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_r_seqs.append(
                np.concatenate(
                    [
                        r_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_pid_seqs.append(
                np.concatenate(
                    [
                        pid_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_ap_seqs.append(
                np.concatenate(
                    [
                        ap_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_pd_seqs.append(
                np.concatenate(
                    [
                        pd_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
            proc_sm_seqs.append(
                np.concatenate(
                    [
                        sm_seq[i:],
                        np.array([pad_val] * (i + max_seq_len - len(q_seq)))
                    ]
                )
            )
        return proc_q_seqs, proc_r_seqs, proc_pid_seqs, proc_ap_seqs, proc_pd_seqs, proc_sm_seqs
    
class IKT_HANDLER:
    def __init__(self, dataset, save_name):
        self.dataset_df = dataset
        self.save_name = save_name

        file_name = IKT_DIR + f'ikt_features_{self.save_name}.csv'
        if not exists(file_name):
            self.ikt_features = self.get_ikt_features(file_name)

    def format_to_ikt(self):
        data = self.dataset_df
        users = data.user_id.unique()
        skill_seq = []
        question_seq = []
        response_seq = []
        for user in users:
            user_df = data[data.user_id == user]
            skill_pre = user_df.skill_id.values.tolist()
            skill_seq = ','.join(list(map(str, skill_pre)))
            question_seq = ','.join(list(map(str, user_df.item_id.values.tolist())))
            response_seq = ','.join(list(map(str, user_df.correct.values.tolist())))
            line1 = f"{len(skill_pre)}, {user}"

            # write the lines to the file
            file_name = IKT_DIR + f'algebra06_ikt_{self.save_name}.csv'
            with open(file_name, 'a') as csv_file:
                csv_file.write(f'{line1}\n')
                csv_file.write(f'{skill_seq}\n')
                csv_file.write(f'{question_seq}\n')
                csv_file.write(f'{response_seq}\n')
        return file_name

    def get_ikt_features(self, file_name):
        csv_file_name = self.format_to_ikt()
        ikt_features_l = FeatureEngineering.main(csv_file_name)
        ikt_features_l.to_csv(file_name)
        return ikt_features_l


def ALGEBRA06_PID_SPLIT(max_seq_len, dataset_dir=DATASET_DIR):
    df = pd.read_csv(dataset_dir, delim_whitespace=True)

    # split into training, testing, validation data
    train_amount = int(df.shape[0] * 0.7)
    test_amount = int(df.shape[0] * 0.2)
    train_data = df.iloc[:train_amount, :]
    test_data = df.iloc[train_amount:train_amount+test_amount, :]
    valid_data = df.iloc[train_amount+test_amount:, :]

    # generate the IKT features
    IKT_HANDLER(train_data, "train")
    IKT_HANDLER(test_data, "test")
    IKT_HANDLER(valid_data, "valid")

    # get the number of items etc for embedding
    fe = FeatureEnumerator()

    # create the ALGEBRA06_PID loaders from this data
    validation_loader = ALGEBRA06_PID(max_seq_len, valid_data, "valid", fe)
    train_loader = ALGEBRA06_PID(max_seq_len, train_data, "train", fe)
    test_loader = ALGEBRA06_PID(max_seq_len, test_data, "test", fe)

    return train_loader, test_loader, validation_loader, fe

if __name__ == "__main__":
    max_s_len = 100

    