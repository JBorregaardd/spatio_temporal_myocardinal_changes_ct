from spatio_temporal_myocardinal_changes_ct.model import Model
from spatio_temporal_myocardinal_changes_ct.data import MyDataset

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
