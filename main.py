from model import ElecGraph

FILE = './data/elec_flow_input.json'
EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP=5
EPOCH = 500

if __name__ == "__main__":

    elec = ElecGraph(file=FILE,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    khop=KHOP,
                    epochs=EPOCH)
    