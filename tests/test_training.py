import yaml

def test_train_config():

    with open("configs/train_config.yaml") as f:
        cfg = yaml.safe_load(f)

    assert "model" in cfg
    assert "epochs" in cfg