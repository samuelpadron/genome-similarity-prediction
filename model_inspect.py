import os
import torch
import huggingface

def main():
    num_epochs = 100  # ~100 seems fine
    max_length = 500  # max len of sequence of dataset (of what you want) ~ should experiment with this
    use_padding = True
    batch_size = 256
    learning_rate = 6e-4  # good default for Hyena
    rc_aug = True  # reverse complement augmentation
    add_eos = False  # add end of sentence token
    weight_decay = 0.1
    
    # for fine-tuning, only the 'tiny' model can fit on colab
    pretrained_model_name = 'hyenadna-tiny-1k-seqlen'
    backbone_cfg = None
    device = torch.device('cpu')
    use_head = True
    n_classes = 2

    save_path = os.path.join(os.getcwd(), "trained_model.pth")
    model = huggingface.HyenaDNAPreTrainedModel.from_pretrained(
                '/scratch/spadronalcala',
                pretrained_model_name,
                download=True,
                config=backbone_cfg,
                device=device,
                use_head=use_head,
                n_classes=n_classes,
            )
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(model.state_dict())

if __name__ == "__main__":
    main()
