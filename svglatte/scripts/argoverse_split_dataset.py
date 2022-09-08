import argparse

import torch


def main(args):
    """
    Splits the given sequences dataset randomly into two subsets.
    
    We used it to split the training dataset into two so that we
    could use one of the subsets for early stopping during hparams
    search. That way we would use the validation subset for model
    selection and the test dataset for the final model evaluation.
    """
    torch.random.manual_seed(args.seed)

    ds = torch.load(args.src_dataset_path)
    assert len(ds) > 0

    indices = torch.randperm(len(ds))
    indices_split1 = indices[:int(args.ratio * len(ds))]
    indices_split2 = indices[int(args.ratio * len(ds)):]
    assert len(indices_split1) > 0
    assert len(indices_split2) > 0
    assert indices_split1[0] not in indices_split2

    torch.save([ds[i] for i in indices_split1], args.out_dataset_split_1)
    torch.save([ds[i] for i in indices_split2], args.out_dataset_split_2)
    torch.save([indices_split1, indices_split2], args.out_dataset_split_indices)
    assert len(torch.load(args.out_dataset_split_1)) + len(torch.load(args.out_dataset_split_2)) == len(ds)
    assert sum(len(x) for x in torch.load(args.out_dataset_split_indices))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=float, default=72)
    parser.add_argument('--ratio', type=float, default=0.7)
    parser.add_argument('--src_dataset_path', type=str, default='data/argoverse/train.sequences.torchsave')
    parser.add_argument('--out_dataset_split_1', type=str, default='data/argoverse/train.split_1.sequences.torchsave')
    parser.add_argument('--out_dataset_split_2', type=str, default='data/argoverse/train.split_2.sequences.torchsave')
    parser.add_argument('--out_dataset_split_indices', type=str, default='data/argoverse/train.split_indices.torchsave')

    args = parser.parse_args()
    print(f"Args: {args}")
    main(args)
