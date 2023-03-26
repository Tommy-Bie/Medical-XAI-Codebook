from datasets.SkinCon import SkinCon


def select_dataset(args):
    if args.dataset == "SkinCon":
        dataset_train = SkinCon(args, train=True, transform=args.transform)
        dataset_val = SkinCon(args, train=False, transform=args.transform)
        return dataset_train, dataset_val

    # TODO: other medical datasets

    raise ValueError(f'unknown {args.dataset}')
