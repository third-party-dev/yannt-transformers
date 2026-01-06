"""
def register(subparsers):
    p = subparsers.add_parser("foo", help="Foo command")
    p.add_argument("--x", type=int, required=True)
    p.set_defaults(func=run)

def run(args):
    print(args.x * 2)
"""


def register_yannt_transformers(subparsers):
    transformers_parser = subparsers.add_parser(
        "transformers", help="transformers command"
    )
    # transformers_parser.add_argument("--breakpoint",
    #     dest="breakpoint",
    #     action="store_true",
    #     help="breakpoint() after operation"
    # )
    transformers_subparser = transformers_parser.add_subparsers(
        dest="transformers_command", required=True
    )

    transformers_list_parser = transformers_subparser.add_parser(
        "list", help="transformers list command"
    )
    transformers_list_parser.set_defaults(func=transformers_list)

    transformers_create_parser = transformers_subparser.add_parser(
        "create", help="transformers create command"
    )
    transformers_create_parser.add_argument("--type", dest="model_type", required=True)
    transformers_create_parser.add_argument("--model", dest="model_name", required=True)
    transformers_create_parser.add_argument(
        "--pytorch_state_path", dest="pytorch_state_path", default=""
    )
    transformers_create_parser.add_argument(
        "--safetensors_path", dest="safetensors_path", default=""
    )
    transformers_create_parser.add_argument(
        "--max_shard", dest="max_shard", default="2147483648"
    )
    transformers_create_parser.set_defaults(func=transformers_create)


def transformers_list(args):
    from thirdparty.yannt.transformers.lib import TransformersModelFactory

    print("Indexing all of the transformer types available. (Takes a moment.)")
    factory = TransformersModelFactory()
    print("Index Complete.")

    long_list = []
    for model_type in factory.auto_model_combos:
        for model_name in factory.auto_model_combos[model_type]:
            long_list.append(f"model: {model_name} type: {model_type}")

    for entry in sorted(long_list):
        print(entry)

    if hasattr(args, "breakpoint") and args.breakpoint:
        print(f"Locals: {list(locals().keys())}")
        breakpoint()


def transformers_create(args):
    import os

    from thirdparty.yannt.transformers.lib import TransformersModelFactory

    print("Indexing all of the transformer types available. (Takes a moment.)")
    factory = TransformersModelFactory()

    if args.model_type not in factory.auto_model_combos:
        print("Bad --type value. Please reference the 'list' command.")
        exit(1)

    if args.model_name not in factory.auto_model_combos[args.model_type]:
        print("Bad --model value. Please reference the 'list' command.")
        exit(1)

    # combo = factory.auto_model_combos[args.model_type][args.model_name]
    tmodel = factory.reconstruct_model(args.model_type, args.model_name)

    if len(args.safetensors_path) > 0:
        print("Serializing to safetensors.")
        if not os.path.exists(args.safetensors_path):
            os.makedirs(args.safetensors_path, exist_ok=True)
        tmodel.save_torch_safetensors(
            args.safetensors_path, max_shard=int(args.max_shard)
        )

    if len(args.pytorch_state_path) > 0:
        print("Serializing model state to pytorch.")
        if not os.path.exists(args.pytorch_state_path):
            os.makedirs(args.pytorch_state_path, exist_ok=True)
        tmodel.save_torch_state(
            args.pytorch_state_path, f"{args.model_name}-{args.model_type}"
        )

    if hasattr(args, "breakpoint") and args.breakpoint:
        print(f"Locals: {list(locals().keys())}")
        breakpoint()
