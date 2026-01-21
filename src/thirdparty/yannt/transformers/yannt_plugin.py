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
    transformers_parser.add_argument("--breakpoint", dest="breakpoint", action="store_true", default=False)
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
    transformers_create_parser.add_argument("--onnx_path", dest="onnx_path", default="")
    transformers_create_parser.add_argument("--export_path", dest="export_path", default="")
    transformers_create_parser.add_argument(
        "--max_shard", dest="max_shard", default="2147483648"
    )
    transformers_create_parser.set_defaults(func=transformers_create)


    transformers_graph_parser = transformers_subparser.add_parser(
        "graph", help="transformers create command"
    )
    transformers_graph_parser.add_argument("--type", dest="model_type", required=True)
    transformers_graph_parser.add_argument("--model", dest="model_name", required=True)
    graph_choices = ["graphviz", "human_ir", "compiler_ir", "nodes", "drawer", "schema", "coverage"]
    transformers_graph_parser.add_argument("--as", choices=graph_choices, default="ir", dest="graph_as")
    transformers_graph_parser.add_argument("--input", choices=["ones", "dumb"], default="ones", dest="inp_type")
    transformers_graph_parser.add_argument("--out", dest="output_path", default=None)
    transformers_graph_parser.set_defaults(func=transformers_graph)




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


def _transformers_build_model(args):
    import sys
    from thirdparty.yannt.transformers.lib import TransformersModelFactory

    print("Indexing all of the transformer types available. (Takes a moment.)", file=sys.stderr)
    factory = TransformersModelFactory()

    if args.model_type not in factory.auto_model_combos:
        print("Bad --type value. Please reference the 'list' command.")
        exit(1)

    if args.model_name not in factory.auto_model_combos[args.model_type]:
        print("Bad --model value. Please reference the 'list' command.")
        exit(1)

    # combo = factory.auto_model_combos[args.model_type][args.model_name]
    tmodel = factory.reconstruct_model(args.model_type, args.model_name)

    return tmodel


def transformers_create(args):
    import os

    tmodel = _transformers_build_model(args)

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

    # yannt transformers create --model bert --type AutoModelForMaskedLM --onnx_path outputs/bert-onnx
    if len(args.onnx_path) > 0:
        print("Serializing model state to onnx.")
        if not os.path.exists(args.onnx_path):
            os.makedirs(args.onnx_path, exist_ok=True)
        tmodel.save_torch_onnx(
            args.onnx_path, f"{args.model_name}-{args.model_type}"
        )

    # yannt transformers create --model bert --type AutoModelForMaskedLM --onnx_path outputs/bert-onnx
    if len(args.export_path) > 0:
        print("Serializing model state to Export IR.")
        if not os.path.exists(args.export_path):
            os.makedirs(args.export_path, exist_ok=True)
        tmodel.save_export(
            args.export_path, f"{args.model_name}-{args.model_type}"
        )

    if hasattr(args, "breakpoint") and args.breakpoint:
        print(f"Locals: {list(locals().keys())}")
        breakpoint()



def collapse_ranges(nums):
    if not nums:
        return []

    nums = sorted(nums)  # make sure list is sorted
    result = []
    start = prev = nums[0]

    for n in nums[1:]:
        if n == prev + 1:  # consecutive
            prev = n
        else:
            # end of a consecutive run
            if start == prev:
                result.append(str(start))
            else:
                result.append(f"{start}-{prev}")
            start = prev = n

    # handle the last run
    if start == prev:
        result.append(str(start))
    else:
        result.append(f"{start}-{prev}")

    return result


import torch
from transformers import AutoModel

# Note: This function originally from GPT, but reviewed and massaged. (Not exhaustively verified.)
def generate_dummy_inputs(model, batch_size=None, seq_len=None, image_size=None):
    # TODO: Consider checking max_position_embedding to validate seq_len

    config = model.config
    input_kwargs = {}

    # --- generate batch_size and seq_len ---
    batch_size = batch_size or 1
    if hasattr(config, "max_position_embeddings"):
        seq_len = seq_len or min(16, config.max_position_embeddings)
    elif hasattr(config, "n_positions"):
        seq_len = seq_len or min(16, config.n_positions)
    else:
        seq_len = seq_len or 16


    # --- Text Models ---
    # tokens
    if hasattr(config, "vocab_size"):
        input_kwargs["input_ids"] = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)

    # attention_mask
    if hasattr(config, "hidden_size") and "input_ids" in input_kwargs:
        input_kwargs["attention_mask"] = torch.ones(batch_size, seq_len, dtype=torch.long)

    # token_type_ids
    if getattr(config, "type_vocab_size", None) is not None and "input_ids" in input_kwargs:
        input_kwargs["token_type_ids"] = torch.zeros(batch_size, seq_len, dtype=torch.long)

    # --- Vision / Image Models ---
    if hasattr(config, "num_channels") or hasattr(config, "image_size"):
        default_channels = 3
        default_height = 256
        deafult_width = 256
        channels = getattr(config, "num_channels", default_channels)
        height = getattr(config, "image_size", image_size[1] if image_size else default_height)
        width = getattr(config, "image_size", image_size[2] if image_size else default_width)
        input_kwargs["pixel_values"] = torch.randn(batch_size, channels, height, width)

    # --- Audio / speech models ---
    if getattr(config, "num_mel_bins", None) is not None:
        mel_bins = config.num_mel_bins
        input_kwargs["input_values"] = torch.randn(batch_size, seq_len, mel_bins)

    # --- Misc ---
    if getattr(config, "num_labels", None) is not None:
        input_kwargs["labels"] = torch.zeros(batch_size, dtype=torch.long)

    # Convert to tuple for positional arguments
    # Note: This feels janky.
    input_args = tuple(input_kwargs.values())
    #return input_args, input_kwargs
    return input_args


def transformers_graph(args):
    import os
    import torch
    import coverage
    import transformers

    tmodel = _transformers_build_model(args)

    input_args = (
        torch.ones(1, 16, dtype=torch.long),  # input_ids
        torch.ones(1, 16, dtype=torch.long),  # attention_mask
    )
    if args.inp_type == "dumb":
        input_args = generate_dummy_inputs(tmodel.model)

    # Note: Janky, but good enough for now.
    src_path = os.path.join(os.path.dirname(transformers.__file__), "models", tmodel.model_type)
    cov = coverage.Coverage(source=[src_path])
    cov.start()

    try:
        exported = torch.export.export(tmodel.model, input_args)
    except Exception as ex:
        print(f"{'!' * 70}\n"
              "PyTorch export failed. This is often due to incompatible python\n"
              "contructs implemented into the model code. You will likely need to\n"
              "adjust your model or find a different strategy to use.\n\n"
              f"Exception: {ex}\n"
              f"{'!' * 70}")
        # TODO: Add stacktrace when using --verbose
        return

    cov.stop()
    cov.save()

    # Failure Examples:
    # yannt transformers graph --model gpt2 --type AutoModelForCausalLM --as nodes
    # yannt transformers graph --model gptj --type AutoModelForCausalLM --as nodes

    # Working Examples:
    # yannt transformers graph --model bert --type AutoModelForMaskedLM --as coverage
    # yannt transformers graph --model bert --type AutoModelForMaskedLM --as human_ir
    # yannt transformers graph --model bert --type AutoModelForMaskedLM --as compiler_ir
    # yannt transformers graph --model bert --type AutoModelForMaskedLM --as nodes
    # yannt transformers graph --model bert --type AutoModelForMaskedLM --as graphviz | dot -Tsvg -o outputs/bert-drawer/my-model.svg
    # yannt transformers graph --model bert --type AutoModelForMaskedLM --as drawer --out outputs/bert-drawer/model.svg

    if args.graph_as == "human_ir":
        print(exported.graph_module.print_readable())

    if args.graph_as == "compiler_ir":
        print(exported.graph_module.graph.print_tabular())

    if args.graph_as == "schema":
        print(exported.graph_signature)

    if args.graph_as == "nodes":
        for node in exported.graph_module.graph.nodes:
            print(f"{'-' * 75}")
            print(f"{node.name} ({node.op})")
            print(f"  Inputs: {node.all_input_nodes}")
            print(f"  Target: {node.target}")
            print(f"  Args: {node.args} {node.kwargs}")

    if args.graph_as == "graphviz":
        print("digraph G {")
        for node in exported.graph_module.graph.nodes:
            print(f'  "{node.name}" [label="{node.op}\\n{node.target}"];')
            for inp in node.all_input_nodes:
                print(f'  "{inp.name}" -> "{node.name}";')
        print("}")

    if args.graph_as == "drawer":
        from torch.fx.passes.graph_drawer import FxGraphDrawer
        drawer = FxGraphDrawer(exported.graph_module, "exported_model")
        if args.output_path:
            dirname = os.path.dirname(args.output_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            drawer.get_dot_graph().write_svg(args.output_path)

    if args.graph_as == "coverage":
        data = cov.get_data()
        for file in data.measured_files():
            executed_lines = data.lines(file)         # set of executed line numbers
            # missing_lines = data.missing_lines(file)  # set of missed line numbers
            if len(executed_lines) > 0:
                print(f"File: {file}")
                print(f"  Executed lines: {collapse_ranges(sorted(executed_lines))}")

        cov.report()

    if hasattr(args, "breakpoint") and args.breakpoint:
        print(f"Locals: {list(locals().keys())}")
        breakpoint()
