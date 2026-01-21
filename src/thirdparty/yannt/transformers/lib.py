import sys

import warnings
#warnings.filterwarnings("ignore", message="Config not found for parakeet.*")
warnings.filterwarnings("ignore", module="parakeet")

print("Loading PyTorch and Transformers.", file=sys.stderr)
try:
    import torch
except:
    print(
        "If you don't have torch installed and don't want to download\n"
        "a bazillion byte version with CUDA and other binaries, install\n"
        "the CPU on version with something like the following:\n\n"
        "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
    )
    raise

import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer

import logging
logging.getLogger("parakeet").setLevel(logging.ERROR)

class TransformersModel:
    def __init__(self, model=None, class_name=None, model_type=None):
        self.model = model
        self.automodel_class_name = class_name
        self.model_type = model_type

    def save_torch_safetensors(self, model_fpath, max_shard=2147483648):
        try:
            max_shard
            self.model.save_pretrained(
                f"{model_fpath}", safe_serialization=True, max_shard_size=max_shard
            )
        except Exception as e:
            print(f"Torch safetensors save error: {e}")

    def save_torch_weights(self, model_fpath, with_arch=True):
        try:
            params = {"model_state_dict": self.model.state_dict()}
            if with_arch:
                params["model_architecture"] = self.model
            torch.save(params, f"{model_fpath}.pth")
        except Exception as e:
            print(f"Torch weights save error: {e}")

    def save_torch_state(self, model_fpath, model_name):
        try:
            torch.save(self.model.state_dict(), f"{model_fpath}/{model_name}.pth")
        except Exception as e:
            print(f"Torch state save error: {e}")

    def save_torch_script(self, model_fpath):
        try:
            torch.jit.script(self.model).save(f"{model_fpath}.torchscript.pt")
        except Exception as e:
            print(f"Torch script save error: {e}")

    def save_torch_onnx(self, model_fpath, model_name):
        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") #self.model_type) #, use_fast=True) # local_files_only=True,
            # tokenizer = AutoTokenizer.from_pretrained("./my_tokenizer", local_files_only=True)
            inputs = tokenizer("Hello, this is a test", return_tensors="pt")
            self.model.config.use_cache = False
            self.model.config.torchscript = True # ??!??
            torch.onnx.export(
                self.model,
                (inputs["input_ids"], inputs["attention_mask"]), # input_ids,
                f"{model_fpath}/{model_name}.onnx",
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
            '''
            torch.onnx.export(
                model,
                (inputs["input_ids"], inputs["attention_mask"]),
                "gpt2_default_random.onnx",
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "logits": {0: "batch", 1: "seq"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
            '''
        except Exception as e:
            print(f"Torch script save error: {e}")

    def save_traced(self, model_fpath):
        try:
            example_input_ids = torch.randint(0, 50257, (1, 8))
            example_attention_mask = torch.ones(1, 8, dtype=torch.long)
            traced = torch.jit.trace(
                self.model, (example_input_ids, example_attention_mask)
            )
            traced.save(f"{model_fpath}.traced.pt")
        except Exception as e:
            print(f"Torch traced save error: {e}")

    # yannt transformers create --model bert --type AutoModelForMaskedLM --export_path outputs/bert-export
    def save_export(self, model_fpath, model_name):
        #input_ids = (torch.ones(1, 16, dtype=torch.long))
        input_ids = (
            torch.ones(1, 16, dtype=torch.long),  # input_ids
            torch.ones(1, 16, dtype=torch.long),  # attention_mask
        )
        exported = torch.export.export(self.model, input_ids)

        # print(exported.graph_module.print_readable())

        # gm = exported.graph_module

        # for node in gm.graph.nodes:
        #     print(f"{'-' * 75}")
        #     print(f"{node.name} ({node.op})")
        #     print(f"  Inputs: {node.all_input_nodes}")
        #     print(f"  Target: {node.target}")
        #     print(f"  Args: {node.args} {node.kwargs}")
        #     # print(
        #     #     node.op,
        #     #     node.target,
        #     #     node.name,
        #     #     node.args,
        #     #     node.kwargs,
        #     # )


        '''
        node.op:
        - placeholder - model input
        - call_function - functional op
        - call_module - submodule call
        - call_method - tensor method
        - output - graph output
        '''

        # nodes = []
        # edges = []

        # for node in gm.graph.nodes:
        #     nodes.append(node.name)
        #     for input_node in node.all_input_nodes:
        #         edges.append((input_node.name, node.name))

        #breakpoint()

class TransformersModelFactory:
    def __init__(self):
        self.auto_model_combos = {}
        self.get_auto_model_combos()

    def get_auto_model_combos(self):
        for automodel_class_name in dir(transformers):
            if not automodel_class_name.startswith("AutoModel"):
                continue
            automodel_class = getattr(transformers, automodel_class_name)

            for automodel_cfg in automodel_class._model_mapping:
                try:
                    type_name = automodel_cfg.model_type
                    cfg_name = automodel_cfg.__name__
                    # ! BUG: Not working in py3.9
                    arch_name = automodel_class._model_mapping[automodel_cfg].__name__
                except AttributeError:
                    continue
                except:
                    breakpoint()

                if automodel_class_name not in self.auto_model_combos:
                    self.auto_model_combos[automodel_class_name] = {}
                if type_name not in self.auto_model_combos[automodel_class_name]:
                    self.auto_model_combos[automodel_class_name][type_name] = {}
                self.auto_model_combos[automodel_class_name][type_name]["config"] = (
                    cfg_name
                )
                self.auto_model_combos[automodel_class_name][type_name]["arch"] = (
                    arch_name
                )

    def reconstruct_model(self, automodel_class_name, model_type):
        tmodel = None
        automodel_class = getattr(transformers, automodel_class_name)
        try:
            # CONSIDER: config = AutoConfig.from_pretrained(model_name)
            # The above will allow predefined config without weights. (In theory.)
            cfg = transformers.AutoConfig.for_model(model_type=model_type)
            model = automodel_class.from_config(cfg)
            model.eval()  # We don't train here.

            # Throw in some random values.
            for param in model.parameters():
                param.data = torch.randn_like(param.data)

            tmodel = TransformersModel(model, automodel_class_name, model_type)

        except Exception as e:
            print(f"Build error: {e}")

        return tmodel
