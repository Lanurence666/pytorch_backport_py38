"""
PyTorch 2.13 Distributed Support Verification Script

This script verifies that the backported PyTorch 2.13 has full distributed
training support and is compatible with transformers and huggingface_hub.

Usage:
    python verify_distributed.py

Requirements:
    pip install torch transformers huggingface_hub
"""

import sys
import os
import traceback

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)
parent_dir = os.path.dirname(script_dir)
if parent_dir in sys.path:
    sys.path.remove(parent_dir)

PASS = 0
FAIL = 0
SKIP = 0


def test(name, func):
    global PASS, FAIL, SKIP
    try:
        result = func()
        if result is None or result is True:
            PASS += 1
            print(f"  [PASS] {name}")
        elif result is False:
            FAIL += 1
            print(f"  [FAIL] {name}")
        elif result == "skip":
            SKIP += 1
            print(f"  [SKIP] {name}")
    except Exception as e:
        FAIL += 1
        print(f"  [FAIL] {name}: {e}")


def main():
    global PASS, FAIL, SKIP

    print("=" * 70)
    print("  PyTorch 2.13 Backport - Distributed Support Verification")
    print("=" * 70)

    print("\n[1/7] Version Information")
    import torch
    import torch.distributed as dist
    print(f"  PyTorch:      {torch.__version__}")
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  CUDA:         {torch.version.cuda or 'N/A'}")
    print(f"  CUDA avail:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")

    print("\n[2/7] Distributed Backend Availability")
    test("torch._C._has_distributed()", lambda: torch._C._has_distributed())
    test("dist.is_available()", lambda: dist.is_available())
    test("dist.is_gloo_available()", lambda: dist.is_gloo_available())
    test("dist.is_nccl_available()", lambda: dist.is_nccl_available() or "skip")

    print("\n[3/7] Distributed Module Imports")
    def import_check(module, attrs=None):
        m = __import__(module, fromlist=attrs or ["__name__"])
        if attrs:
            for a in attrs:
                getattr(m, a)
        return True

    test("ProcessGroupGloo", lambda: import_check("torch.distributed", ["ProcessGroupGloo"]))
    test("TCPStore", lambda: import_check("torch.distributed", ["TCPStore"]))
    test("DistributedDataParallel", lambda: import_check("torch.nn.parallel", ["DistributedDataParallel"]))
    test("DistributedSampler", lambda: import_check("torch.utils.data.distributed", ["DistributedSampler"]))
    test("torch.distributed.autograd", lambda: import_check("torch.distributed.autograd"))
    test("torch.distributed.rpc", lambda: import_check("torch.distributed.rpc"))
    test("DTensor", lambda: import_check("torch.distributed.tensor", ["DTensor"]))
    test("Shard / Replicate / Partial", lambda: import_check("torch.distributed.tensor", ["Shard", "Replicate", "Partial"]))
    test("ColwiseParallel / RowwiseParallel", lambda: import_check("torch.distributed.tensor.parallel", ["ColwiseParallel", "RowwiseParallel"]))
    test("checkpoint save / load", lambda: import_check("torch.distributed.checkpoint", ["save", "load"]))
    test("fully_shard", lambda: import_check("torch.distributed._composable.fsdp", ["fully_shard"]))
    test("replicate", lambda: import_check("torch.distributed._composable.replicate", ["replicate"]))
    test("PipelineStage", lambda: import_check("torch.distributed.pipeline.sync", ["PipelineStage"]) if sys.platform != "win32" else "skip")

    print("\n[4/7] ProcessGroup Initialization (Gloo)")
    def test_init_process_group():
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(29500 + os.getpid() % 1000)
        port = 29500 + os.getpid() % 1000 + 1
        store = dist.TCPStore("127.0.0.1", port, is_master=True, wait_for_workers=False)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
        return True

    test("init_process_group(gloo)", test_init_process_group)

    print("\n[5/7] DDP Functional Test")
    def test_ddp():
        if not torch.cuda.is_available():
            return "skip"
        from torch.nn.parallel import DistributedDataParallel as DDP
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        ).cuda()
        ddp_model = DDP(model)
        x = torch.randn(4, 64, device="cuda")
        with torch.no_grad():
            y = ddp_model(x)
        assert y.shape == (4, 10), f"Expected (4,10), got {y.shape}"
        return True

    test("DDP forward pass", test_ddp)

    def test_ddp_amp():
        if not torch.cuda.is_available():
            return "skip"
        from torch.nn.parallel import DistributedDataParallel as DDP
        import torch.nn as nn
        model = nn.Linear(64, 10).cuda()
        ddp_model = DDP(model)
        x = torch.randn(4, 64, device="cuda")
        with torch.amp.autocast("cuda"):
            y = ddp_model(x)
        return True

    test("DDP + AMP mixed precision", test_ddp_amp)

    print("\n[6/7] Transformers Compatibility")
    def test_transformers_import():
        import transformers
        print(f"  transformers version: {transformers.__version__}")
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        return True

    test("transformers import", test_transformers_import)

    def test_transformers_model():
        if not torch.cuda.is_available():
            return "skip"
        from transformers import BertConfig, BertModel
        config = BertConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2, num_attention_heads=2)
        model = BertModel(config).cuda()
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda")
        with torch.no_grad():
            out = model(input_ids=input_ids)
        assert out.last_hidden_state.shape == (1, 16, 64)
        return True

    test("transformers BertModel forward", test_transformers_model)

    def test_transformers_ddp():
        if not torch.cuda.is_available():
            return "skip"
        from transformers import BertConfig, BertModel
        from torch.nn.parallel import DistributedDataParallel as DDP
        config = BertConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2, num_attention_heads=2)
        model = BertModel(config).cuda()
        ddp_model = DDP(model)
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda")
        with torch.no_grad():
            out = ddp_model(input_ids=input_ids)
        return True

    test("transformers + DDP", test_transformers_ddp)

    print("\n[7/7] huggingface_hub Compatibility")
    def test_huggingface_hub_import():
        import huggingface_hub
        print(f"  huggingface_hub version: {huggingface_hub.__version__}")
        from huggingface_hub import HfApi, snapshot_download, PyTorchModelHubMixin
        return True

    test("huggingface_hub import", test_huggingface_hub_import)

    def test_huggingface_hub_api():
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            info = api.model_info("bert-base-uncased")
            assert "bert-base-uncased" in info.id
            return True
        except Exception as e:
            if "connect" in str(e).lower() or "timeout" in str(e).lower() or "network" in str(e).lower():
                return "skip"
            raise

    test("huggingface_hub API (online)", test_huggingface_hub_api)

    def test_hub_mixin():
        import torch.nn as nn
        from huggingface_hub import PyTorchModelHubMixin

        class MyModel(nn.Module, PyTorchModelHubMixin):
            def __init__(self, hidden_size=64):
                super().__init__()
                self.linear = nn.Linear(hidden_size, hidden_size)

        model = MyModel(hidden_size=64)
        assert hasattr(model, "push_to_hub")
        assert hasattr(model, "save_pretrained")
        return True

    test("PyTorchModelHubMixin", test_hub_mixin)

    try:
        dist.destroy_process_group()
    except Exception:
        pass

    print("\n" + "=" * 70)
    total = PASS + FAIL + SKIP
    print(f"  Results: {PASS} passed, {FAIL} failed, {SKIP} skipped (total: {total})")
    if FAIL == 0:
        print("  ALL TESTS PASSED!")
    else:
        print(f"  {FAIL} test(s) FAILED")
    print("=" * 70)
    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
