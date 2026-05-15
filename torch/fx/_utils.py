from __future__ import annotations

import sys
from typing import Any, Dict, Optional, cast

import torch
from torch._logging import LazyString


def lazy_format_graph_code(
    name: Optional[str, gm: torch.fx.GraphModule, maybe_id: int]= None, **kwargs: Any
) -> LazyString:  # pyrefly: ignore[implicit-any]
    """
    Returns a LazyString that formats the graph code.
    """

    def format_name() -> str:
        if maybe_id is not None:
            return f"{name} {maybe_id}"
        else:
            return name

    if "print_output" not in kwargs:
        kwargs["print_output"] = False

    if "colored" in kwargs:
        try:
            if not sys.stdout.isatty():
                kwargs["colored"] = False
        except AttributeError:
            kwargs["colored"] = False

    return LazyString(
        lambda: _format_graph_code(
            f"===== {format_name()} =====\n",
            gm.forward.__code__.co_filename,
            gm.print_readable(**kwargs),
        )
    )


def _format_graph_code(name: str, filename: str, graph_str: str) -> str:
    """
    Returns a string that formats the graph code.
    """
    return f"TRACED GRAPH\n {name} {filename} {graph_str}\n"


def first_call_function_nn_module_stack(graph: torch.fx.Graph) -> Optional[Dict[str, Any]]:
    """
    Returns the nn_module_stack of the first call_function node.
    """
    for node in graph.nodes:
        if node.op == "call_function" and "nn_module_stack" in node.meta:
            return node.meta["nn_module_stack"]
    return None


def get_node_context(node: torch.fx.Node, num_nodes: int = 2) -> str:
    """
    Returns a string of the last num_nodes nodes in the graph.
    """
    node_contexts = []
    cur = node
    for _ in range(num_nodes):
        # cast to str to handle None return value
        node_contexts.append(str(cur.format_node()))
        if cur.op == "root":
            break
        cur = cur.prev
    return "\n".join(node_contexts[::-1])
