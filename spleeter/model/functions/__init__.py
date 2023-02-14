#!/usr/bin/env python
# coding: utf8

""" This package provide model functions. """

from typing import Any, Callable, Dict, Iterable, Optional

# pyright: reportMissingImports=false
# pylint: disable=import-error
import tensorflow as tf  # type: ignore

# pylint: enable=import-error

__email__ = "spleeter@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"


def apply(
    function: Callable[[tf.Tensor, str, Dict[str, Any]], Dict[str, tf.Tensor]],
    input_tensor: tf.Tensor,
    instruments: Iterable[str],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, tf.Tensor]:
    """
    Apply given function to the input tensor.

    Parameters:
        function:
            Function to be applied to tensor.
        input_tensor (tensorflow.Tensor):
            Tensor to apply blstm to.
        instruments (Iterable[str]):
            Iterable that provides a collection of instruments.
        params:
            (Optional) dict of BLSTM parameters.

    Returns:
        Created output tensor dict.
    """
    output_dict: Dict[str, tf.Tensor] = {}
    for instrument in instruments:
        out_name = f"{instrument}_spectrogram"
        output_dict[out_name] = function(input_tensor, out_name, params or {})
    return output_dict
