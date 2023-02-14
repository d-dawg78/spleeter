#!/usr/bin/env python
# coding: utf8

"""
    Python oneliner script usage.

    USAGE: python -m spleeter {train,evaluate,separate} ...

    Notes:
        All critical import involving TF, numpy or Pandas are deported to
        command function scope to avoid heavy import on CLI evaluation,
        leading to large bootstraping time.
"""
import json
from functools import partial
from glob import glob
from itertools import product
from os.path import join
from typing import Any, Dict, List, Tuple

# pyright: reportMissingImports=false
# pylint: disable=import-error
from typer import Argument, Exit, Option, Typer
from typer.models import ArgumentInfo, OptionInfo

from . import SpleeterError
from .audio import Codec
from .options import *
from .utils.logging import configure_logger, logger

# pylint: enable=import-error

spleeter: Typer = Typer(add_completion=False, no_args_is_help=True, short_help="-h")
""" CLI application. """


@spleeter.callback()
def default(
    version: OptionInfo = VersionOption,
) -> None:
    pass


@spleeter.command(no_args_is_help=True)
def train(
    adapter: OptionInfo = AudioAdapterOption,
    data: OptionInfo = TrainingDataDirectoryOption,
    params_filename: OptionInfo = ModelParametersOption,
    verbose: OptionInfo = VerboseOption,
) -> None:
    """
    Train a source separation model
    """
    import tensorflow as tf  # type: ignore

    from .audio.adapter import AudioAdapter
    from .dataset import get_training_dataset, get_validation_dataset
    from .model import model_fn
    from .model.provider import ModelProvider
    from .utils.configuration import load_configuration

    configure_logger(bool(verbose))
    audio_adapter = AudioAdapter.get(str(adapter))
    audio_path = str(data)
    params = load_configuration(str(params_filename))
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params["model_dir"],
        params=params,
        config=tf.estimator.RunConfig(
            save_checkpoints_steps=params["save_checkpoints_steps"],
            tf_random_seed=params["random_seed"],
            save_summary_steps=params["save_summary_steps"],
            session_config=session_config,
            log_step_count_steps=10,
            keep_checkpoint_max=2,
        ),
    )
    input_fn = partial(get_training_dataset, params, audio_adapter, audio_path)
    train_spec = tf.estimator.TrainSpec(
        input_fn=input_fn, max_steps=params["train_max_steps"]
    )
    input_fn = partial(get_validation_dataset, params, audio_adapter, audio_path)
    evaluation_spec = tf.estimator.EvalSpec(
        input_fn=input_fn, steps=None, throttle_secs=params["throttle_secs"]
    )
    logger.info("Start model training")
    tf.estimator.train_and_evaluate(estimator, train_spec, evaluation_spec)
    ModelProvider.writeProbe(params["model_dir"])
    logger.info("Model training done")


@spleeter.command(no_args_is_help=True)
def separate(
    deprecated_files: OptionInfo = AudioInputOption,
    files: ArgumentInfo = AudioInputArgument,
    adapter: OptionInfo = AudioAdapterOption,
    bitrate: OptionInfo = AudioBitrateOption,
    codec: OptionInfo = AudioCodecOption,
    duration: OptionInfo = AudioDurationOption,
    offset: OptionInfo = AudioOffsetOption,
    output_path: OptionInfo = AudioOutputOption,
    filename_format: OptionInfo = FilenameFormatOption,
    params_filename: OptionInfo = ModelParametersOption,
    mwf: OptionInfo = MWFOption,
    verbose: OptionInfo = VerboseOption,
) -> None:
    """
    Separate audio file(s)
    """
    from .audio.adapter import AudioAdapter
    from .separator import Separator

    configure_logger(bool(verbose))
    if deprecated_files is not None:
        logger.error(
            "⚠️ -i option is not supported anymore, audio files must be supplied "
            "using input argument instead (see spleeter separate --help)"
        )
        raise Exit(20)
    audio_adapter: AudioAdapter = AudioAdapter.get(adapter)
    separator: Separator = Separator(params_filename, MWF=mwf)

    for filename in files:
        separator.separate_to_file(
            str(filename),
            str(output_path),
            audio_adapter=audio_adapter,
            offset=offset,
            duration=duration,
            codec=codec,
            bitrate=bitrate,
            filename_format=filename_format,
            synchronous=False,
        )
    separator.join()


EVALUATION_SPLIT: str = "test"
EVALUATION_METRICS_DIRECTORY: str = "metrics"
EVALUATION_INSTRUMENTS: Tuple[str, ...] = ("vocals", "drums", "bass", "other")
EVALUATION_METRICS: Tuple[str, ...] = ("SDR", "SAR", "SIR", "ISR")
EVALUATION_MIXTURE: str = "mixture.wav"
EVALUATION_AUDIO_DIRECTORY: str = "audio"


def _compile_metrics(metrics_output_directory: str) -> Dict[Any, Dict[Any, List[Any]]]:
    """
    Compiles metrics from given directory and returns results as dict.

    Parameters:
        metrics_output_directory (str):
            Directory to get metrics from.

    Returns:
        Dict:
            Compiled metrics as dict.
    """
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    songs = glob(join(metrics_output_directory, "test/*.json"))
    index = pd.MultiIndex.from_tuples(
        product(EVALUATION_INSTRUMENTS, EVALUATION_METRICS),
        names=["instrument", "metric"],
    )
    pd.DataFrame([], index=["config1", "config2"], columns=index)
    metrics: Dict[Any, Dict[Any, List[Any]]] = {
        instrument: {k: [] for k in EVALUATION_METRICS}
        for instrument in EVALUATION_INSTRUMENTS
    }
    for song in songs:
        with open(song, "r") as stream:
            data = json.load(stream)
        for target in data["targets"]:
            instrument = target["name"]
            for metric in EVALUATION_METRICS:
                sdr_med = np.median(
                    [
                        frame["metrics"][metric]
                        for frame in target["frames"]
                        if not np.isnan(frame["metrics"][metric])
                    ]
                )
                metrics[instrument][metric].append(sdr_med)
    return metrics


@spleeter.command(no_args_is_help=True)
def evaluate(
    adapter: OptionInfo = AudioAdapterOption,
    output_path: OptionInfo = AudioOutputOption,
    params_filename: OptionInfo = ModelParametersOption,
    mus_dir: OptionInfo = MUSDBDirectoryOption,
    mwf: OptionInfo = MWFOption,
    verbose: OptionInfo = VerboseOption,
) -> Dict[Any, Dict[Any, List[Any]]]:
    """
    Evaluate a model on the musDB test dataset
    """
    import numpy as np

    configure_logger(bool(verbose))
    try:
        import musdb  # type: ignore
        import museval  # type: ignore
    except ImportError:
        logger.error("Extra dependencies musdb and museval not found")
        logger.error("Please install musdb and museval first, abort")
        raise Exit(10)
    # Separate musdb sources.
    songs = glob(join(str(mus_dir), EVALUATION_SPLIT, "*/"))
    mixtures = [join(song, EVALUATION_MIXTURE) for song in songs]
    audio_output_directory = join(str(output_path), EVALUATION_AUDIO_DIRECTORY)
    separate(
        deprecated_files=Option(None),
        files=Argument(mixtures),
        adapter=adapter,
        bitrate=Option("128k"),
        codec=Option(Codec.WAV),
        duration=Option(600.0),
        offset=Option(0),
        output_path=Option(join(audio_output_directory, EVALUATION_SPLIT)),
        filename_format=Option("{foldername}/{instrument}.{codec}"),
        params_filename=params_filename,
        mwf=mwf,
        verbose=verbose,
    )
    # Compute metrics with musdb.
    metrics_output_directory = join(str(output_path), EVALUATION_METRICS_DIRECTORY)
    logger.info("Starting musdb evaluation (this could be long) ...")
    dataset = musdb.DB(root=mus_dir, is_wav=True, subsets=[EVALUATION_SPLIT])
    museval.eval_mus_dir(
        dataset=dataset,
        estimates_dir=audio_output_directory,
        output_dir=metrics_output_directory,
    )
    logger.info("musdb evaluation done")
    # Compute and pretty print median metrics.
    metrics = _compile_metrics(metrics_output_directory)
    for instrument, metric in metrics.items():
        logger.info(f"{instrument}:")
        for metric, value in metric.items():
            logger.info(f"{metric}: {np.median(value):.3f}")
    return metrics


def entrypoint() -> None:
    """Application entrypoint."""
    try:
        spleeter()
    except SpleeterError as e:
        logger.error(e)


if __name__ == "__main__":
    entrypoint()
