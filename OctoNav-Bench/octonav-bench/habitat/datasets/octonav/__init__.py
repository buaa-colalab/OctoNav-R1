#!/usr/bin/env python3

from habitat.core.dataset import Dataset
from habitat.core.registry import registry


def _try_register_octonavdatasetv1():
    try:
        from habitat.datasets.octonav.octonav_dataset import (  # noqa: F401
            OctoNavDatasetV1,
        )

    except ImportError as e:
        octo_import_error = e

        @registry.register_dataset(name="OctoNav-v1")
        class OctoNavDatasetImportError(Dataset):
            def __init__(self, *args, **kwargs):
                raise octo_import_error
