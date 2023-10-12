import array
import asyncio
import concurrent.futures
import dataclasses
import json
import logging
from operator import itemgetter  # pylint: disable=g-importing-member
import sys
import time
from typing import List

import numpy as np

sys.path.append('/saxml/bazel-bin/saxml/client/python/')
import sax



class ThreadedLMClient:
  """Holds a thread pool and a sax client for LM inference."""

  _thread_pool: concurrent.futures.ThreadPoolExecutor
  _sax_model: sax.Model
  _sax_language_model: sax.LanguageModel
  _futures = List[concurrent.futures.Future]

  def __init__(
      self,
      model_path: str,
      num_threads: int = 200,
  ):
    self._thread_pool = concurrent.futures.ThreadPoolExecutor(num_threads)
    self._sax_model = sax.Model(model_path)
    self._sax_language_model = self._sax_model.LM()
    self._futures = []
    self._loop = asyncio.get_running_loop()

  def _process_query(self, input, option=None):
    """Processes a single sample."""
    response = self._sax_language_model.Generate(input, option)
    return response

  def process_single_query_async(self, input, option=None):
    """Executes a single query and marks responses complete asynchronously.

    Args:
      query_sample: Single prompt
      warmup: Indicates that this is a warmup request.
    """
    future = self._thread_pool.submit(
        self._process_query, input, option
    )
    self._futures.append(future)

  async def process_single_sample(self, input, option=None):
    result = await self._loop.run_in_executor(
            self._thread_pool, self._process_query, input, option)
    return result

  def flush(self):
    concurrent.futures.wait(self._futures)
    self._futures = []
