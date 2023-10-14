import array
import asyncio
import concurrent.futures
import dataclasses
import json
import logging
from operator import itemgetter  # pylint: disable=g-importing-member
import time
from typing import List

import numpy as np
from saxml.client.python import sax



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

  def _process_query(self, input, extra_input=None):
    """Processes a single sample."""
    option = None
    if extra_input:
      option = sax.ModelOptions()
      for k, v in extra_input.items():
        option.SetExtraInput(k, v)
    response = self._sax_language_model.Generate(input, option)
    return response

  def process_single_query_async(self, input, extra_input=None):
    """Executes a single query and marks responses complete asynchronously.

    Args:
      query_sample: Single prompt
      warmup: Indicates that this is a warmup request.
    """
    future = self._thread_pool.submit(
      self._process_query, input, extra_input
    )
    self._futures.append(future)

  async def process_single_query(self, input, extra_input=None):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
      self._thread_pool, self._process_query, input, extra_input
    )
    return result

  def flush(self):
    concurrent.futures.wait(self._futures)
    self._futures = []
