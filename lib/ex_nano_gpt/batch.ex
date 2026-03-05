defmodule ExNanoGPT.Batch do
  @moduledoc """
  Batch generation for training.

  Mirrors nanoGPT's get_batch() function:
  - Memory-maps the binary data
  - Picks random starting indices
  - Returns {x, y} where y = x shifted right by 1
  """

  import Nx.Defn

  @doc """
  Generate a random batch of training data.

  Given a 1D tensor of token IDs, picks `batch_size` random windows
  of length `block_size` and returns {x, y} where:
  - x has shape {batch_size, block_size}
  - y has shape {batch_size, block_size} (x shifted right by 1)

  The `key` is an Nx.Random PRNG key for reproducible randomness.
  """
  defn get_batch(data, key, opts \\ []) do
    batch_size = opts[:batch_size]
    block_size = opts[:block_size]

    data_len = Nx.axis_size(data, 0)
    max_start = data_len - block_size - 1

    {starts, _new_key} = Nx.Random.randint(key, 0, max_start, shape: {batch_size})

    {x, y} = build_batch(data, starts, batch_size, block_size)
    {x, y}
  end

  deftransform build_batch(data, starts, batch_size, block_size) do
    x_rows =
      for i <- 0..(batch_size - 1) do
        start = starts[i]
        Nx.slice(data, [start], [block_size])
      end

    y_rows =
      for i <- 0..(batch_size - 1) do
        start = Nx.add(starts[i], 1)
        Nx.slice(data, [start], [block_size])
      end

    x = Nx.stack(x_rows)
    y = Nx.stack(y_rows)
    {x, y}
  end
end
