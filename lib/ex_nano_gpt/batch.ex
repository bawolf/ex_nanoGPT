defmodule ExNanoGPT.Batch do
  @moduledoc """
  Batch generation for training.

  Mirrors nanoGPT's get_batch() function:
  - Picks random starting indices from the token data
  - Returns {x, y} where y = x shifted right by 1
  """

  import Nx.Defn

  @doc """
  Generate a random batch of training data.

  Given a 1D tensor of token IDs, picks `batch_size` random windows
  of length `block_size`.

  ## Inputs
    * `data` - 1D tensor of token IDs, shape `{n_tokens}`
    * `key` - PRNG key for reproducible random sampling

  ## Options
    * `:batch_size` - number of sequences per batch (required)
    * `:block_size` - length of each sequence (required)

  ## Returns
  `{x, y}` where both have shape `{batch_size, block_size}`.
  `y` is `x` shifted right by one position (the prediction target).
  """
  defn get_batch(data, key, opts \\ []) do
    batch_size = opts[:batch_size]
    block_size = opts[:block_size]

    data_len = Nx.axis_size(data, 0)
    # torch.randint(len(data) - block_size, ...) gives [0, len-block_size)
    # Nx.Random.randint upper bound is exclusive, same as torch.randint
    max_start = data_len - block_size

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
