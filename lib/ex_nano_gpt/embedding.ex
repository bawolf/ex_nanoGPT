defmodule ExNanoGPT.Embedding do
  @moduledoc """
  Token and position embeddings.

  Mirrors nanoGPT's GPT.__init__ embedding setup:
  - wte: token embedding (vocab_size, n_embd)
  - wpe: position embedding (block_size, n_embd)

  Forward pass: tok_emb + pos_emb + dropout
  """

  import Nx.Defn

  @typedoc """
  Embedding parameters.
  - `:wte` - token embedding table, shape `{vocab_size, n_embd}`
  - `:wpe` - position embedding table, shape `{block_size, n_embd}`
  """
  @type params :: %{wte: Nx.Tensor.t(), wpe: Nx.Tensor.t()}

  @typedoc "PRNG key for random operations, shape `{2}` (Nx.Random key)"
  @type prng_key :: Nx.Tensor.t()

  @doc """
  Initialize embedding parameters from Normal(0, 0.02).

  ## Returns
  A params map with `:wte` (shape `{vocab_size, n_embd}`) and
  `:wpe` (shape `{block_size, n_embd}`).
  """
  @spec init_params(pos_integer(), pos_integer(), pos_integer(), prng_key()) :: params()
  def init_params(vocab_size, block_size, n_embd, key) do
    keys = Nx.Random.split(key, parts: 2)
    wte_key = keys[0]
    wpe_key = keys[1]

    {wte, _} = Nx.Random.normal(wte_key, 0.0, 0.02, shape: {vocab_size, n_embd})
    {wpe, _} = Nx.Random.normal(wpe_key, 0.0, 0.02, shape: {block_size, n_embd})

    %{wte: wte, wpe: wpe}
  end

  @doc """
  Look up token embeddings for the given indices.

  ## Inputs
    * `idx` - integer tensor, shape `{batch, seq_len}`
    * `wte` - token embedding table, shape `{vocab_size, n_embd}`

  ## Returns
  Tensor of shape `{batch, seq_len, n_embd}`.
  """
  defn token_embedding(idx, wte) do
    Nx.take(wte, idx, axis: 0)
  end

  @doc """
  Look up position embeddings for positions 0..seq_len-1.

  ## Options
    * `:seq_len` - sequence length (required)

  ## Returns
  Tensor of shape `{seq_len, n_embd}`.
  """
  defn position_embedding(wpe, opts \\ []) do
    seq_len = opts[:seq_len]
    pos = Nx.iota({seq_len})
    Nx.take(wpe, pos, axis: 0)
  end

  @doc """
  Compute combined token + position embeddings with dropout.

  ## Inputs
    * `idx` - integer tensor, shape `{batch, seq_len}`
    * `params` - embedding params from `init_params/4`
    * `key` - PRNG key for dropout randomness

  ## Options
    * `:dropout_rate` - probability of zeroing an element (default: 0.0)
    * `:training` - whether to apply dropout (default: false)

  ## Returns
  Tensor of shape `{batch, seq_len, n_embd}`.
  """
  defn forward(idx, params, key, opts \\ []) do
    dropout_rate = opts[:dropout_rate]
    training = opts[:training]
    seq_len = Nx.axis_size(idx, 1)

    tok_emb = token_embedding(idx, params.wte)
    pos_emb = position_embedding(params.wpe, seq_len: seq_len)

    x = tok_emb + pos_emb

    if training do
      dropout(x, key, rate: dropout_rate)
    else
      x
    end
  end

  defnp dropout(x, key, opts \\ []) do
    rate = opts[:rate]

    if rate > 0.0 do
      {mask, _} = Nx.Random.uniform(key, shape: Nx.shape(x))
      keep = Nx.greater(mask, rate)
      x * keep / (1.0 - rate)
    else
      x
    end
  end
end
