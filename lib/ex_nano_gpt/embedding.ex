defmodule ExNanoGPT.Embedding do
  @moduledoc """
  Token and position embeddings.

  Mirrors nanoGPT's GPT.__init__ embedding setup:
  - wte: token embedding (vocab_size, n_embd)
  - wpe: position embedding (block_size, n_embd)

  Forward pass: tok_emb + pos_emb + dropout
  """

  import Nx.Defn

  @doc """
  Initialize embedding parameters.

  Returns a map with:
  - :wte - token embedding table, shape {vocab_size, n_embd}
  - :wpe - position embedding table, shape {block_size, n_embd}

  Both initialized from Normal(0, 0.02) matching nanoGPT's _init_weights.
  """
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

  idx: integer tensor of shape {batch, seq_len}
  Returns tensor of shape {batch, seq_len, n_embd}
  """
  defn token_embedding(idx, wte) do
    Nx.take(wte, idx, axis: 0)
  end

  @doc """
  Look up position embeddings for positions 0..seq_len-1.

  Returns tensor of shape {seq_len, n_embd}
  """
  defn position_embedding(wpe, opts \\ []) do
    seq_len = opts[:seq_len]
    pos = Nx.iota({seq_len})
    Nx.take(wpe, pos, axis: 0)
  end

  @doc """
  Compute combined token + position embeddings with dropout.

  idx: integer tensor of shape {batch, seq_len}
  params: map with :wte and :wpe keys
  key: PRNG key for dropout
  opts: [dropout_rate: float, training: boolean]
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
