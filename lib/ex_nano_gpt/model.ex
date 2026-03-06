defmodule ExNanoGPT.Model do
  @moduledoc """
  Full GPT language model.

  Mirrors nanoGPT's GPT class (model.py lines 118-198).

  Architecture:
    1. Token embedding (wte) + position embedding (wpe) + dropout
    2. N transformer blocks
    3. Final layer norm (ln_f)
    4. Linear projection to vocab (lm_head) -- weight-tied with wte

  Weight tying: the lm_head projection reuses the token embedding weights.
  This reduces parameters and improves training (see "Using the Output Embedding
  to Improve Language Models", Press & Wolf 2016).
  """

  import Nx.Defn

  alias ExNanoGPT.{Block, Embedding, LayerNorm}

  @typedoc """
  Model configuration.
  - `:vocab_size` - number of tokens
  - `:block_size` - maximum sequence length
  - `:n_layer` - number of transformer blocks
  - `:n_head` - number of attention heads per block
  - `:n_embd` - embedding dimension
  - `:dropout` - dropout rate
  - `:bias` - whether to use bias in linear layers and layer norms
  """
  @type config :: %{
          vocab_size: pos_integer(),
          block_size: pos_integer(),
          n_layer: pos_integer(),
          n_head: pos_integer(),
          n_embd: pos_integer(),
          dropout: float(),
          bias: boolean()
        }

  @typedoc """
  Model parameters.
  - `:wte` - token embedding, shape `{vocab_size, n_embd}`
  - `:wpe` - position embedding, shape `{block_size, n_embd}`
  - `:blocks` - tuple of N transformer block params (tuple for defn compatibility)
  - `:ln_f` - final layer norm params
  (lm_head is weight-tied to wte, so not stored separately)
  """
  @type params :: %{
          wte: Nx.Tensor.t(),
          wpe: Nx.Tensor.t(),
          blocks: tuple(),
          ln_f: LayerNorm.params()
        }

  @doc """
  Initialize all model parameters.
  """
  @spec init_params(config(), Nx.Tensor.t()) :: params()
  def init_params(config, key) do
    %{
      vocab_size: vocab_size,
      block_size: block_size,
      n_layer: n_layer,
      n_head: n_head,
      n_embd: n_embd,
      bias: bias?
    } = config

    keys = Nx.Random.split(key, parts: n_layer + 1)

    emb_params = Embedding.init_params(vocab_size, block_size, n_embd, keys[0])

    blocks =
      for i <- 0..(n_layer - 1) do
        Block.init_params(n_embd, n_head, keys[i + 1], bias: bias?, n_layer: n_layer)
      end
      |> List.to_tuple()

    ln_f = LayerNorm.init_params(n_embd, bias: bias?)

    %{
      wte: emb_params.wte,
      wpe: emb_params.wpe,
      blocks: blocks,
      ln_f: ln_f
    }
  end

  @doc """
  Forward pass of the GPT model (inference mode).

  Returns logits for the last token only: shape `{batch, 1, vocab_size}`.
  For training (full logits + gradient support), use `forward_train/4`.
  """
  @spec forward(Nx.Tensor.t(), params(), config(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward(idx, params, config, key) do
    %{n_head: n_head, dropout: dropout} = config

    seq_len = Nx.axis_size(idx, 1)

    tok_emb = Embedding.token_embedding(idx, params.wte)
    pos_emb = Embedding.position_embedding(params.wpe, seq_len: seq_len)
    x = Nx.add(tok_emb, pos_emb)

    block_opts = [n_head: n_head, dropout_rate: dropout, training: false]

    x =
      params.blocks
      |> Tuple.to_list()
      |> Enum.reduce(x, fn block_params, acc ->
        Block.forward(acc, block_params, key, block_opts)
      end)

    x = LayerNorm.forward(x, params.ln_f)

    {_batch, seq, _n_embd} = Nx.shape(x)
    x = Nx.slice_along_axis(x, seq - 1, 1, axis: 1)
    project_to_vocab(x, params.wte)
  end

  @doc """
  Forward pass for training (defn, supports gradient tracing).

  Returns logits for ALL positions: shape `{batch, seq_len, vocab_size}`.
  This is a defn function so it can be used inside `Nx.Defn.value_and_grad`.

  ## Options
    * `:n_head` - number of attention heads (required)
    * `:dropout_rate` - dropout probability (required)
  """
  defn forward_train(idx, params, key, opts \\ []) do
    n_head = opts[:n_head]
    dropout_rate = opts[:dropout_rate]
    seq_len = Nx.axis_size(idx, 1)

    tok_emb = Embedding.token_embedding(idx, params.wte)
    pos_emb = Embedding.position_embedding(params.wpe, seq_len: seq_len)
    x = tok_emb + pos_emb

    {emb_drop_key, blocks_key} = split_key(key)
    x = maybe_dropout(x, emb_drop_key, dropout_rate)

    x = forward_blocks_train(x, params.blocks, blocks_key, n_head, dropout_rate)

    x = LayerNorm.forward(x, params.ln_f)

    Nx.dot(x, [-1], params.wte, [-1])
  end

  defnp maybe_dropout(x, key, rate) do
    if rate > 0.0 do
      {mask, _} = Nx.Random.uniform(key, shape: Nx.shape(x))
      keep = Nx.greater(mask, rate)
      x * keep / (1.0 - rate)
    else
      x
    end
  end

  deftransform forward_blocks_train(x, blocks, key, n_head, dropout_rate) do
    n_blocks = tuple_size(blocks)
    block_keys = Nx.Random.split(key, parts: n_blocks)

    blocks
    |> Tuple.to_list()
    |> Enum.with_index()
    |> Enum.reduce(x, fn {block_params, i}, acc ->
      Block.forward(acc, block_params, block_keys[i],
        n_head: n_head,
        dropout_rate: dropout_rate,
        training: true
      )
    end)
  end

  defnp split_key(key) do
    keys = Nx.Random.split(key)
    {keys[0], keys[1]}
  end

  defn project_to_vocab(x, wte) do
    Nx.dot(x, [-1], wte, [-1])
  end

  @doc """
  Compute cross-entropy loss between logits and targets.

  Targets with value -1 are ignored (matching PyTorch's `ignore_index=-1`).

  ## Inputs
    * `logits` - model output, shape `{batch, seq_len, vocab_size}`
    * `targets` - ground truth token IDs, shape `{batch, seq_len}`

  ## Returns
  Scalar loss value.
  """
  defn cross_entropy_loss(logits, targets) do
    {batch, seq_len, vocab_size} = Nx.shape(logits)
    flat_logits = Nx.reshape(logits, {batch * seq_len, vocab_size})
    flat_targets = Nx.reshape(targets, {batch * seq_len})

    log_probs = log_softmax(flat_logits)
    nll = gather_log_probs(log_probs, flat_targets)

    # Mask out positions where target == -1 (ignore_index)
    valid_mask = Nx.not_equal(flat_targets, -1)
    nll = nll * valid_mask
    n_valid = Nx.max(Nx.sum(valid_mask), 1)

    -Nx.sum(nll) / n_valid
  end

  defnp log_softmax(logits) do
    max = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = logits - max
    shifted - Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true))
  end

  deftransform gather_log_probs(log_probs, targets) do
    {n, vocab_size} = Nx.shape(log_probs)
    one_hot = Nx.equal(Nx.iota({n, vocab_size}, axis: 1), Nx.reshape(targets, {n, 1}))
    Nx.sum(Nx.multiply(log_probs, one_hot), axes: [-1])
  end

  @doc """
  Count the number of parameters in the model.

  ## Options
    * `:non_embedding` - if true, subtract position embedding params (default: true)
  """
  @spec count_params(params(), keyword()) :: non_neg_integer()
  def count_params(params, opts \\ []) do
    non_embedding = Keyword.get(opts, :non_embedding, true)

    total = count_tensor_params(params)

    if non_embedding do
      total - tensor_numel(params.wpe)
    else
      total
    end
  end

  defp count_tensor_params(%Nx.Tensor{} = t), do: tensor_numel(t)

  defp count_tensor_params(map) when is_map(map) do
    map |> Map.values() |> Enum.map(&count_tensor_params/1) |> Enum.sum()
  end

  defp count_tensor_params(tuple) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.map(&count_tensor_params/1) |> Enum.sum()
  end

  defp count_tensor_params(list) when is_list(list) do
    list |> Enum.map(&count_tensor_params/1) |> Enum.sum()
  end

  defp count_tensor_params(_), do: 0

  defp tensor_numel(%Nx.Tensor{} = t) do
    t |> Nx.shape() |> Tuple.to_list() |> Enum.product()
  end
end
