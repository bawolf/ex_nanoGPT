defmodule ExNanoGPT.Attention do
  @moduledoc """
  Multi-head causal self-attention.

  Mirrors nanoGPT's CausalSelfAttention (model.py lines 29-76).

  The attention mechanism:
  1. Project input to Q, K, V via a single linear layer (c_attn)
  2. Split into multiple heads
  3. Compute scaled dot-product attention with causal mask
  4. Concatenate heads and project output (c_proj)
  """

  import Nx.Defn

  @typedoc """
  Attention parameters.
  - `:c_attn_weight` - combined Q/K/V projection, shape `{n_embd, 3 * n_embd}`
  - `:c_attn_bias` - combined Q/K/V bias, shape `{3 * n_embd}` (optional)
  - `:c_proj_weight` - output projection, shape `{n_embd, n_embd}`
  - `:c_proj_bias` - output projection bias, shape `{n_embd}` (optional)
  """
  @type params :: %{
          c_attn_weight: Nx.Tensor.t(),
          c_proj_weight: Nx.Tensor.t(),
          c_attn_bias: Nx.Tensor.t() | nil,
          c_proj_bias: Nx.Tensor.t() | nil
        }

  @doc """
  Initialize attention parameters.

  ## Options
    * `:bias` - whether to include bias terms (default: true)
    * `:n_layer` - number of layers, for scaling c_proj init (default: 1)
  """
  @spec init_params(pos_integer(), pos_integer(), Nx.Tensor.t(), keyword()) :: params()
  def init_params(n_embd, n_head, key, opts \\ []) do
    bias? = Keyword.get(opts, :bias, true)
    n_layer = Keyword.get(opts, :n_layer, 1)

    if rem(n_embd, n_head) != 0 do
      raise ArgumentError, "n_embd (#{n_embd}) must be divisible by n_head (#{n_head})"
    end

    keys = Nx.Random.split(key, parts: 2)

    {c_attn_weight, _} = Nx.Random.normal(keys[0], 0.0, 0.02, shape: {n_embd, 3 * n_embd})
    proj_std = 0.02 / :math.sqrt(2 * n_layer)
    {c_proj_weight, _} = Nx.Random.normal(keys[1], 0.0, proj_std, shape: {n_embd, n_embd})

    params = %{c_attn_weight: c_attn_weight, c_proj_weight: c_proj_weight}

    if bias? do
      params
      |> Map.put(:c_attn_bias, Nx.broadcast(0.0, {3 * n_embd}))
      |> Map.put(:c_proj_bias, Nx.broadcast(0.0, {n_embd}))
    else
      params
    end
  end

  @doc """
  Forward pass for multi-head causal self-attention.

  ## Inputs
    * `x` - input tensor, shape `{batch, seq_len, n_embd}`
    * `params` - attention params from `init_params/4`
    * `key` - PRNG key for dropout

  ## Options
    * `:n_head` - number of attention heads (required)
    * `:dropout_rate` - dropout probability (default: 0.0)
    * `:training` - whether in training mode (default: false)

  ## Returns
  Tensor of shape `{batch, seq_len, n_embd}`.
  """
  defn forward(x, params, key, opts \\ []) do
    n_head = opts[:n_head]
    dropout_rate = opts[:dropout_rate]
    training = opts[:training]

    {attn_drop_key, resid_drop_key} = split_key(key)

    dims = get_dims(x, n_head)

    qkv = linear(x, params.c_attn_weight, params[:c_attn_bias])

    {q, k, v} = split_qkv(qkv, dims)

    q = reshape_to_heads(q, dims)
    k = reshape_to_heads(k, dims)
    v = reshape_to_heads(v, dims)

    scale = compute_scale(dims)
    att = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) * scale

    att = apply_causal_mask(att, dims)

    att = softmax(att)

    {att, _} =
      if training do
        apply_dropout(att, attn_drop_key, dropout_rate)
      else
        {att, attn_drop_key}
      end

    y = Nx.dot(att, [3], [0, 1], v, [2], [0, 1])

    y = reshape_from_heads(y, dims)

    y = linear(y, params.c_proj_weight, params[:c_proj_bias])

    {y, _} =
      if training do
        apply_dropout(y, resid_drop_key, dropout_rate)
      else
        {y, resid_drop_key}
      end

    y
  end

  defnp split_key(key) do
    keys = Nx.Random.split(key)
    {keys[0], keys[1]}
  end

  # Extract dimensions as plain integers for use in shapes
  deftransform get_dims(x, n_head) do
    {batch, seq_len, n_embd} = Nx.shape(x)
    head_dim = div(n_embd, n_head)
    %{batch: batch, seq_len: seq_len, n_embd: n_embd, n_head: n_head, head_dim: head_dim}
  end

  # Split QKV tensor into separate Q, K, V
  deftransform split_qkv(qkv, dims) do
    n = dims.n_embd
    q = Nx.slice_along_axis(qkv, 0, n, axis: 2)
    k = Nx.slice_along_axis(qkv, n, n, axis: 2)
    v = Nx.slice_along_axis(qkv, 2 * n, n, axis: 2)
    {q, k, v}
  end

  # {batch, seq_len, n_embd} -> {batch, n_head, seq_len, head_dim}
  deftransform reshape_to_heads(x, dims) do
    x
    |> Nx.reshape({dims.batch, dims.seq_len, dims.n_head, dims.head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # {batch, n_head, seq_len, head_dim} -> {batch, seq_len, n_embd}
  deftransform reshape_from_heads(x, dims) do
    x
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({dims.batch, dims.seq_len, dims.n_embd})
  end

  deftransform compute_scale(dims) do
    Nx.tensor(1.0 / :math.sqrt(dims.head_dim), type: :f32)
  end

  deftransform apply_causal_mask(att, dims) do
    %{batch: batch, n_head: n_head, seq_len: seq_len} = dims
    rows = Nx.iota({batch, n_head, seq_len, seq_len}, axis: 2)
    cols = Nx.iota({batch, n_head, seq_len, seq_len}, axis: 3)
    mask = Nx.greater_equal(rows, cols)
    neg_inf = Nx.broadcast(Nx.Constants.neg_infinity(:f32), {batch, n_head, seq_len, seq_len})
    Nx.select(mask, att, neg_inf)
  end

  @doc """
  Build a lower-triangular causal mask of shape `{seq_len, seq_len}`.
  """
  @spec causal_mask(pos_integer()) :: Nx.Tensor.t()
  def causal_mask(seq_len) do
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    Nx.greater_equal(rows, cols)
  end

  # Numerically stable softmax over last axis
  defnp softmax(x) do
    max = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    exp = Nx.exp(x - max)
    exp / Nx.sum(exp, axes: [-1], keep_axes: true)
  end

  defnp linear(x, weight, bias) do
    result = Nx.dot(x, [-1], weight, [0])
    maybe_add_bias(result, bias)
  end

  deftransform maybe_add_bias(result, nil), do: result
  deftransform maybe_add_bias(result, bias), do: Nx.add(result, bias)

  defnp apply_dropout(x, key, rate) do
    if rate > 0.0 do
      {mask, new_key} = Nx.Random.uniform(key, shape: Nx.shape(x))
      keep = Nx.greater(mask, rate)
      {x * keep / (1.0 - rate), new_key}
    else
      {x, key}
    end
  end
end
