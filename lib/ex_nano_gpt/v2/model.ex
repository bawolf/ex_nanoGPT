defmodule ExNanoGPT.V2.Model do
  @moduledoc """
  Modern GPT model mirroring nanochat/gpt.py.

  Key differences from v1 (nanoGPT):
  - RoPE instead of learned position embeddings
  - RMSNorm instead of LayerNorm (no learnable params)
  - GQA (grouped-query attention) with separate Q/K/V projections
  - QK normalization before attention
  - Sliding window attention (per-layer window sizes)
  - ReLU² activation instead of GELU
  - Untied embedding/lm_head weights
  - No bias in any linear layer
  - Per-layer residual scalars (resid_lambda, x0_lambda)
  - Value embeddings on alternating layers
  - Logit softcapping
  """

  import Nx.Defn

  @softcap 20.0
  @rope_base 10_000
  @ve_gate_channels 32

  defstruct [
    :sequence_len, :vocab_size, :n_layer, :n_head,
    :n_kv_head, :n_embd, :window_pattern
  ]

  def tiny_config do
    %__MODULE__{
      sequence_len: 256, vocab_size: 256, n_layer: 2,
      n_head: 4, n_kv_head: 2, n_embd: 64, window_pattern: "SL"
    }
  end

  def base_config do
    %__MODULE__{
      sequence_len: 2048, vocab_size: 32768, n_layer: 12,
      n_head: 6, n_kv_head: 6, n_embd: 768, window_pattern: "SSSL"
    }
  end

  # ---------------------------------------------------------------------------
  # RMSNorm (no learnable params)
  # ---------------------------------------------------------------------------

  defn rms_norm(x) do
    variance = Nx.mean(Nx.multiply(x, x), axes: [-1], keep_axes: true)
    Nx.divide(x, Nx.sqrt(Nx.add(variance, 1.0e-5)))
  end

  # ---------------------------------------------------------------------------
  # Rotary Position Embeddings
  # ---------------------------------------------------------------------------

  def precompute_rope(seq_len, head_dim) do
    half = div(head_dim, 2)
    channel_range = Nx.multiply(Nx.iota({half}, type: :f32), 2)
    exponents = Nx.divide(channel_range, head_dim)
    inv_freq = Nx.exp(Nx.multiply(-:math.log(@rope_base), exponents))
    positions = Nx.iota({seq_len}, type: :f32)
    freqs = Nx.multiply(Nx.new_axis(positions, 1), Nx.new_axis(inv_freq, 0))
    cos = Nx.cos(freqs) |> Nx.reshape({1, seq_len, 1, half})
    sin = Nx.sin(freqs) |> Nx.reshape({1, seq_len, 1, half})
    {cos, sin}
  end

  defn apply_rope(x, cos, sin) do
    half = div(Nx.axis_size(x, 3), 2)
    x1 = Nx.slice_along_axis(x, 0, half, axis: 3)
    x2 = Nx.slice_along_axis(x, half, half, axis: 3)
    y1 = Nx.add(Nx.multiply(x1, cos), Nx.multiply(x2, sin))
    y2 = Nx.add(Nx.multiply(x1, Nx.negate(sin)), Nx.multiply(x2, cos))
    Nx.concatenate([y1, y2], axis: 3)
  end

  # ---------------------------------------------------------------------------
  # Init params
  # ---------------------------------------------------------------------------

  def init_params(%__MODULE__{} = config, key) do
    %{vocab_size: v, n_embd: d, n_layer: n, n_head: nh, n_kv_head: nkv} = config
    head_dim = div(d, nh)
    kv_dim = nkv * head_dim
    s = :math.sqrt(3) * :math.pow(d, -0.5)

    {wte, key} = Nx.Random.normal(key, 0.0, 1.0, shape: {v, d})
    {lm_head, key} = Nx.Random.normal(key, 0.0, 0.001, shape: {v, d})

    {blocks, key} =
      Enum.map_reduce(0..(n - 1), key, fn _i, k ->
        {c_q, k} = Nx.Random.uniform(k, -s, s, shape: {d, nh * head_dim})
        {c_k, k} = Nx.Random.uniform(k, -s, s, shape: {d, kv_dim})
        {c_v, k} = Nx.Random.uniform(k, -s, s, shape: {d, kv_dim})
        {c_fc, k} = Nx.Random.uniform(k, -s, s, shape: {d, 4 * d})
        block = %{
          c_q: c_q, c_k: c_k, c_v: c_v,
          c_proj: Nx.broadcast(Nx.tensor(0.0, type: :f32), {nh * head_dim, d}),
          ve_gate: Nx.broadcast(Nx.tensor(0.0, type: :f32), {@ve_gate_channels, nkv}),
          c_fc: c_fc,
          c_proj_mlp: Nx.broadcast(Nx.tensor(0.0, type: :f32), {4 * d, d})
        }
        {block, k}
      end)

    {value_embeds, _key} =
      Enum.map_reduce(0..(n - 1), key, fn i, k ->
        if has_ve?(i, n) do
          {ve, k} = Nx.Random.uniform(k, -s, s, shape: {v, kv_dim})
          {ve, k}
        else
          {:none, k}
        end
      end)

    %{
      wte: wte,
      lm_head: lm_head,
      resid_lambdas: Nx.broadcast(Nx.tensor(1.0, type: :f32), {n}),
      x0_lambdas: Nx.broadcast(Nx.tensor(0.1, type: :f32), {n}),
      blocks: List.to_tuple(blocks),
      value_embeds: List.to_tuple(value_embeds)
    }
  end

  def has_ve?(layer_idx, n_layer), do: rem(layer_idx, 2) == rem(n_layer - 1, 2)

  def compute_window_sizes(%__MODULE__{} = config) do
    pattern = config.window_pattern |> String.upcase() |> String.graphemes()
    long = config.sequence_len
    short = div(long, 2)

    sizes =
      for i <- 0..(config.n_layer - 1) do
        case Enum.at(pattern, rem(i, length(pattern))) do
          "L" -> long
          "S" -> short
        end
      end

    List.update_at(sizes, -1, fn _ -> long end)
  end

  # ---------------------------------------------------------------------------
  # Attention forward (GQA + QK norm + sliding window)
  # ---------------------------------------------------------------------------

  defn attention_forward(x, block_params, v_extra, cos, sin, opts \\ []) do
    n_head = opts[:n_head]
    n_kv_head = opts[:n_kv_head]
    window_size = opts[:window_size]

    dims = get_attn_dims(x, n_head, n_kv_head)

    q = Nx.dot(x, [-1], block_params.c_q, [0])
    k = Nx.dot(x, [-1], block_params.c_k, [0])
    v = Nx.dot(x, [-1], block_params.c_v, [0])

    q = reshape_heads(q, dims.batch, dims.seq_len, n_head, dims.head_dim)
    k = reshape_heads(k, dims.batch, dims.seq_len, n_kv_head, dims.head_dim)
    v = reshape_heads(v, dims.batch, dims.seq_len, n_kv_head, dims.head_dim)

    # Value embedding addition (v_extra is zeros when layer has no VE)
    v = Nx.add(v, v_extra)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    q = rms_norm(q)
    k = rms_norm(k)

    # GQA: expand K/V heads to match Q heads
    {k, v} = expand_kv_heads(k, v, dims)

    # (B, T, H, D) -> (B, H, T, D) for matmul
    q = Nx.transpose(q, axes: [0, 2, 1, 3])
    k = Nx.transpose(k, axes: [0, 2, 1, 3])
    v = Nx.transpose(v, axes: [0, 2, 1, 3])

    scale = compute_scale(dims)
    scores = Nx.multiply(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    scores = apply_causal_window_mask(scores, dims.seq_len, window_size)

    attn = stable_softmax(scores)

    y = Nx.dot(attn, [3], [0, 1], v, [2], [0, 1])
    y = Nx.transpose(y, axes: [0, 2, 1, 3])
    y = Nx.reshape(y, {dims.batch, dims.seq_len, n_head * dims.head_dim})

    Nx.dot(y, [-1], block_params.c_proj, [0])
  end

  deftransform get_attn_dims(x, n_head, n_kv_head) do
    {batch, seq_len, n_embd} = Nx.shape(x)
    head_dim = div(n_embd, n_head)
    %{batch: batch, seq_len: seq_len, n_embd: n_embd, head_dim: head_dim,
      n_head: n_head, n_kv_head: n_kv_head, repeat: div(n_head, n_kv_head)}
  end

  deftransform reshape_heads(x, batch, seq_len, n_heads, head_dim) do
    Nx.reshape(x, {batch, seq_len, n_heads, head_dim})
  end

  deftransform compute_scale(dims) do
    Nx.rsqrt(Nx.tensor(dims.head_dim * 1.0, type: :f32))
  end

  @doc false
  def compute_v_extra(x, idx, ve, gate_w, n_kv_head, head_dim) do
    ve_looked = Nx.take(ve, idx, axis: 0)
    {batch, seq_len, _} = Nx.shape(ve_looked)
    ve_looked = Nx.reshape(ve_looked, {batch, seq_len, n_kv_head, head_dim})
    gate_channels = Nx.axis_size(gate_w, 0)
    x_slice = Nx.slice_along_axis(x, 0, gate_channels, axis: 2)
    gate = Nx.multiply(2.0, Nx.sigmoid(Nx.dot(x_slice, [-1], gate_w, [0])))
    Nx.multiply(Nx.new_axis(gate, -1), ve_looked)
  end

  deftransform expand_kv_heads(k, v, %{n_head: nh, n_kv_head: nkv} = _dims) when nh == nkv do
    {k, v}
  end
  deftransform expand_kv_heads(k, v, dims) do
    %{batch: b, seq_len: t, n_kv_head: nkv, head_dim: hd, repeat: rep} = dims
    k = k |> Nx.reshape({b, t, nkv, 1, hd}) |> Nx.broadcast({b, t, nkv, rep, hd}) |> Nx.reshape({b, t, nkv * rep, hd})
    v = v |> Nx.reshape({b, t, nkv, 1, hd}) |> Nx.broadcast({b, t, nkv, rep, hd}) |> Nx.reshape({b, t, nkv * rep, hd})
    {k, v}
  end

  deftransform apply_causal_window_mask(scores, seq_len, window_size) do
    {batch, n_head, _, _} = Nx.shape(scores)
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    causal = Nx.greater_equal(rows, cols)
    window = Nx.greater(cols, Nx.subtract(rows, window_size))
    mask = Nx.logical_and(causal, window)
    mask = Nx.broadcast(Nx.reshape(mask, {1, 1, seq_len, seq_len}), {batch, n_head, seq_len, seq_len})
    neg_inf = Nx.broadcast(Nx.Constants.neg_infinity(:f32), {batch, n_head, seq_len, seq_len})
    Nx.select(mask, scores, neg_inf)
  end

  defn stable_softmax(x) do
    max = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    exp = Nx.exp(Nx.subtract(x, max))
    Nx.divide(exp, Nx.sum(exp, axes: [-1], keep_axes: true))
  end

  # ---------------------------------------------------------------------------
  # MLP (ReLU²)
  # ---------------------------------------------------------------------------

  defn mlp_forward(x, block_params) do
    h = Nx.dot(x, [-1], block_params.c_fc, [0])
    h = Nx.multiply(Nx.max(h, 0.0), Nx.max(h, 0.0))
    Nx.dot(h, [-1], block_params.c_proj_mlp, [0])
  end

  # ---------------------------------------------------------------------------
  # Block forward (regular fn to handle VE dispatch, calls defn internals)
  # ---------------------------------------------------------------------------

  def block_forward(x, block_params, ve, idx, cos, sin, opts) do
    n_kv_head = opts[:n_kv_head]
    n_head = opts[:n_head]
    head_dim = div(Nx.axis_size(x, 2), n_head)
    {batch, seq_len, _} = Nx.shape(x)

    x_normed = rms_norm(x)

    v_extra =
      case ve do
        :none ->
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {batch, seq_len, n_kv_head, head_dim})
        ve_tensor ->
          compute_v_extra(x_normed, idx, ve_tensor, block_params.ve_gate, n_kv_head, head_dim)
      end

    attn_out = attention_forward(x_normed, block_params, v_extra, cos, sin, opts)
    x = Nx.add(x, attn_out)
    mlp_out = mlp_forward(rms_norm(x), block_params)
    Nx.add(x, mlp_out)
  end

  # ---------------------------------------------------------------------------
  # Full forward pass
  # ---------------------------------------------------------------------------

  def forward(idx, params, %__MODULE__{} = config) do
    %{n_layer: n, n_head: nh, n_kv_head: nkv, n_embd: d} = config
    head_dim = div(d, nh)
    seq_len = Nx.axis_size(idx, 1)
    window_sizes = compute_window_sizes(config)
    {cos, sin} = precompute_rope(seq_len, head_dim)

    x = Nx.take(params.wte, idx, axis: 0)
    x = rms_norm(x)
    x0 = x

    x =
      0..(n - 1)
      |> Enum.reduce(x, fn i, x ->
        rl = Nx.reshape(params.resid_lambdas[i], {1, 1, 1})
        xl = Nx.reshape(params.x0_lambdas[i], {1, 1, 1})
        x = Nx.add(Nx.multiply(rl, x), Nx.multiply(xl, x0))

        block_params = elem(params.blocks, i)
        ve = elem(params.value_embeds, i)
        ws = Enum.at(window_sizes, i)

        block_forward(x, block_params, ve, idx, cos, sin,
          n_head: nh, n_kv_head: nkv, window_size: ws)
      end)

    x = rms_norm(x)
    logits = Nx.dot(x, [-1], params.lm_head, [1])
    softcap_logits(logits)
  end

  defn softcap_logits(logits) do
    Nx.multiply(@softcap, Nx.tanh(Nx.divide(logits, @softcap)))
  end

  # ---------------------------------------------------------------------------
  # Loss (with optional mask for SFT)
  # ---------------------------------------------------------------------------

  defn cross_entropy_loss(logits, targets, mask \\ Nx.tensor(1.0)) do
    {batch, seq_len, vocab_size} = Nx.shape(logits)
    flat_logits = Nx.reshape(logits, {batch * seq_len, vocab_size})
    flat_targets = Nx.reshape(targets, {batch * seq_len})
    flat_mask = Nx.reshape(Nx.broadcast(mask, {batch, seq_len}), {batch * seq_len})

    log_probs = log_softmax(flat_logits)
    nll = gather_log_probs(log_probs, flat_targets)

    valid = Nx.multiply(Nx.not_equal(flat_targets, -1), flat_mask)
    nll = Nx.multiply(nll, valid)
    n_valid = Nx.max(Nx.sum(valid), 1)

    Nx.divide(Nx.negate(Nx.sum(nll)), n_valid)
  end

  defnp log_softmax(logits) do
    max = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max)
    Nx.subtract(shifted, Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true)))
  end

  deftransform gather_log_probs(log_probs, targets) do
    {n, vocab_size} = Nx.shape(log_probs)
    one_hot = Nx.equal(Nx.iota({n, vocab_size}, axis: 1), Nx.reshape(targets, {n, 1}))
    Nx.sum(Nx.multiply(log_probs, one_hot), axes: [-1])
  end

  # ---------------------------------------------------------------------------
  # Utilities
  # ---------------------------------------------------------------------------

  def count_params(params) do
    count_tensor_params(params)
  end

  defp count_tensor_params(%Nx.Tensor{} = t) do
    t |> Nx.shape() |> Tuple.to_list() |> Enum.product()
  end
  defp count_tensor_params(map) when is_map(map) do
    map |> Map.values() |> Enum.map(&count_tensor_params/1) |> Enum.sum()
  end
  defp count_tensor_params(tuple) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.map(&count_tensor_params/1) |> Enum.sum()
  end
  defp count_tensor_params(nil), do: 0
  defp count_tensor_params(_), do: 0
end
