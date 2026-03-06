defmodule ExNanoGPT.V2Compact do
  @moduledoc """
  Complete nanochat model in a single module (~200 lines of core logic).

  Mirrors nanochat/gpt.py. This is the expanded v2 codebase compressed into
  one file for side-by-side comparison with the Python original.

  Architecture: wte -> RMSNorm -> save x0 ->
    N x [λ_r·x + λ_0·x0, RMSNorm -> GQA(RoPE, QK-norm, sliding window, VE) + res, RMSNorm -> ReLU²MLP + res]
    -> RMSNorm -> lm_head (untied) -> softcap
  """

  import Nx.Defn

  @softcap 20.0
  @rope_base 10_000
  @ve_gate_channels 32

  defstruct [:seq_len, :vocab_size, :n_layer, :n_head, :n_kv_head, :n_embd, :window_pattern]

  # -- RMSNorm -----------------------------------------------------------------

  defn rms_norm(x) do
    var = Nx.mean(Nx.multiply(x, x), axes: [-1], keep_axes: true)
    Nx.divide(x, Nx.sqrt(Nx.add(var, 1.0e-5)))
  end

  # -- RoPE --------------------------------------------------------------------

  def precompute_rope(seq_len, head_dim) do
    half = div(head_dim, 2)
    ch = Nx.multiply(Nx.iota({half}, type: :f32), 2)
    inv = Nx.exp(Nx.multiply(-:math.log(@rope_base), Nx.divide(ch, head_dim)))
    pos = Nx.iota({seq_len}, type: :f32)
    freqs = Nx.multiply(Nx.new_axis(pos, 1), Nx.new_axis(inv, 0))

    {Nx.reshape(Nx.cos(freqs), {1, seq_len, 1, half}),
     Nx.reshape(Nx.sin(freqs), {1, seq_len, 1, half})}
  end

  defn apply_rope(x, cos, sin) do
    half = div(Nx.axis_size(x, 3), 2)
    x1 = Nx.slice_along_axis(x, 0, half, axis: 3)
    x2 = Nx.slice_along_axis(x, half, half, axis: 3)

    Nx.concatenate(
      [
        Nx.add(Nx.multiply(x1, cos), Nx.multiply(x2, sin)),
        Nx.add(Nx.multiply(x1, Nx.negate(sin)), Nx.multiply(x2, cos))
      ],
      axis: 3
    )
  end

  defn softmax(x) do
    max = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    exp = Nx.exp(Nx.subtract(x, max))
    Nx.divide(exp, Nx.sum(exp, axes: [-1], keep_axes: true))
  end

  # -- Attention (GQA + QK norm + sliding window) ------------------------------

  defn attention(x, params, v_extra, cos, sin, opts \\ []) do
    n_head = opts[:n_head]
    n_kv_head = opts[:n_kv_head]
    window_size = opts[:window_size]
    {batch, seq_len, n_embd} = Nx.shape(x)
    head_dim = div(n_embd, n_head)

    q = Nx.dot(x, [-1], params.c_q, [0]) |> Nx.reshape({batch, seq_len, n_head, head_dim})
    k = Nx.dot(x, [-1], params.c_k, [0]) |> Nx.reshape({batch, seq_len, n_kv_head, head_dim})
    v = Nx.dot(x, [-1], params.c_v, [0]) |> Nx.reshape({batch, seq_len, n_kv_head, head_dim})

    v = Nx.add(v, v_extra)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    q = rms_norm(q)
    k = rms_norm(k)

    {k, v} = expand_kv(k, v, n_head, n_kv_head)

    q = Nx.transpose(q, axes: [0, 2, 1, 3])
    k = Nx.transpose(k, axes: [0, 2, 1, 3])
    v = Nx.transpose(v, axes: [0, 2, 1, 3])

    scale = attn_scale(head_dim)
    scores = Nx.multiply(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)
    scores = causal_window_mask(scores, seq_len, window_size)
    attn = softmax(scores)

    y = Nx.dot(attn, [3], [0, 1], v, [2], [0, 1])
    y = y |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({batch, seq_len, n_head * head_dim})
    Nx.dot(y, [-1], params.c_proj, [0])
  end

  deftransform attn_scale(hd), do: Nx.rsqrt(Nx.tensor(hd * 1.0, type: :f32))

  deftransform expand_kv(k, v, nh, nkv) when nh == nkv, do: {k, v}

  deftransform expand_kv(k, v, nh, nkv) do
    {b, t, _, hd} = Nx.shape(k)
    rep = div(nh, nkv)

    k =
      k
      |> Nx.reshape({b, t, nkv, 1, hd})
      |> Nx.broadcast({b, t, nkv, rep, hd})
      |> Nx.reshape({b, t, nh, hd})

    v =
      v
      |> Nx.reshape({b, t, nkv, 1, hd})
      |> Nx.broadcast({b, t, nkv, rep, hd})
      |> Nx.reshape({b, t, nh, hd})

    {k, v}
  end

  deftransform causal_window_mask(scores, seq_len, window_size) do
    {b, h, _, _} = Nx.shape(scores)
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)

    mask =
      Nx.logical_and(
        Nx.greater_equal(rows, cols),
        Nx.greater(cols, Nx.subtract(rows, window_size))
      )

    mask = Nx.broadcast(Nx.reshape(mask, {1, 1, seq_len, seq_len}), {b, h, seq_len, seq_len})
    Nx.select(mask, scores, Nx.Constants.neg_infinity(:f32))
  end

  # -- MLP (ReLU²) -------------------------------------------------------------

  defn mlp(x, params) do
    h = Nx.dot(x, [-1], params.c_fc, [0])
    h = Nx.multiply(Nx.max(h, 0.0), Nx.max(h, 0.0))
    Nx.dot(h, [-1], params.c_proj_mlp, [0])
  end

  # -- Value embedding helper --------------------------------------------------

  def compute_v_extra(x, idx, ve, gate_w, n_kv_head, head_dim) do
    ve_looked = Nx.take(ve, idx, axis: 0)
    {batch, seq_len, _} = Nx.shape(ve_looked)
    ve_looked = Nx.reshape(ve_looked, {batch, seq_len, n_kv_head, head_dim})
    gc = Nx.axis_size(gate_w, 0)

    gate =
      Nx.multiply(
        2.0,
        Nx.sigmoid(Nx.dot(Nx.slice_along_axis(x, 0, gc, axis: 2), [-1], gate_w, [0]))
      )

    Nx.multiply(Nx.new_axis(gate, -1), ve_looked)
  end

  # -- Block -------------------------------------------------------------------

  def block(x, params, ve, idx, cos, sin, opts) do
    n_kv_head = opts[:n_kv_head]
    n_head = opts[:n_head]
    head_dim = div(Nx.axis_size(x, 2), n_head)
    {batch, seq_len, _} = Nx.shape(x)

    x_n = rms_norm(x)

    v_extra =
      case ve do
        :none -> Nx.broadcast(Nx.tensor(0.0, type: :f32), {batch, seq_len, n_kv_head, head_dim})
        ve_t -> compute_v_extra(x_n, idx, ve_t, params.ve_gate, n_kv_head, head_dim)
      end

    x = Nx.add(x, attention(x_n, params, v_extra, cos, sin, opts))
    Nx.add(x, mlp(rms_norm(x), params))
  end

  # -- Window sizes ------------------------------------------------------------

  def window_sizes(%__MODULE__{} = c) do
    pattern = c.window_pattern |> String.upcase() |> String.graphemes()
    long = c.seq_len
    short = div(long, 2)

    sizes =
      for i <- 0..(c.n_layer - 1),
          do: if(Enum.at(pattern, rem(i, length(pattern))) == "L", do: long, else: short)

    List.update_at(sizes, -1, fn _ -> long end)
  end

  def has_ve?(i, n), do: rem(i, 2) == rem(n - 1, 2)

  # -- Forward -----------------------------------------------------------------

  def forward(idx, params, %__MODULE__{} = config) do
    %{n_layer: n, n_head: nh, n_kv_head: nkv, n_embd: d} = config
    head_dim = div(d, nh)
    seq_len = Nx.axis_size(idx, 1)
    ws = window_sizes(config)
    {cos, sin} = precompute_rope(seq_len, head_dim)

    x = Nx.take(params.wte, idx, axis: 0) |> rms_norm()
    x0 = x

    x =
      Enum.reduce(0..(n - 1), x, fn i, x ->
        rl = Nx.reshape(params.resid_lambdas[i], {1, 1, 1})
        xl = Nx.reshape(params.x0_lambdas[i], {1, 1, 1})
        x = Nx.add(Nx.multiply(rl, x), Nx.multiply(xl, x0))

        block(x, elem(params.blocks, i), elem(params.value_embeds, i), idx, cos, sin,
          n_head: nh,
          n_kv_head: nkv,
          window_size: Enum.at(ws, i)
        )
      end)

    logits = Nx.dot(rms_norm(x), [-1], params.lm_head, [1])
    Nx.multiply(@softcap, Nx.tanh(Nx.divide(logits, @softcap)))
  end

  # -- Init params -------------------------------------------------------------

  def init_params(%__MODULE__{} = config, key) do
    %{vocab_size: v, n_embd: d, n_layer: n, n_head: nh, n_kv_head: nkv} = config
    hd = div(d, nh)
    kv = nkv * hd
    s = :math.sqrt(3) * :math.pow(d, -0.5)

    {wte, key} = Nx.Random.normal(key, 0.0, 1.0, shape: {v, d})
    {lm_head, key} = Nx.Random.normal(key, 0.0, 0.001, shape: {v, d})

    {blocks, key} =
      Enum.map_reduce(0..(n - 1), key, fn _i, k ->
        {c_q, k} = Nx.Random.uniform(k, -s, s, shape: {d, nh * hd})
        {c_k, k} = Nx.Random.uniform(k, -s, s, shape: {d, kv})
        {c_v, k} = Nx.Random.uniform(k, -s, s, shape: {d, kv})
        {c_fc, k} = Nx.Random.uniform(k, -s, s, shape: {d, 4 * d})

        {%{
           c_q: c_q,
           c_k: c_k,
           c_v: c_v,
           c_proj: Nx.broadcast(Nx.tensor(0.0, type: :f32), {nh * hd, d}),
           ve_gate: Nx.broadcast(Nx.tensor(0.0, type: :f32), {@ve_gate_channels, nkv}),
           c_fc: c_fc,
           c_proj_mlp: Nx.broadcast(Nx.tensor(0.0, type: :f32), {4 * d, d})
         }, k}
      end)

    {ves, _key} =
      Enum.map_reduce(0..(n - 1), key, fn i, k ->
        if has_ve?(i, n), do: Nx.Random.uniform(k, -s, s, shape: {v, kv}), else: {:none, k}
      end)

    %{
      wte: wte,
      lm_head: lm_head,
      resid_lambdas: Nx.broadcast(Nx.tensor(1.0, type: :f32), {n}),
      x0_lambdas: Nx.broadcast(Nx.tensor(0.1, type: :f32), {n}),
      blocks: List.to_tuple(blocks),
      value_embeds: List.to_tuple(ves)
    }
  end

  # -- Loss --------------------------------------------------------------------

  defn cross_entropy_loss(logits, targets, mask \\ Nx.tensor(1.0)) do
    {batch, seq_len, vocab_size} = Nx.shape(logits)
    flat_l = Nx.reshape(logits, {batch * seq_len, vocab_size})
    flat_t = Nx.reshape(targets, {batch * seq_len})
    flat_m = Nx.reshape(Nx.broadcast(mask, {batch, seq_len}), {batch * seq_len})
    lp = log_softmax(flat_l)
    nll = gather_lp(lp, flat_t)
    valid = Nx.multiply(Nx.not_equal(flat_t, -1), flat_m)
    Nx.divide(Nx.negate(Nx.sum(Nx.multiply(nll, valid))), Nx.max(Nx.sum(valid), 1))
  end

  defnp log_softmax(x) do
    max = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    s = Nx.subtract(x, max)
    Nx.subtract(s, Nx.log(Nx.sum(Nx.exp(s), axes: [-1], keep_axes: true)))
  end

  deftransform gather_lp(lp, targets) do
    {n, v} = Nx.shape(lp)
    oh = Nx.equal(Nx.iota({n, v}, axis: 1), Nx.reshape(targets, {n, 1}))
    Nx.sum(Nx.multiply(lp, oh), axes: [-1])
  end
end
