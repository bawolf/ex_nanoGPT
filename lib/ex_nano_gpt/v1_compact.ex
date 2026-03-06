defmodule ExNanoGPT.V1Compact do
  @moduledoc """
  Complete GPT-2 model in a single module (~120 lines of core logic).

  Mirrors nanoGPT/model.py. This is the expanded v1 codebase compressed into
  one file for side-by-side comparison with the Python original.

  Architecture: token+pos embed -> N x [LN->MHA+res, LN->MLP+res] -> LN -> lm_head (tied to wte)
  """

  import Nx.Defn

  # -- LayerNorm ---------------------------------------------------------------

  defn layer_norm(x, weight, bias) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    var = Nx.variance(x, axes: [-1], keep_axes: true)
    normalized = (x - mean) / Nx.sqrt(var + 1.0e-5)
    normalized * weight + bias
  end

  # -- GELU --------------------------------------------------------------------

  defn gelu(x), do: x * 0.5 * (1.0 + Nx.erf(x / Nx.sqrt(Nx.tensor(2.0, type: :f32))))

  # -- Attention ---------------------------------------------------------------

  defn attention(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b, opts \\ []) do
    n_head = opts[:n_head]
    {batch, seq_len, n_embd} = Nx.shape(x)
    head_dim = div(n_embd, n_head)

    qkv = Nx.dot(x, [-1], c_attn_w, [0]) + c_attn_b
    q = Nx.slice_along_axis(qkv, 0, n_embd, axis: 2)
    k = Nx.slice_along_axis(qkv, n_embd, n_embd, axis: 2)
    v = Nx.slice_along_axis(qkv, 2 * n_embd, n_embd, axis: 2)

    q = q |> Nx.reshape({batch, seq_len, n_head, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, n_head, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, n_head, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = attn_scale(head_dim)
    att = Nx.multiply(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)
    att = causal_mask(att, seq_len)
    att = softmax(att)

    y = Nx.dot(att, [3], [0, 1], v, [2], [0, 1])
    y = y |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({batch, seq_len, n_embd})
    Nx.dot(y, [-1], c_proj_w, [0]) + c_proj_b
  end

  deftransform attn_scale(head_dim), do: Nx.rsqrt(Nx.tensor(head_dim * 1.0, type: :f32))

  deftransform causal_mask(att, seq_len) do
    {b, h, _, _} = Nx.shape(att)
    rows = Nx.iota({b, h, seq_len, seq_len}, axis: 2)
    cols = Nx.iota({b, h, seq_len, seq_len}, axis: 3)
    mask = Nx.greater_equal(rows, cols)
    Nx.select(mask, att, Nx.Constants.neg_infinity(:f32))
  end

  defn softmax(x) do
    max = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    exp = Nx.exp(x - max)
    exp / Nx.sum(exp, axes: [-1], keep_axes: true)
  end

  # -- MLP ---------------------------------------------------------------------

  defn mlp(x, c_fc_w, c_fc_b, c_proj_w, c_proj_b) do
    h = gelu(Nx.dot(x, [-1], c_fc_w, [0]) + c_fc_b)
    Nx.dot(h, [-1], c_proj_w, [0]) + c_proj_b
  end

  # -- Block -------------------------------------------------------------------

  defn block(x, params, opts \\ []) do
    n_head = opts[:n_head]
    h = layer_norm(x, params.ln_1_w, params.ln_1_b)

    x =
      x +
        attention(h, params.c_attn_w, params.c_attn_b, params.c_proj_w, params.c_proj_b,
          n_head: n_head
        )

    h = layer_norm(x, params.ln_2_w, params.ln_2_b)
    x + mlp(h, params.c_fc_w, params.c_fc_b, params.c_proj_mlp_w, params.c_proj_mlp_b)
  end

  # -- Forward -----------------------------------------------------------------

  def forward(idx, params, config) do
    %{n_head: n_head, n_layer: n_layer} = config
    seq_len = Nx.axis_size(idx, 1)

    tok_emb = Nx.take(params.wte, idx, axis: 0)
    pos_emb = Nx.take(params.wpe, Nx.iota({seq_len}), axis: 0)
    x = Nx.add(tok_emb, pos_emb)

    x =
      Enum.reduce(0..(n_layer - 1), x, fn i, x ->
        block(x, elem(params.blocks, i), n_head: n_head)
      end)

    x = layer_norm(x, params.ln_f_w, params.ln_f_b)
    Nx.dot(x, [-1], params.wte, [-1])
  end

  # -- Init params -------------------------------------------------------------

  def init_params(config, key) do
    %{vocab_size: v, block_size: bs, n_layer: n, n_embd: d} = config

    {wte, key} = Nx.Random.normal(key, 0.0, 0.02, shape: {v, d})
    {wpe, key} = Nx.Random.normal(key, 0.0, 0.02, shape: {bs, d})

    {blocks, _key} =
      Enum.map_reduce(0..(n - 1), key, fn _i, k ->
        proj_std = 0.02 / :math.sqrt(2 * n)
        {c_attn_w, k} = Nx.Random.normal(k, 0.0, 0.02, shape: {d, 3 * d})
        {c_proj_w, k} = Nx.Random.normal(k, 0.0, proj_std, shape: {d, d})
        {c_fc_w, k} = Nx.Random.normal(k, 0.0, 0.02, shape: {d, 4 * d})
        {c_proj_mlp_w, k} = Nx.Random.normal(k, 0.0, proj_std, shape: {4 * d, d})

        block = %{
          ln_1_w: Nx.broadcast(1.0, {d}),
          ln_1_b: Nx.broadcast(0.0, {d}),
          ln_2_w: Nx.broadcast(1.0, {d}),
          ln_2_b: Nx.broadcast(0.0, {d}),
          c_attn_w: c_attn_w,
          c_attn_b: Nx.broadcast(0.0, {3 * d}),
          c_proj_w: c_proj_w,
          c_proj_b: Nx.broadcast(0.0, {d}),
          c_fc_w: c_fc_w,
          c_fc_b: Nx.broadcast(0.0, {4 * d}),
          c_proj_mlp_w: c_proj_mlp_w,
          c_proj_mlp_b: Nx.broadcast(0.0, {d})
        }

        {block, k}
      end)

    %{
      wte: wte,
      wpe: wpe,
      blocks: List.to_tuple(blocks),
      ln_f_w: Nx.broadcast(1.0, {d}),
      ln_f_b: Nx.broadcast(0.0, {d})
    }
  end

  # -- Loss --------------------------------------------------------------------

  defn cross_entropy_loss(logits, targets) do
    {batch, seq_len, vocab_size} = Nx.shape(logits)
    flat_logits = Nx.reshape(logits, {batch * seq_len, vocab_size})
    flat_targets = Nx.reshape(targets, {batch * seq_len})
    log_probs = log_softmax(flat_logits)
    nll = gather_log_probs(log_probs, flat_targets)
    valid = Nx.not_equal(flat_targets, -1)
    -Nx.sum(nll * valid) / Nx.max(Nx.sum(valid), 1)
  end

  defnp log_softmax(x) do
    max = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    s = x - max
    s - Nx.log(Nx.sum(Nx.exp(s), axes: [-1], keep_axes: true))
  end

  deftransform gather_log_probs(log_probs, targets) do
    {n, vocab_size} = Nx.shape(log_probs)
    one_hot = Nx.equal(Nx.iota({n, vocab_size}, axis: 1), Nx.reshape(targets, {n, 1}))
    Nx.sum(Nx.multiply(log_probs, one_hot), axes: [-1])
  end
end
