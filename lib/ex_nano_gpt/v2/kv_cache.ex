defmodule ExNanoGPT.V2.KVCache do
  @moduledoc """
  KV Cache for fast autoregressive inference.

  Mirrors nanochat's KVCache: pre-allocates K/V tensors per layer,
  tracks current position, and supports incremental updates.

  During generation, instead of recomputing attention over all previous tokens,
  we cache the K and V projections and only compute the new token's Q/K/V.
  """

  defstruct [:k_cache, :v_cache, :pos, :max_seq, :n_layers]

  @doc """
  Create a new empty KV cache.

  ## Options
    * `:batch_size` - batch dimension
    * `:n_layers` - number of transformer layers
    * `:n_kv_head` - number of K/V heads (GQA)
    * `:head_dim` - dimension per head
    * `:max_seq` - maximum sequence length to cache
  """
  def new(opts) do
    batch = opts[:batch_size]
    n_layers = opts[:n_layers]
    n_kv_head = opts[:n_kv_head]
    head_dim = opts[:head_dim]
    max_seq = opts[:max_seq]

    shape = {batch, max_seq, n_kv_head, head_dim}
    k_cache = for _ <- 0..(n_layers - 1), do: Nx.broadcast(Nx.tensor(0.0, type: :f32), shape)
    v_cache = for _ <- 0..(n_layers - 1), do: Nx.broadcast(Nx.tensor(0.0, type: :f32), shape)

    %__MODULE__{
      k_cache: List.to_tuple(k_cache),
      v_cache: List.to_tuple(v_cache),
      pos: 0,
      max_seq: max_seq,
      n_layers: n_layers
    }
  end

  @doc "Get the K and V cache tensors for a specific layer."
  def get_layer(%__MODULE__{} = cache, layer_idx) do
    {elem(cache.k_cache, layer_idx), elem(cache.v_cache, layer_idx)}
  end

  @doc """
  Update the cache for a specific layer with new K/V values.

  `new_k` and `new_v` have shape `{batch, new_tokens, n_kv_head, head_dim}`.
  They are written into the cache starting at the current position.
  """
  def update(%__MODULE__{} = cache, layer_idx, new_k, new_v) do
    pos = cache.pos
    new_tokens = Nx.axis_size(new_k, 1)

    k = elem(cache.k_cache, layer_idx)
    v = elem(cache.v_cache, layer_idx)

    k = Nx.put_slice(k, [0, pos, 0, 0], new_k)
    v = Nx.put_slice(v, [0, pos, 0, 0], new_v)

    k_cache = put_elem(cache.k_cache, layer_idx, k)
    v_cache = put_elem(cache.v_cache, layer_idx, v)

    new_pos = if layer_idx == cache.n_layers - 1, do: pos + new_tokens, else: cache.pos
    %{cache | k_cache: k_cache, v_cache: v_cache, pos: new_pos}
  end

  @doc "Current position in the cache (number of tokens processed)."
  def get_pos(%__MODULE__{pos: pos}), do: pos

  @doc "Reset the cache to empty."
  def reset(%__MODULE__{} = cache) do
    %{cache | pos: 0}
  end
end
