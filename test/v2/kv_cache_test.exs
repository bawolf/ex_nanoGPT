defmodule ExNanoGPT.V2.KVCacheTest do
  use ExUnit.Case, async: false

  alias ExNanoGPT.V2.{Model, KVCache}
  alias ExNanoGPT.Test.GoldenHelpers

  @config %Model{
    sequence_len: 32,
    vocab_size: 64,
    n_layer: 2,
    n_head: 4,
    n_kv_head: 2,
    n_embd: 32,
    window_pattern: "SL"
  }

  setup do
    params = Model.init_params(@config, Nx.Random.key(42))
    %{params: params}
  end

  describe "KVCache" do
    test "new/1 creates cache with correct shapes" do
      cache =
        KVCache.new(
          batch_size: 1,
          n_layers: 2,
          n_kv_head: 2,
          head_dim: 8,
          max_seq: 32
        )

      assert cache.pos == 0
      assert cache.n_layers == 2
      assert cache.max_seq == 32
      assert tuple_size(cache.k_cache) == 2

      {k, v} = KVCache.get_layer(cache, 0)
      assert Nx.shape(k) == {1, 32, 2, 8}
      assert Nx.shape(v) == {1, 32, 2, 8}
    end

    test "update/4 writes into correct position" do
      cache =
        KVCache.new(
          batch_size: 1,
          n_layers: 1,
          n_kv_head: 2,
          head_dim: 4,
          max_seq: 8
        )

      new_k = Nx.broadcast(Nx.tensor(1.0), {1, 2, 2, 4})
      new_v = Nx.broadcast(Nx.tensor(2.0), {1, 2, 2, 4})
      cache = KVCache.update(cache, 0, new_k, new_v)

      {k, _v} = KVCache.get_layer(cache, 0)
      assert Nx.to_number(k[[0, 0, 0, 0]]) == 1.0
      assert Nx.to_number(k[[0, 1, 0, 0]]) == 1.0
      assert Nx.to_number(k[[0, 2, 0, 0]]) == 0.0
    end
  end

  describe "forward_cached vs forward" do
    test "single-pass forward matches incremental cached forward", %{params: params} do
      seq = Nx.tensor([[5, 12, 3, 8, 15, 20, 1, 9]])

      # Full forward (no cache) -- get logits for last position
      full_logits = Model.forward(seq, params, @config)
      last_logits_full = full_logits[[0, 7, ..]]

      # Cached forward: process all tokens at once
      head_dim = div(@config.n_embd, @config.n_head)

      cache =
        KVCache.new(
          batch_size: 1,
          n_layers: @config.n_layer,
          n_kv_head: @config.n_kv_head,
          head_dim: head_dim,
          max_seq: @config.sequence_len
        )

      {cached_logits, _cache} = Model.forward_cached(seq, params, @config, cache)
      last_logits_cached = cached_logits[[0, 0, ..]]

      GoldenHelpers.assert_close(last_logits_cached, last_logits_full, atol: 1.0e-4, rtol: 1.0e-4)
    end

    test "incremental token-by-token matches full forward", %{params: params} do
      tokens = [5, 12, 3, 8]
      full_seq = Nx.tensor([tokens])

      full_logits = Model.forward(full_seq, params, @config)
      last_logits_full = full_logits[[0, 3, ..]]

      head_dim = div(@config.n_embd, @config.n_head)

      cache =
        KVCache.new(
          batch_size: 1,
          n_layers: @config.n_layer,
          n_kv_head: @config.n_kv_head,
          head_dim: head_dim,
          max_seq: @config.sequence_len
        )

      # Process tokens one at a time
      {_logits, cache} = Model.forward_cached(Nx.tensor([[5]]), params, @config, cache)
      {_logits, cache} = Model.forward_cached(Nx.tensor([[12]]), params, @config, cache)
      {_logits, cache} = Model.forward_cached(Nx.tensor([[3]]), params, @config, cache)

      {last_logits_cached, _cache} =
        Model.forward_cached(Nx.tensor([[8]]), params, @config, cache)

      last_logits_cached = last_logits_cached[[0, 0, ..]]

      GoldenHelpers.assert_close(last_logits_cached, last_logits_full, atol: 1.0e-3, rtol: 1.0e-3)
    end
  end
end
