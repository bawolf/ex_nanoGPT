defmodule ExNanoGPT.V2CompactTest do
  use ExUnit.Case, async: false

  alias ExNanoGPT.V2.Model
  alias ExNanoGPT.V2Compact
  alias ExNanoGPT.Test.GoldenHelpers

  @expanded_config %Model{
    sequence_len: 32,
    vocab_size: 64,
    n_layer: 2,
    n_head: 4,
    n_kv_head: 2,
    n_embd: 64,
    window_pattern: "SL"
  }

  @compact_config %V2Compact{
    seq_len: 32,
    vocab_size: 64,
    n_layer: 2,
    n_head: 4,
    n_kv_head: 2,
    n_embd: 64,
    window_pattern: "SL"
  }

  setup do
    key = Nx.Random.key(99)
    # Use same key for both so PRNG sequence is identical
    expanded_params = Model.init_params(@expanded_config, key)
    compact_params = V2Compact.init_params(@compact_config, key)
    idx = Nx.tensor([[5, 12, 3, 8, 15, 20, 1, 9]], type: :s64)
    %{expanded_params: expanded_params, compact_params: compact_params, idx: idx}
  end

  describe "component equivalence" do
    test "rms_norm matches" do
      x = Nx.tensor([[10.0, 20.0, 30.0, 40.0], [-1.0, 0.0, 1.0, 2.0]])
      GoldenHelpers.assert_close(V2Compact.rms_norm(x), Model.rms_norm(x), atol: 1.0e-6)
    end

    test "precompute_rope matches" do
      {ec, es} = Model.precompute_rope(16, 8)
      {cc, cs} = V2Compact.precompute_rope(16, 8)
      GoldenHelpers.assert_close(cc, ec, atol: 1.0e-6)
      GoldenHelpers.assert_close(cs, es, atol: 1.0e-6)
    end

    test "apply_rope matches" do
      {cos, sin} = Model.precompute_rope(4, 8)
      x = Nx.iota({1, 4, 2, 8}, type: :f32)

      GoldenHelpers.assert_close(
        V2Compact.apply_rope(x, cos, sin),
        Model.apply_rope(x, cos, sin),
        atol: 1.0e-5
      )
    end

    test "window_sizes matches" do
      expanded = Model.compute_window_sizes(@expanded_config)
      compact = V2Compact.window_sizes(@compact_config)
      assert expanded == compact
    end

    test "has_ve? matches" do
      for i <- 0..3, n <- 2..4 do
        assert V2Compact.has_ve?(i, n) == Model.has_ve?(i, n),
               "has_ve?(#{i}, #{n}) mismatch"
      end
    end
  end

  describe "forward pass equivalence" do
    test "compact forward matches expanded forward (shared params)", ctx do
      expanded_logits = Model.forward(ctx.idx, ctx.expanded_params, @expanded_config)
      compact_logits = V2Compact.forward(ctx.idx, ctx.compact_params, @compact_config)

      GoldenHelpers.assert_close(compact_logits, expanded_logits, atol: 1.0e-4, rtol: 1.0e-4)
    end

    test "compact forward matches expanded forward (converted params)", ctx do
      # Also test with EXACT same param tensors to rule out init differences
      compact_logits = V2Compact.forward(ctx.idx, ctx.expanded_params, @compact_config)
      expanded_logits = Model.forward(ctx.idx, ctx.expanded_params, @expanded_config)

      GoldenHelpers.assert_close(compact_logits, expanded_logits, atol: 1.0e-4, rtol: 1.0e-4)
    end
  end

  describe "cross_entropy_loss equivalence" do
    test "compact loss matches expanded loss" do
      logits = Nx.tensor([[[2.0, 1.0, 0.5], [0.1, 2.0, 0.3]]])
      targets = Nx.tensor([[0, 1]])

      expanded_loss = Model.cross_entropy_loss(logits, targets) |> Nx.to_number()
      compact_loss = V2Compact.cross_entropy_loss(logits, targets) |> Nx.to_number()

      assert abs(expanded_loss - compact_loss) < 1.0e-5
    end

    test "compact masked loss matches expanded" do
      logits = Nx.tensor([[[2.0, 1.0, 0.5], [0.1, 2.0, 0.3]]])
      targets = Nx.tensor([[0, 1]])
      mask = Nx.tensor([[1, 0]])

      expanded_loss = Model.cross_entropy_loss(logits, targets, mask) |> Nx.to_number()
      compact_loss = V2Compact.cross_entropy_loss(logits, targets, mask) |> Nx.to_number()

      assert abs(expanded_loss - compact_loss) < 1.0e-5
    end
  end
end
