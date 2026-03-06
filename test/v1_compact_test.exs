defmodule ExNanoGPT.V1CompactTest do
  use ExUnit.Case, async: false

  alias ExNanoGPT.{Model, V1Compact}
  alias ExNanoGPT.Test.GoldenHelpers

  @config %{
    vocab_size: 65,
    block_size: 16,
    n_layer: 2,
    n_head: 4,
    n_embd: 32,
    dropout: 0.0,
    bias: true
  }

  setup do
    key = Nx.Random.key(42)
    expanded_params = Model.init_params(@config, key)
    compact_params = convert_to_compact(expanded_params)
    idx = Nx.tensor([[1, 5, 10, 20, 30, 15, 8, 3]], type: :s32)
    %{expanded_params: expanded_params, compact_params: compact_params, idx: idx, key: key}
  end

  describe "forward pass equivalence" do
    test "compact forward matches expanded forward", ctx do
      expanded_logits = Model.forward(ctx.idx, ctx.expanded_params, @config, ctx.key)
      compact_logits = V1Compact.forward(ctx.idx, ctx.compact_params, @config)

      # Expanded returns last-position only {batch, 1, vocab}, compact returns all {batch, seq, vocab}
      # Compare last position
      compact_last = Nx.slice_along_axis(compact_logits, Nx.axis_size(ctx.idx, 1) - 1, 1, axis: 1)
      GoldenHelpers.assert_close(compact_last, expanded_logits, atol: 1.0e-4, rtol: 1.0e-4)
    end
  end

  describe "cross_entropy_loss equivalence" do
    test "compact loss matches expanded loss" do
      logits = Nx.tensor([[[2.0, 1.0, 0.5], [0.1, 2.0, 0.3]]])
      targets = Nx.tensor([[0, 1]])

      expanded_loss = Model.cross_entropy_loss(logits, targets) |> Nx.to_number()
      compact_loss = V1Compact.cross_entropy_loss(logits, targets) |> Nx.to_number()

      assert abs(expanded_loss - compact_loss) < 1.0e-5
    end

    test "compact loss ignores target=-1 same as expanded" do
      logits = Nx.tensor([[[2.0, 1.0, 0.5], [0.1, 2.0, 0.3]]])
      targets = Nx.tensor([[0, -1]])

      expanded_loss = Model.cross_entropy_loss(logits, targets) |> Nx.to_number()
      compact_loss = V1Compact.cross_entropy_loss(logits, targets) |> Nx.to_number()

      assert abs(expanded_loss - compact_loss) < 1.0e-5
    end
  end

  describe "layer_norm equivalence" do
    test "compact layer_norm matches expanded" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, -5.0, 0.0, 3.0]])
      w = Nx.tensor([1.0, 2.0, 0.5, 1.5])
      b = Nx.tensor([0.1, -0.1, 0.0, 0.2])

      expanded = ExNanoGPT.LayerNorm.forward(x, %{weight: w, bias: b})
      compact = V1Compact.layer_norm(x, w, b)

      GoldenHelpers.assert_close(compact, expanded, atol: 1.0e-5)
    end
  end

  # Convert expanded v1 params to compact flat format
  defp convert_to_compact(params) do
    blocks =
      params.blocks
      |> Tuple.to_list()
      |> Enum.map(fn bp ->
        %{
          ln_1_w: bp.ln_1.weight,
          ln_1_b: bp.ln_1.bias,
          ln_2_w: bp.ln_2.weight,
          ln_2_b: bp.ln_2.bias,
          c_attn_w: bp.attn.c_attn_weight,
          c_attn_b: bp.attn.c_attn_bias,
          c_proj_w: bp.attn.c_proj_weight,
          c_proj_b: bp.attn.c_proj_bias,
          c_fc_w: bp.mlp.c_fc_weight,
          c_fc_b: bp.mlp.c_fc_bias,
          c_proj_mlp_w: bp.mlp.c_proj_weight,
          c_proj_mlp_b: bp.mlp.c_proj_bias
        }
      end)
      |> List.to_tuple()

    %{
      wte: params.wte,
      wpe: params.wpe,
      blocks: blocks,
      ln_f_w: params.ln_f.weight,
      ln_f_b: params.ln_f.bias
    }
  end
end
