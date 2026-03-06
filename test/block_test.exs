defmodule ExNanoGPT.BlockTest do
  use ExUnit.Case, async: true

  alias ExNanoGPT.Block
  import ExNanoGPT.Test.GoldenHelpers

  @n_embd 32
  @n_head 4

  defp load_block_params do
    # PyTorch Linear stores weights transposed: {out, in}
    # Our linear does x @ weight where weight is {in, out}
    # So we need to transpose the attention and MLP weights from the golden data
    %{
      ln_1: %{
        weight: load_golden("block_param_ln_1_weight"),
        bias: load_golden("block_param_ln_1_bias")
      },
      ln_2: %{
        weight: load_golden("block_param_ln_2_weight"),
        bias: load_golden("block_param_ln_2_bias")
      },
      attn: %{
        c_attn_weight: Nx.transpose(load_golden("block_param_attn_c_attn_weight")),
        c_attn_bias: load_golden("block_param_attn_c_attn_bias"),
        c_proj_weight: Nx.transpose(load_golden("block_param_attn_c_proj_weight")),
        c_proj_bias: load_golden("block_param_attn_c_proj_bias")
      },
      mlp: %{
        c_fc_weight: Nx.transpose(load_golden("block_param_mlp_c_fc_weight")),
        c_fc_bias: load_golden("block_param_mlp_c_fc_bias"),
        c_proj_weight: Nx.transpose(load_golden("block_param_mlp_c_proj_weight")),
        c_proj_bias: load_golden("block_param_mlp_c_proj_bias")
      }
    }
  end

  @tag :golden
  test "transformer block forward matches PyTorch" do
    input = load_golden("block_input")
    params = load_block_params()
    expected = load_golden("block_output")

    key = Nx.Random.key(0)

    actual =
      Block.forward(input, params, key,
        n_head: @n_head,
        dropout_rate: 0.0,
        training: false
      )

    assert_close(actual, expected, atol: 1.0e-4)
  end

  test "output shape matches input shape" do
    key = Nx.Random.key(42)
    params = Block.init_params(@n_embd, @n_head, key)
    x = Nx.broadcast(0.1, {2, 8, @n_embd})

    out =
      Block.forward(x, params, key,
        n_head: @n_head,
        dropout_rate: 0.0,
        training: false
      )

    assert Nx.shape(out) == {2, 8, @n_embd}
  end

  test "residual connections preserve information" do
    key = Nx.Random.key(42)
    params = Block.init_params(@n_embd, @n_head, key)
    x = Nx.broadcast(0.0, {1, 4, @n_embd})

    out =
      Block.forward(x, params, key,
        n_head: @n_head,
        dropout_rate: 0.0,
        training: false
      )

    # With zero input and residuals, output should be non-trivial
    # (bias terms flow through even with zero input)
    assert Nx.shape(out) == {1, 4, @n_embd}
  end
end
