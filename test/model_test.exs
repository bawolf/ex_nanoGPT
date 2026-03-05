defmodule ExNanoGPT.ModelTest do
  use ExUnit.Case, async: true

  alias ExNanoGPT.Model
  import ExNanoGPT.Test.GoldenHelpers

  @config %{
    vocab_size: 65,
    block_size: 16,
    n_layer: 2,
    n_head: 4,
    n_embd: 32,
    dropout: 0.0,
    bias: true
  }

  defp load_gpt_params do
    # Load from golden data, transposing linear weights as needed
    # (PyTorch stores Linear as {out_features, in_features})
    load_block = fn i ->
      prefix = "gpt_param_transformer_h_#{i}"

      %{
        ln_1: %{
          weight: load_golden("#{prefix}_ln_1_weight"),
          bias: load_golden("#{prefix}_ln_1_bias")
        },
        ln_2: %{
          weight: load_golden("#{prefix}_ln_2_weight"),
          bias: load_golden("#{prefix}_ln_2_bias")
        },
        attn: %{
          c_attn_weight: Nx.transpose(load_golden("#{prefix}_attn_c_attn_weight")),
          c_attn_bias: load_golden("#{prefix}_attn_c_attn_bias"),
          c_proj_weight: Nx.transpose(load_golden("#{prefix}_attn_c_proj_weight")),
          c_proj_bias: load_golden("#{prefix}_attn_c_proj_bias")
        },
        mlp: %{
          c_fc_weight: Nx.transpose(load_golden("#{prefix}_mlp_c_fc_weight")),
          c_fc_bias: load_golden("#{prefix}_mlp_c_fc_bias"),
          c_proj_weight: Nx.transpose(load_golden("#{prefix}_mlp_c_proj_weight")),
          c_proj_bias: load_golden("#{prefix}_mlp_c_proj_bias")
        }
      }
    end

    %{
      wte: load_golden("gpt_param_transformer_wte_weight"),
      wpe: load_golden("gpt_param_transformer_wpe_weight"),
      blocks: {load_block.(0), load_block.(1)},
      ln_f: %{
        weight: load_golden("gpt_param_transformer_ln_f_weight"),
        bias: load_golden("gpt_param_transformer_ln_f_bias")
      }
    }
  end

  test "GPT training forward matches PyTorch" do
    input = load_golden("gpt_input")
    params = load_gpt_params()
    expected_logits = load_golden("gpt_logits_training")

    key = Nx.Random.key(0)
    actual_logits = Model.forward(input, params, @config, key, training: true)

    assert Nx.shape(actual_logits) == {2, 8, 65}
    assert_close(actual_logits, expected_logits, atol: 1.0e-4)
  end

  test "GPT inference forward matches PyTorch" do
    input = load_golden("gpt_input")
    params = load_gpt_params()
    expected_logits = load_golden("gpt_logits_inference")

    key = Nx.Random.key(0)
    actual_logits = Model.forward(input, params, @config, key, training: false)

    assert Nx.shape(actual_logits) == {2, 1, 65}
    assert_close(actual_logits, expected_logits, atol: 1.0e-4)
  end

  test "cross-entropy loss matches PyTorch" do
    input = load_golden("gpt_input")
    targets = load_golden("gpt_targets")
    params = load_gpt_params()
    expected_loss = load_golden("gpt_loss")

    key = Nx.Random.key(0)
    logits = Model.forward(input, params, @config, key, training: true)
    actual_loss = Model.cross_entropy_loss(logits, targets)

    expected_scalar = Nx.squeeze(expected_loss)
    assert_close(actual_loss, expected_scalar, atol: 1.0e-4)
  end

  test "init_params creates correct structure" do
    key = Nx.Random.key(42)
    params = Model.init_params(@config, key)

    assert Nx.shape(params.wte) == {65, 32}
    assert Nx.shape(params.wpe) == {16, 32}
    assert tuple_size(params.blocks) == 2
    assert Map.has_key?(params.ln_f, :weight)
  end

  test "count_params reports correct count" do
    key = Nx.Random.key(42)
    params = Model.init_params(@config, key)
    count = Model.count_params(params)

    assert is_integer(count)
    assert count > 0
  end
end
