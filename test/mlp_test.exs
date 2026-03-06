defmodule ExNanoGPT.MLPTest do
  use ExUnit.Case, async: true

  alias ExNanoGPT.MLP
  import ExNanoGPT.Test.GoldenHelpers

  @n_embd 32

  @tag :golden
  test "MLP forward matches PyTorch" do
    input = load_golden("mlp_input")
    # extract.py saves weights in our convention {in, out} already
    c_fc_weight = load_golden("mlp_c_fc_weight")
    c_fc_bias = load_golden("mlp_c_fc_bias")
    c_proj_weight = load_golden("mlp_c_proj_weight")
    c_proj_bias = load_golden("mlp_c_proj_bias")
    expected = load_golden("mlp_output")

    params = %{
      c_fc_weight: c_fc_weight,
      c_fc_bias: c_fc_bias,
      c_proj_weight: c_proj_weight,
      c_proj_bias: c_proj_bias
    }

    key = Nx.Random.key(0)
    actual = MLP.forward(input, params, key, dropout_rate: 0.0, training: false)

    assert_close(actual, expected, atol: 1.0e-5)
  end

  test "output shape matches input shape" do
    key = Nx.Random.key(42)
    params = MLP.init_params(@n_embd, key)
    x = Nx.broadcast(0.0, {2, 8, @n_embd})

    out = MLP.forward(x, params, key, dropout_rate: 0.0, training: false)
    assert Nx.shape(out) == {2, 8, @n_embd}
  end

  test "init_params creates correct shapes" do
    key = Nx.Random.key(42)
    params = MLP.init_params(@n_embd, key)

    assert Nx.shape(params.c_fc_weight) == {32, 128}
    assert Nx.shape(params.c_fc_bias) == {128}
    assert Nx.shape(params.c_proj_weight) == {128, 32}
    assert Nx.shape(params.c_proj_bias) == {32}
  end
end
