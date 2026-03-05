defmodule ExNanoGPT.LayerNormTest do
  use ExUnit.Case, async: true

  alias ExNanoGPT.LayerNorm
  import ExNanoGPT.Test.GoldenHelpers

  test "layer norm with bias matches PyTorch" do
    input = load_golden("ln_input")
    weight = load_golden("ln_weight")
    bias = load_golden("ln_bias")
    expected = load_golden("ln_output")

    params = %{weight: weight, bias: bias}
    actual = LayerNorm.forward(input, params)

    assert_close(actual, expected, atol: 1.0e-5)
  end

  test "layer norm without bias" do
    input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
    params = LayerNorm.init_params(4, bias: false)
    output = LayerNorm.forward(input, params)

    mean = Nx.mean(output) |> Nx.to_number()
    assert abs(mean) < 1.0e-5, "output mean should be ~0, got #{mean}"
  end

  test "init_params creates correct shapes" do
    params = LayerNorm.init_params(384)
    assert Nx.shape(params.weight) == {384}
    assert Nx.shape(params.bias) == {384}

    params_no_bias = LayerNorm.init_params(384, bias: false)
    assert Nx.shape(params_no_bias.weight) == {384}
    refute Map.has_key?(params_no_bias, :bias)
  end
end
