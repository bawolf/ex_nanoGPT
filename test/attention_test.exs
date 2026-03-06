defmodule ExNanoGPT.AttentionTest do
  use ExUnit.Case, async: true

  alias ExNanoGPT.Attention
  import ExNanoGPT.Test.GoldenHelpers

  @n_head 4
  @n_embd 32

  defp load_attention_params do
    # Note: nanoGPT stores Linear weights transposed (PyTorch convention).
    # Our extract.py saves them as-is from the model, so c_attn_weight is {96, 32}
    # but our Elixir linear does x @ weight where weight is {n_embd, 3*n_embd}.
    # We need to transpose.
    c_attn_weight = load_golden("attn_c_attn_weight")
    c_attn_bias = load_golden("attn_c_attn_bias")
    c_proj_weight = load_golden("attn_c_proj_weight")
    c_proj_bias = load_golden("attn_c_proj_bias")

    %{
      c_attn_weight: c_attn_weight,
      c_attn_bias: c_attn_bias,
      c_proj_weight: c_proj_weight,
      c_proj_bias: c_proj_bias
    }
  end

  test "full attention forward matches PyTorch" do
    input = load_golden("attn_input")
    params = load_attention_params()
    expected = load_golden("attn_output")

    key = Nx.Random.key(0)

    actual =
      Attention.forward(input, params, key,
        n_head: @n_head,
        dropout_rate: 0.0,
        training: false
      )

    assert_close(actual, expected, atol: 1.0e-5)
  end

  test "causal mask is lower triangular" do
    mask = Attention.causal_mask(4)

    expected =
      Nx.tensor(
        [
          [1, 0, 0, 0],
          [1, 1, 0, 0],
          [1, 1, 1, 0],
          [1, 1, 1, 1]
        ],
        type: :u8
      )

    assert Nx.equal(mask, expected) |> Nx.all() |> Nx.to_number() == 1
  end

  test "init_params creates correct shapes" do
    key = Nx.Random.key(42)
    params = Attention.init_params(@n_embd, @n_head, key)

    assert Nx.shape(params.c_attn_weight) == {32, 96}
    assert Nx.shape(params.c_attn_bias) == {96}
    assert Nx.shape(params.c_proj_weight) == {32, 32}
    assert Nx.shape(params.c_proj_bias) == {32}
  end

  test "init_params without bias omits bias tensors" do
    key = Nx.Random.key(42)
    params = Attention.init_params(@n_embd, @n_head, key, bias: false)

    assert Map.has_key?(params, :c_attn_weight)
    refute Map.has_key?(params, :c_attn_bias)
    assert Map.has_key?(params, :c_proj_weight)
    refute Map.has_key?(params, :c_proj_bias)
  end

  test "output shape matches input shape" do
    key = Nx.Random.key(42)
    params = Attention.init_params(@n_embd, @n_head, key)
    x = Nx.broadcast(0.0, {2, 8, @n_embd})

    out =
      Attention.forward(x, params, key,
        n_head: @n_head,
        dropout_rate: 0.0,
        training: false
      )

    assert Nx.shape(out) == {2, 8, @n_embd}
  end
end
