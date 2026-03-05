defmodule ExNanoGPT.SamplerTest do
  use ExUnit.Case, async: true

  alias ExNanoGPT.{Model, Sampler}

  @model_config %{
    vocab_size: 65,
    block_size: 16,
    n_layer: 1,
    n_head: 2,
    n_embd: 16,
    dropout: 0.0,
    bias: true
  }

  test "generate produces correct output shape" do
    key = Nx.Random.key(42)
    params = Model.init_params(@model_config, key)

    # Start with a sequence of length 3, generate 5 more tokens
    idx = Nx.tensor([[1, 5, 10]], type: :s32)
    result = Sampler.generate(idx, params, @model_config, key,
      max_new_tokens: 5,
      temperature: 1.0,
      top_k: nil
    )

    assert Nx.shape(result) == {1, 8}
    assert Nx.type(result) == {:s, 32}

    # First 3 tokens should be the original prompt
    original = Nx.slice_along_axis(result, 0, 3, axis: 1)
    assert Nx.equal(original, Nx.tensor([[1, 5, 10]])) |> Nx.all() |> Nx.to_number() == 1
  end

  test "generate with top_k restricts token range" do
    key = Nx.Random.key(123)
    params = Model.init_params(@model_config, key)

    idx = Nx.tensor([[1, 2, 3]], type: :s32)
    result = Sampler.generate(idx, params, @model_config, key,
      max_new_tokens: 10,
      temperature: 1.0,
      top_k: 5
    )

    assert Nx.shape(result) == {1, 13}
  end

  test "temperature=0.01 produces near-deterministic output" do
    key = Nx.Random.key(7)
    params = Model.init_params(@model_config, key)

    idx = Nx.tensor([[1, 2]], type: :s32)

    result1 = Sampler.generate(idx, params, @model_config, Nx.Random.key(100),
      max_new_tokens: 5, temperature: 0.01, top_k: nil)
    result2 = Sampler.generate(idx, params, @model_config, Nx.Random.key(200),
      max_new_tokens: 5, temperature: 0.01, top_k: nil)

    # With very low temperature, both runs should produce the same (greedy) output
    assert Nx.equal(result1, result2) |> Nx.all() |> Nx.to_number() == 1
  end

  test "apply_top_k zeros out low-probability tokens" do
    logits = Nx.tensor([[10.0, 5.0, 3.0, 1.0, 0.5]])

    filtered = Sampler.apply_top_k(logits, 2)

    # Top 2 values (10.0, 5.0) should be preserved
    assert Nx.to_number(filtered[0][0]) == 10.0
    assert Nx.to_number(filtered[0][1]) == 5.0

    assert Nx.to_number(filtered[0][2]) == :neg_infinity
    assert Nx.to_number(filtered[0][3]) == :neg_infinity
    assert Nx.to_number(filtered[0][4]) == :neg_infinity
  end

  test "generate crops context when exceeding block_size" do
    key = Nx.Random.key(99)
    config = %{@model_config | block_size: 4}
    params = Model.init_params(config, key)

    # Start with sequence longer than block_size
    idx = Nx.tensor([[1, 2, 3, 4, 5, 6]], type: :s32)
    result = Sampler.generate(idx, params, config, key,
      max_new_tokens: 3,
      temperature: 1.0,
      top_k: nil
    )

    # Should have original 6 + 3 new = 9 tokens
    assert Nx.shape(result) == {1, 9}
  end

  test "same seed produces same output (deterministic)" do
    key = Nx.Random.key(42)
    params = Model.init_params(@model_config, key)

    idx = Nx.tensor([[1, 2, 3]], type: :s32)

    result1 = Sampler.generate(idx, params, @model_config, Nx.Random.key(555),
      max_new_tokens: 10, temperature: 0.8, top_k: 10)
    result2 = Sampler.generate(idx, params, @model_config, Nx.Random.key(555),
      max_new_tokens: 10, temperature: 0.8, top_k: 10)

    assert Nx.equal(result1, result2) |> Nx.all() |> Nx.to_number() == 1
  end
end
